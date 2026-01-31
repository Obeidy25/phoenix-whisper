#!/usr/bin/env python3
# ==============================================================================
# main.py - Main Entry Point and Orchestrator
# ==============================================================================
"""
Central coordination for video transcription with signal handling.
"""

import sys
import signal
import threading
import queue
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple

# Graceful import handling
def check_imports():
    """Check for required dependencies and provide helpful errors."""
    missing = []
    
    try:
        import rich
    except ImportError:
        missing.append(("rich", "pip install rich"))
    
    try:
        import whisper
    except ImportError:
        missing.append(("openai-whisper", "pip install openai-whisper"))
    
    if missing:
        print("\n" + "=" * 60)
        print("  ERROR: Missing Required Dependencies")
        print("=" * 60 + "\n")
        for pkg, cmd in missing:
            print(f"  [X] {pkg} is not installed")
            print(f"      Fix: {cmd}\n")
        print("  Or run: python run.py (auto-installs dependencies)")
        print("  Or run: setup.bat (Windows one-click setup)")
        print("\n" + "=" * 60 + "\n")
        sys.exit(1)

# Check dependencies before importing local modules
check_imports()

from .config import (
    get_chunk_duration, 
    PROGRESS_FILE_NAME, 
    APP_VERSION,
    CHUNK_DURATION_GPU,
    CHUNK_DURATION_STRONG_CPU,
    CHUNK_DURATION_WEAK_CPU,
    PARTIAL_SUBTITLE_SUFFIX,
    TEMP_CHUNK_PREFIX,
)
from .cli import parse_arguments
from .environment import DependencyChecker, get_hardware_info
from .file_handler import FileHandler
from .process_handler import ProcessHandler, TranscriptionResult
from .state_manager import StateManager, TranscriptionState
from .subtitle_parser import SubtitleParser
from .ui import TranscriptionUI


class TranscriptionApp:
    """Main application orchestrator."""
    
    def __init__(self):
        self.ui: Optional[TranscriptionUI] = None
        self.state_manager: Optional[StateManager] = None
        self.file_handler = FileHandler()
        self.process_handler: Optional[ProcessHandler] = None
        self.subtitle_parser = SubtitleParser()
        self._shutdown_requested = False
    
    def setup_signal_handler(self) -> None:
        """Setup graceful shutdown on Ctrl+C."""
        def handler(sig, frame):
            if self._shutdown_requested:
                self.ui.log_warning("Force quit requested. Exiting immediately...")
                sys.exit(1)
            self._shutdown_requested = True
            self.ui.log_warning("Shutdown signal received. Saving progress...")
            
            # Save state check
            if self.state_manager and self.state_manager.state:
                state = self.state_manager.state
                self.state_manager.save_progress(state)
                
                # Instant Gratification: Save partial results
                if state.completed_chunks:
                    try:
                        self.ui.log_info(f"Generating partial subtitle for {len(state.completed_chunks)} chunks...")
                        self.save_partial_results(state)
                    except Exception as e:
                        self.ui.log_error(f"Failed to save partial subtitle: {e}")
            
            self.ui.log_warning("Exiting gracefully. You can resume later.")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, handler)
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, handler)
    
    def should_stop(self) -> bool:
        """Check if shutdown was requested."""
        return self._shutdown_requested
    
    def save_partial_results(self, state: TranscriptionState, temp_dir: Path = None) -> Path:
        """
        v4.0: Save/update the unified partial SRT file.
        
        Creates or updates a single _IN_PROGRESS.srt file that contains
        all completed subtitles, merged and sorted by start time.
        
        Args:
            state: Current transcription state
            temp_dir: Temporary directory containing chunk SRT files
            
        Returns:
            Path to the partial subtitle file
        """
        if not state.completed_chunks:
            return None
            
        video_path = Path(state.video_path)
        
        # v4.0: Single unified file name
        partial_srt = video_path.parent / f"{video_path.stem}{PARTIAL_SUBTITLE_SUFFIX}.srt"
        
        # Build list of all chunk SRT paths
        if temp_dir is None:
            temp_dir = self.file_handler.get_temp_directory(video_path)
        
        # Create list of SRT file paths for all chunks
        srt_files = [
            temp_dir / f"{TEMP_CHUNK_PREFIX}{i:04d}.srt" 
            for i in range(state.total_chunks)
        ]
        
        # Get completed indices
        completed_indices = sorted(state.completed_chunks)
        
        # Use subtitle parser's update method for merge/sort
        self.subtitle_parser.update_partial_file(
            partial_path=partial_srt,
            srt_files=srt_files,
            chunk_duration=state.chunk_duration,
            completed_indices=completed_indices
        )
        
        self.ui.log_success(f"Live partial subtitle updated: [bold underline]{partial_srt.name}[/]")
        return partial_srt

    def check_dependencies(self) -> bool:
        """Verify all dependencies are available."""
        self.ui.log_info("Checking dependencies...")
        checker = DependencyChecker()
        all_valid, statuses = checker.validate_all()
        
        if all_valid:
            for name, status in statuses.items():
                self.ui.log_success(f"{name}: {status.version or 'OK'}")
            return True
        else:
            for name, status in statuses.items():
                if not status.available:
                    self.ui.show_error_panel(f"Missing: {name}", status.error_message)
            return False

    def _producer_task(
        self,
        video_path: Path,
        boundaries: List[Tuple[float, float]],
        output_dir: Path,
        chunk_queue: queue.Queue,
        completed_indices: set,
    ) -> None:
        """Background task to extract chunks and put them in a queue."""
        self.ui.log_info(f"Producer started: Processing {len(boundaries)} boundaries")
        
        for i, (start, end) in enumerate(boundaries):
            if self.should_stop():
                break
                
            # Skip if already completed (from resume)
            if i in completed_indices:
                continue
            
            # v3.1 Fast Resume: Check if chunk file already exists on disk
            # This avoids calling ffmpeg entirely for already-extracted chunks
            from .config import TEMP_CHUNK_PREFIX
            chunk_path = output_dir / f"{TEMP_CHUNK_PREFIX}{i:04d}.wav"
            if chunk_path.exists():
                chunk_queue.put((i, chunk_path))
                continue
                
            try:
                # Extract chunk using ffmpeg (only if not already on disk)
                chunk_path = self.file_handler.extract_chunk(
                    video_path=video_path,
                    start=start,
                    duration=end - start,
                    chunk_index=i,
                    output_dir=output_dir
                )
                chunk_queue.put((i, chunk_path))
            except Exception as e:
                self.ui.log_error(f"Producer error on chunk {i}: {e}")
                
        # Signal end of production
        chunk_queue.put(None)
        self.ui.log_info("Producer finished.")
    
    def progress_callback(self, completed: int, total: int, result: TranscriptionResult) -> None:
        """Handle progress updates."""
        if result.success:
            # v3.1: Removed individual chunk log messages to avoid out-of-order confusion
            # The progress bar is now the single source of truth for progress
            if self.state_manager.state:
                self.state_manager.state.mark_chunk_completed(result.chunk_index, result.srt_path)
                self.state_manager.save_progress(self.state_manager.state)
        else:
            # Keep error messages visible - these are important for debugging
            self.ui.log_error(f"Chunk {result.chunk_index + 1} failed: {result.error_message}")
            if self.state_manager.state:
                self.state_manager.state.mark_chunk_failed(result.chunk_index)
        
        self.ui.update_progress(completed)
    
    def run(self) -> int:
        """Run the transcription application."""
        import time  # v3.2: For timing instrumentation
        
        # Parse arguments
        args = parse_arguments()
        
        # Initialize UI
        self.ui = TranscriptionUI(verbose=args.verbose)
        self.ui.show_banner()
        
        # v3.2: Handle diagnostic mode
        if getattr(args, 'diagnose', False):
            from .diagnostics import run_diagnostics
            return run_diagnostics(self.ui.console)
        
        # Setup signal handler
        self.setup_signal_handler()
        
        # Check dependencies
        if not self.check_dependencies():
            self.ui.log_error("Please install missing dependencies and try again.")
            return 1
        
        # ----------------------------------------------------------------------
        # v3.0 Adaptive Performance Logic
        # ----------------------------------------------------------------------
        hw_info = get_hardware_info()
        self.ui.log_info(f"Hardware Detected: {hw_info.description}")
        if not hw_info.has_nvidia_gpu and hw_info.gpu_reason:
            self.ui.log_warning(f"GPU Status: {hw_info.gpu_reason}")
        
        # 1. Configure Workers
        if args.num_workers is None:
            if hw_info.has_nvidia_gpu:
                num_workers = 1
                self.ui.log_info(f"Auto-configured: 1 worker (GPU optimized)")
            else:
                num_workers = max(1, hw_info.cpu_cores // 2)
                self.ui.log_info(f"Auto-configured: {num_workers} workers (CPU balanced)")
        else:
            num_workers = args.num_workers
            
        # 2. Configure Chunk Duration
        if args.ram_profile is None:
            if hw_info.has_nvidia_gpu:
                chunk_duration = CHUNK_DURATION_GPU
                profile_name = "Adaptive (GPU)"
            elif hw_info.cpu_cores > 8:
                chunk_duration = CHUNK_DURATION_STRONG_CPU
                profile_name = "Adaptive (Strong CPU)"
            else:
                chunk_duration = CHUNK_DURATION_WEAK_CPU
                profile_name = "Adaptive (Standard CPU)"
            self.ui.log_info(f"Chunk Strategy: {profile_name} -> {chunk_duration}s chunks")
        else:
            chunk_duration = get_chunk_duration(args.ram_profile)
            self.ui.log_info(f"Using {args.ram_profile} RAM profile ({chunk_duration // 60} min chunks)")
        
        # Initialize process handler with timeout
        self.process_handler = ProcessHandler(num_workers=num_workers)
        if num_workers > 1:
            self.ui.log_info(f"Concurrent mode: {num_workers} workers")
        
        # Setup paths
        input_path = args.input.resolve()
        output_dir = args.output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        
        temp_dir = self.file_handler.get_temp_directory(input_path)
        progress_file = temp_dir / PROGRESS_FILE_NAME
        self.state_manager = StateManager(progress_file)
        
        # Check for existing progress
        state: Optional[TranscriptionState] = None
        # Note: We relax strict checking for v3 to allow resuming with different adaptive settings if consistent
        # For now, simplistic resume check
        if args.resume:
            state = self.state_manager.load_progress()
            if state and state.completed_chunks:
                self.ui.show_resume_info(len(state.completed_chunks), state.total_chunks)
                # Adopt chunk duration from state to ensure consistency
                if chunk_duration != state.chunk_duration:
                    self.ui.log_warning(f"Resuming with saved chunk duration {state.chunk_duration}s (overriding adaptive {chunk_duration}s)")
                    chunk_duration = state.chunk_duration 

        
        # Get video info
        # v3.2: Timing instrumentation
        analysis_start = time.perf_counter()
        try:
            with self.ui.show_status(f"Analyzing: {input_path.name}"):
                video_info = self.file_handler.get_video_info(input_path)
            duration_mins = video_info.duration / 60
            analysis_elapsed = time.perf_counter() - analysis_start
            self.ui.log_success(f"Analyzed: {duration_mins:.1f} minutes (took {analysis_elapsed:.2f}s)")
        except Exception as e:
            self.ui.log_error(f"Failed to analyze video: {e}")
            return 1
        
        # Calculate boundaries once
        boundaries = self.file_handler.calculate_chunks(video_info.duration, chunk_duration)
        total_chunks = len(boundaries)
        
        # Create/update state
        if state is None:
            state = self.state_manager.create_new_state(
                video_path=input_path,
                video_duration=video_info.duration,
                total_chunks=total_chunks,
                model=args.model,
                language=args.language,
                chunk_duration=chunk_duration,
            )
        
        # v3.2: Timing for verification/resume logic
        verify_start = time.perf_counter()
        completed_indices = set(state.completed_chunks)
        pending_count = total_chunks - len(completed_indices)
        verify_elapsed = time.perf_counter() - verify_start
        self.ui.log_info(f"Verified {len(completed_indices)}/{total_chunks} chunks complete (took {verify_elapsed:.4f}s)")
        
        if pending_count == 0:
            self.ui.log_success("All chunks already transcribed!")
        else:
            # ------------------------------------------------------------------
            # Producer-Consumer Pipeline
            # ------------------------------------------------------------------
            self.ui.log_info(f"Starting pipeline for {pending_count} pending chunks...")
            
            # Use a bounded queue to prevent disk filling
            chunk_queue = queue.Queue(maxsize=20)
            
            # Start Producer Thread
            producer_thread = threading.Thread(
                target=self._producer_task,
                args=(input_path, boundaries, temp_dir, chunk_queue, completed_indices),
                daemon=True
            )
            producer_thread.start()
            
            # Start Progress UI
            progress = self.ui.create_progress()
            with progress:
                self.ui.start_progress(total_chunks, num_workers)
                self.ui.update_progress(len(completed_indices))
                
                # Manual interaction with process_handler to handle the stream
                # We'll use a local ThreadPool for the consumers to manage the queue
                from concurrent.futures import ThreadPoolExecutor, as_completed
                
                with ThreadPoolExecutor(max_workers=num_workers) as executor:
                    futures = []
                    active_tasks = 0
                    
                    while True:
                        # 1. Check for new chunks if we have capacity
                        if active_tasks < num_workers:
                            try:
                                # Non-blocking get from queue
                                item = chunk_queue.get_nowait()
                                if item is None: # End signal
                                    # Don't break yet, need to wait for active tasks
                                    chunk_queue.put(None) # Put it back for other workers/logic
                                    if active_tasks == 0:
                                        break
                                else:
                                    idx, cpath = item
                                    future = executor.submit(
                                        self.process_handler.transcribe_chunk,
                                        chunk_path=cpath,
                                        chunk_index=idx,
                                        model=args.model,
                                        language=args.language,
                                        output_dir=temp_dir,
                                        timeout=args.timeout
                                    )
                                    futures.append(future)
                                    active_tasks += 1
                                    continue # Try to fill more workers
                            except queue.Empty:
                                if producer_thread.is_alive() or active_tasks > 0:
                                    pass # Just wait for results
                                else:
                                    break
                        
                        # 2. Re-check for stopped signal
                        if self.should_stop():
                            break
                            
                        # 3. Check for completed futures
                        completed_futures = [f for f in futures if f.done()]
                        for f in completed_futures:
                            result = f.result()
                            self.progress_callback(len(state.completed_chunks) + 1, total_chunks, result)
                            futures.remove(f)
                            active_tasks -= 1
                            
                        # 4. Sleep a bit if no work can be done
                        if active_tasks == num_workers or (chunk_queue.empty() and producer_thread.is_alive()):
                            import time
                            time.sleep(0.1)
                        elif active_tasks == 0 and chunk_queue.empty() and not producer_thread.is_alive():
                            break
                
                self.ui.finish_progress()
        
        # Check if we have all chunks (Reload state)
        state = self.state_manager.load_progress()
        if not state.is_complete():
            self.ui.log_warning(f"Incomplete: {len(state.completed_chunks)}/{state.total_chunks} chunks")
            return 1
        
        # v4.0 Finalization: Create final subtitle file
        self.ui.log_info("Finalizing subtitles...")
        
        # Check for existing _IN_PROGRESS.srt file
        partial_srt = input_path.parent / f"{input_path.stem}{PARTIAL_SUBTITLE_SUFFIX}.srt"
        output_srt = output_dir / f"{input_path.stem}.srt"
        
        if partial_srt.exists():
            # Rename _IN_PROGRESS.srt to final .srt
            try:
                # If output already exists, remove it first
                if output_srt.exists():
                    output_srt.unlink()
                partial_srt.rename(output_srt)
                self.ui.log_success(f"Finalized: {partial_srt.name} → {output_srt.name}")
            except Exception as e:
                self.ui.log_warning(f"Could not rename partial file: {e}")
                # Fall back to merge approach
                self._merge_final_subtitle(temp_dir, state, chunk_duration, output_srt)
        else:
            # No partial file exists - merge all chunks directly
            self._merge_final_subtitle(temp_dir, state, chunk_duration, output_srt)
        
        # Show summary
        self.ui.show_summary(
            video_name=input_path.name,
            total_chunks=total_chunks,
            duration_mins=duration_mins,
            output_path=str(output_srt),
        )

        # Interactive Cleanup Prompt
        from rich.prompt import Confirm
        if Confirm.ask("\n[bold cyan]Cleanup temporary files?[/]", default=True):
            self.ui.log_info("Cleaning up temporary files...")
            self.file_handler.cleanup_temp_directory(input_path)
            self.ui.log_success("Cleanup complete.")
        else:
            self.ui.log_info("Temporary files kept.")
        
        return 0
    
    def _merge_final_subtitle(self, temp_dir: Path, state: TranscriptionState, chunk_duration: int, output_srt: Path) -> None:
        """Merge all chunk SRT files into final output."""
        chunks_paths = [temp_dir / f"{TEMP_CHUNK_PREFIX}{i:04d}.wav" for i in range(state.total_chunks)]
        srt_files = [temp_dir / f"{c.stem}.srt" for c in chunks_paths]
        
        try:
            self.subtitle_parser.merge_srt_files(srt_files, chunk_duration, output_srt)
            self.ui.log_success(f"Output: {output_srt}")
        except Exception as e:
            self.ui.log_error(f"Failed to merge subtitles: {e}")


def main() -> int:
    """Entry point."""
    app = TranscriptionApp()
    return app.run()


if __name__ == "__main__":
    sys.exit(main())
