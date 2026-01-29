# ==============================================================================
# process_handler.py - External Process Execution Module
# ==============================================================================
"""
Handles execution of external commands (ffmpeg, whisper) with support
for concurrent processing using ThreadPoolExecutor.
"""

import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Callable, Any
from dataclasses import dataclass


@dataclass
class TranscriptionResult:
    """Result of a single chunk transcription."""
    chunk_index: int
    chunk_path: Path
    srt_path: Optional[Path]
    success: bool
    error_message: Optional[str] = None


class ProcessHandler:
    """
    Handles external command execution with concurrency support.
    
    This class manages the execution of whisper transcription commands,
    supporting both sequential and parallel processing modes.
    """
    
    def __init__(self, num_workers: int = 1):
        """
        Initialize the process handler.
        
        Args:
            num_workers: Number of concurrent workers (1 = sequential)
        """
        self.num_workers = max(1, num_workers)
        self._executor: Optional[ThreadPoolExecutor] = None
    
    def run_command(
        self,
        cmd: List[str],
        timeout: Optional[int] = None,
        cwd: Optional[Path] = None,
    ) -> subprocess.CompletedProcess:
        """
        Run a command and return the result.
        
        Args:
            cmd: Command and arguments as a list
            timeout: Optional timeout in seconds
            cwd: Optional working directory
            
        Returns:
            CompletedProcess with stdout, stderr, and return code
        """
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
    
    def transcribe_chunk(
        self,
        chunk_path: Path,
        chunk_index: int,
        model: str = "small",
        language: Optional[str] = None,
        output_dir: Optional[Path] = None,
        timeout: int = 1800,
    ) -> TranscriptionResult:
        """
        Transcribe a single audio chunk using whisper.
        
        Args:
            chunk_path: Path to the audio chunk file
            chunk_index: Index of this chunk (for ordering)
            model: Whisper model to use
            language: Optional language code
            output_dir: Optional output directory
            timeout: Timeout in seconds
            
        Returns:
            TranscriptionResult with success status and paths
        """
        if output_dir is None:
            output_dir = chunk_path.parent
        
        # Build whisper command
        cmd = [
            "whisper",
            str(chunk_path),
            "--model", model,
            "--output_format", "srt",
            "--output_dir", str(output_dir),
        ]
        
        if language:
            cmd.extend(["--language", language])
        
        try:
            result = self.run_command(
                cmd,
                timeout=timeout,
                cwd=output_dir,
            )
            
            if result.returncode != 0:
                return TranscriptionResult(
                    chunk_index=chunk_index,
                    chunk_path=chunk_path,
                    srt_path=None,
                    success=False,
                    error_message=f"Whisper failed: {result.stderr}",
                )
            
            # Find the generated SRT file
            srt_path = output_dir / f"{chunk_path.stem}.srt"
            
            if not srt_path.exists():
                return TranscriptionResult(
                    chunk_index=chunk_index,
                    chunk_path=chunk_path,
                    srt_path=None,
                    success=False,
                    error_message="SRT file not generated",
                )
            
            return TranscriptionResult(
                chunk_index=chunk_index,
                chunk_path=chunk_path,
                srt_path=srt_path,
                success=True,
            )
            
        except subprocess.TimeoutExpired:
            return TranscriptionResult(
                chunk_index=chunk_index,
                chunk_path=chunk_path,
                srt_path=None,
                success=False,
                error_message="Transcription timed out",
            )
        except Exception as e:
            return TranscriptionResult(
                chunk_index=chunk_index,
                chunk_path=chunk_path,
                srt_path=None,
                success=False,
                error_message=str(e),
            )
    
    def transcribe_chunks_sequential(
        self,
        chunks: List[Path],
        model: str = "small",
        language: Optional[str] = None,
        output_dir: Optional[Path] = None,
        timeout: int = 1800,
        progress_callback: Optional[Callable[[int, int, TranscriptionResult], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> List[TranscriptionResult]:
        """
        Transcribe chunks sequentially.
        
        Args:
            chunks: List of chunk file paths
            model: Whisper model to use
            language: Optional language code
            output_dir: Optional output directory
            progress_callback: Optional callback(completed, total, result)
            should_stop: Optional callable that returns True to stop processing
            
        Returns:
            List of TranscriptionResult objects
        """
        results = []
        total = len(chunks)
        
        for i, chunk_path in enumerate(chunks):
            if should_stop and should_stop():
                break
            
            result = self.transcribe_chunk(
                chunk_path=chunk_path,
                chunk_index=i,
                model=model,
                language=language,
                output_dir=output_dir,
                timeout=timeout,
            )
            
            results.append(result)
            
            if progress_callback:
                progress_callback(i + 1, total, result)
        
        return results
    
    def transcribe_chunks_concurrent(
        self,
        chunks: List[Path],
        model: str = "small",
        language: Optional[str] = None,
        output_dir: Optional[Path] = None,
        timeout: int = 1800,
        progress_callback: Optional[Callable[[int, int, TranscriptionResult], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> List[TranscriptionResult]:
        """
        Transcribe chunks concurrently using ThreadPoolExecutor.
        
        Args:
            chunks: List of chunk file paths
            model: Whisper model to use
            language: Optional language code
            output_dir: Optional output directory
            progress_callback: Optional callback(completed, total, result)
            should_stop: Optional callable that returns True to stop processing
            
        Returns:
            List of TranscriptionResult objects, sorted by chunk index
        """
        results = []
        total = len(chunks)
        completed = 0
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(
                    self.transcribe_chunk,
                    chunk_path=chunk,
                    chunk_index=i,
                    model=model,
                    language=language,
                    output_dir=output_dir,
                    timeout=timeout,
                ): i
                for i, chunk in enumerate(chunks)
            }
            
            # Process completed tasks
            for future in as_completed(future_to_index):
                if should_stop and should_stop():
                    # Cancel remaining tasks
                    for f in future_to_index:
                        f.cancel()
                    break
                
                result = future.result()
                results.append(result)
                completed += 1
                
                if progress_callback:
                    progress_callback(completed, total, result)
        
        # Sort results by chunk index
        results.sort(key=lambda r: r.chunk_index)
        return results
    
    def transcribe_chunks(
        self,
        chunks: List[Path],
        model: str = "small",
        language: Optional[str] = None,
        output_dir: Optional[Path] = None,
        timeout: int = 1800,
        progress_callback: Optional[Callable[[int, int, TranscriptionResult], None]] = None,
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> List[TranscriptionResult]:
        """
        Transcribe chunks using the configured concurrency level.
        
        Automatically chooses between sequential and concurrent processing
        based on the num_workers setting.
        
        Args:
            chunks: List of chunk file paths
            model: Whisper model to use
            language: Optional language code
            output_dir: Optional output directory
            progress_callback: Optional callback(completed, total, result)
            should_stop: Optional callable that returns True to stop processing
            
        Returns:
            List of TranscriptionResult objects
        """
        if self.num_workers <= 1:
            return self.transcribe_chunks_sequential(
                chunks=chunks,
                model=model,
                language=language,
                output_dir=output_dir,
                timeout=timeout,
                progress_callback=progress_callback,
                should_stop=should_stop,
            )
        else:
            return self.transcribe_chunks_concurrent(
                chunks=chunks,
                model=model,
                language=language,
                output_dir=output_dir,
                timeout=timeout,
                progress_callback=progress_callback,
                should_stop=should_stop,
            )


if __name__ == "__main__":
    # Test process handler initialization
    handler = ProcessHandler(num_workers=4)
    print(f"ProcessHandler initialized with {handler.num_workers} workers")
