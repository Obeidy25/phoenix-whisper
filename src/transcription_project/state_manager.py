# ==============================================================================
# state_manager.py - Checkpointing and State Management Module
# ==============================================================================
"""
Handles save/load of transcription progress for stop/resume functionality.
Provides atomic writes to prevent corruption on unexpected exits.
"""

import json
import tempfile
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict, field


@dataclass
class TranscriptionState:
    """Complete state of a transcription job."""
    video_path: str
    video_duration: float
    total_chunks: int
    completed_chunks: List[int] = field(default_factory=list)
    failed_chunks: List[int] = field(default_factory=list)
    model: str = "small"
    language: Optional[str] = None
    chunk_duration: int = 600
    started_at: str = ""
    updated_at: str = ""
    srt_files: Dict[int, str] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.started_at:
            self.started_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TranscriptionState":
        """Create state from dictionary."""
        return cls(**data)
    
    def mark_chunk_completed(self, chunk_index: int, srt_path: Optional[Path] = None) -> None:
        """Mark a chunk as successfully completed."""
        if chunk_index not in self.completed_chunks:
            self.completed_chunks.append(chunk_index)
        if chunk_index in self.failed_chunks:
            self.failed_chunks.remove(chunk_index)
        if srt_path:
            self.srt_files[chunk_index] = str(srt_path)
        self.updated_at = datetime.now().isoformat()
    
    def mark_chunk_failed(self, chunk_index: int) -> None:
        """Mark a chunk as failed."""
        if chunk_index not in self.failed_chunks:
            self.failed_chunks.append(chunk_index)
        self.updated_at = datetime.now().isoformat()
    
    def get_pending_chunks(self) -> List[int]:
        """Get list of chunks that haven't been processed."""
        all_chunks = set(range(self.total_chunks))
        completed = set(self.completed_chunks)
        return sorted(list(all_chunks - completed))
    
    def is_complete(self) -> bool:
        """Check if all chunks have been processed."""
        return len(self.completed_chunks) == self.total_chunks
    
    def get_progress_percentage(self) -> float:
        """Get completion percentage."""
        if self.total_chunks == 0:
            return 0.0
        return (len(self.completed_chunks) / self.total_chunks) * 100


class StateManager:
    """
    Manages transcription state persistence.
    
    Provides atomic file operations to prevent corruption and
    supports resume functionality for interrupted transcriptions.
    """
    
    def __init__(self, progress_file: Path):
        """
        Initialize the state manager.
        
        Args:
            progress_file: Path to the progress JSON file
        """
        self.progress_file = progress_file
        self._state: Optional[TranscriptionState] = None
    
    @property
    def state(self) -> Optional[TranscriptionState]:
        """Get the current state (loads from file if not cached)."""
        if self._state is None:
            self._state = self.load_progress()
        return self._state
    
    def save_progress(self, state: TranscriptionState) -> None:
        """
        Save progress to file atomically.
        
        Uses a temporary file and rename to ensure atomic writes,
        preventing corruption if the process is interrupted.
        
        Args:
            state: Current transcription state to save
        """
        state.updated_at = datetime.now().isoformat()
        self._state = state
        
        # Create parent directory if needed
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to temporary file first
        temp_fd, temp_path = tempfile.mkstemp(
            dir=self.progress_file.parent,
            prefix=".progress_",
            suffix=".tmp",
        )
        
        try:
            with open(temp_fd, "w", encoding="utf-8") as f:
                json.dump(state.to_dict(), f, indent=2)
            
            # Atomic rename
            shutil.move(temp_path, self.progress_file)
        except Exception:
            # Clean up temp file on error
            try:
                Path(temp_path).unlink()
            except OSError:
                pass
            raise
    
    def load_progress(self) -> Optional[TranscriptionState]:
        """
        Load progress from file.
        
        Returns:
            TranscriptionState if file exists and is valid, None otherwise
        """
        if not self.progress_file.exists():
            return None
        
        try:
            with open(self.progress_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            state = TranscriptionState.from_dict(data)
            self._state = state
            return state
            
        except (json.JSONDecodeError, TypeError, KeyError) as e:
            # Corrupted or invalid file
            return None
    
    def clear_progress(self) -> None:
        """Remove the progress file."""
        self._state = None
        if self.progress_file.exists():
            self.progress_file.unlink()
    
    def has_existing_progress(self) -> bool:
        """Check if there's existing progress for this video."""
        return self.progress_file.exists()
    
    def is_compatible(self, video_path: Path, model: str, chunk_duration: int) -> bool:
        """
        Check if existing progress is compatible with current settings.
        
        Args:
            video_path: Path to the video file
            model: Whisper model name
            chunk_duration: Duration of chunks in seconds
            
        Returns:
            True if progress can be resumed with current settings
        """
        state = self.load_progress()
        if state is None:
            return False
        
        # Check if key parameters match
        return (
            state.video_path == str(video_path) and
            state.model == model and
            state.chunk_duration == chunk_duration
        )
    
    def create_new_state(
        self,
        video_path: Path,
        video_duration: float,
        total_chunks: int,
        model: str = "small",
        language: Optional[str] = None,
        chunk_duration: int = 600,
    ) -> TranscriptionState:
        """
        Create a new transcription state.
        
        Args:
            video_path: Path to the video file
            video_duration: Duration of the video in seconds
            total_chunks: Total number of chunks
            model: Whisper model name
            language: Optional language code
            chunk_duration: Duration of each chunk
            
        Returns:
            New TranscriptionState instance
        """
        state = TranscriptionState(
            video_path=str(video_path),
            video_duration=video_duration,
            total_chunks=total_chunks,
            model=model,
            language=language,
            chunk_duration=chunk_duration,
        )
        
        self._state = state
        return state


def get_progress_file_path(video_path: Path) -> Path:
    """
    Get the default progress file path for a video.
    
    Args:
        video_path: Path to the video file
        
    Returns:
        Path to the progress JSON file
    """
    temp_dir = video_path.parent / f".transcription_temp_{video_path.stem}"
    return temp_dir / "progress.json"


if __name__ == "__main__":
    # Test state manager
    from pathlib import Path
    import tempfile
    
    with tempfile.TemporaryDirectory() as tmpdir:
        progress_file = Path(tmpdir) / "test_progress.json"
        manager = StateManager(progress_file)
        
        # Create new state
        state = manager.create_new_state(
            video_path=Path("/test/video.mp4"),
            video_duration=3600.0,
            total_chunks=6,
            model="small",
            chunk_duration=600,
        )
        
        # Mark some chunks complete
        state.mark_chunk_completed(0, Path("/test/chunk_0000.srt"))
        state.mark_chunk_completed(1, Path("/test/chunk_0001.srt"))
        
        # Save and reload
        manager.save_progress(state)
        loaded = manager.load_progress()
        
        print(f"Loaded state: {loaded}")
        print(f"Completed: {loaded.completed_chunks}")
        print(f"Pending: {loaded.get_pending_chunks()}")
        print(f"Progress: {loaded.get_progress_percentage():.1f}%")
