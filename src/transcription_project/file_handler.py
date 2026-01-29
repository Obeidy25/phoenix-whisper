# ==============================================================================
# file_handler.py - Filesystem Operations Module
# ==============================================================================
"""
Handles all filesystem operations including video duration detection,
chunk creation, and temporary file management.
"""

import subprocess
import json
import shutil
from pathlib import Path
from typing import List, Optional, Tuple
from dataclasses import dataclass

from .config import TEMP_CHUNK_PREFIX


@dataclass
class VideoInfo:
    """Information about a video file."""
    path: Path
    duration: float  # in seconds
    format_name: str
    size_bytes: int


class FileHandler:
    """
    Handles filesystem operations for video processing.
    
    This class provides methods for video analysis, chunk creation,
    and temporary file management.
    """
    
    def __init__(self, temp_dir: Optional[Path] = None):
        """
        Initialize the file handler.
        
        Args:
            temp_dir: Optional custom temporary directory for chunks
        """
        self.temp_dir = temp_dir
    
    def get_video_info(self, video_path: Path) -> VideoInfo:
        """
        Get detailed information about a video file using ffprobe.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            VideoInfo dataclass with video metadata
            
        Raises:
            RuntimeError: If ffprobe fails to analyze the file
        """
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            str(video_path),
        ]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )
            
            if result.returncode != 0:
                raise RuntimeError(f"ffprobe failed: {result.stderr}")
            
            data = json.loads(result.stdout)
            format_info = data.get("format", {})
            
            return VideoInfo(
                path=video_path,
                duration=float(format_info.get("duration", 0)),
                format_name=format_info.get("format_name", "unknown"),
                size_bytes=int(format_info.get("size", 0)),
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError("ffprobe timed out while analyzing video")
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse ffprobe output: {e}")
    
    def get_video_duration(self, video_path: Path) -> float:
        """
        Get the duration of a video file in seconds.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Duration in seconds
        """
        info = self.get_video_info(video_path)
        return info.duration
    
    def create_output_directory(self, path: Path) -> Path:
        """
        Create output directory if it doesn't exist.
        
        Args:
            path: Directory path to create
            
        Returns:
            The created directory path
        """
        path.mkdir(parents=True, exist_ok=True)
        return path
    
    def get_temp_directory(self, video_path: Path) -> Path:
        """
        Get or create temporary directory for chunk files.
        
        Args:
            video_path: Path to the source video
            
        Returns:
            Path to temporary directory
        """
        if self.temp_dir:
            temp_path = self.temp_dir
        else:
            temp_path = video_path.parent / f".transcription_temp_{video_path.stem}"
        
        temp_path.mkdir(parents=True, exist_ok=True)
        return temp_path
    
    def calculate_chunks(self, duration: float, chunk_duration: int) -> List[Tuple[float, float]]:
        """
        Calculate chunk boundaries for a video.
        
        Args:
            duration: Total video duration in seconds
            chunk_duration: Duration of each chunk in seconds
            
        Returns:
            List of (start_time, end_time) tuples for each chunk
        """
        chunks = []
        start = 0.0
        
        while start < duration:
            end = min(start + chunk_duration, duration)
            chunks.append((start, end))
            start = end
        
        return chunks
    
    def extract_chunk(
        self,
        video_path: Path,
        start: float,
        duration: float,
        chunk_index: int,
        output_dir: Path,
    ) -> Path:
        """
        Extract a single audio chunk from a video file.
        
        Args:
            video_path: Path to the source video
            start: Start time in seconds
            duration: Duration in seconds
            chunk_index: Index of the chunk
            output_dir: Directory where the chunk will be saved
            
        Returns:
            Path to the created audio chunk file
            
        Raises:
            RuntimeError: If extraction fails after all retry attempts
        """
        chunk_path = output_dir / f"{TEMP_CHUNK_PREFIX}{chunk_index:04d}.wav"
        
        # Skip if chunk already exists (for resume functionality)
        if chunk_path.exists() and chunk_path.stat().st_size > 0:
            return chunk_path
        
        # v4.1: Enhanced ffmpeg command for problematic videos
        # Using output seeking (-ss after -i) for more accurate but slower extraction
        # Added analyzeduration and probesize for videos with missing metadata
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-analyzeduration", "100M",  # Analyze more data for problematic files
            "-probesize", "100M",  # Read more data to find stream info
            "-i", str(video_path),
            "-ss", str(start),  # Output seeking (after -i) - more accurate for problematic files
            "-t", str(duration),
            "-vn",  # No video
            "-acodec", "pcm_s16le",  # WAV format
            "-ar", "16000",  # 16kHz sample rate (optimal for Whisper)
            "-ac", "1",  # Mono
            str(chunk_path),
        ]
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout per chunk
            errors='replace',  # Handle encoding issues in stderr
        )
        
        # v4.1: Check if output file was created and is not empty
        if chunk_path.exists() and chunk_path.stat().st_size > 0:
            return chunk_path
        
        # If first attempt failed, try with input seeking (faster but may miss keyframes)
        # This is a fallback for some edge cases
        if result.returncode != 0 or not chunk_path.exists() or chunk_path.stat().st_size == 0:
            # Retry with different approach: input seeking + copy codec for audio
            cmd_retry = [
                "ffmpeg",
                "-y",
                "-ss", str(start),  # Input seeking (before -i) - faster
                "-analyzeduration", "100M",
                "-probesize", "100M", 
                "-i", str(video_path),
                "-t", str(duration),
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", "16000",
                "-ac", "1",
                str(chunk_path),
            ]
            
            result = subprocess.run(
                cmd_retry,
                capture_output=True,
                text=True,
                timeout=300,
                errors='replace',
            )
            
            # Final check
            if chunk_path.exists() and chunk_path.stat().st_size > 0:
                return chunk_path
        
        # If still failed, raise error with details
        error_msg = result.stderr if result.stderr else "Unknown error - output file empty or not created"
        raise RuntimeError(f"Failed to create chunk {chunk_index}: {error_msg}")

    def split_video_to_chunks(
        self,
        video_path: Path,
        chunk_duration: int,
        output_dir: Optional[Path] = None,
    ) -> List[Path]:
        """
        Split a video into audio chunks for transcription.
        
        Args:
            video_path: Path to the source video
            chunk_duration: Duration of each chunk in seconds
            output_dir: Optional output directory for chunks
            
        Returns:
            List of paths to the created audio chunk files
        """
        # Get video duration
        duration = self.get_video_duration(video_path)
        
        # Calculate chunk boundaries
        chunk_boundaries = self.calculate_chunks(duration, chunk_duration)
        
        # Set up output directory
        if output_dir is None:
            output_dir = self.get_temp_directory(video_path)
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
        
        chunk_paths = []
        
        for i, (start, end) in enumerate(chunk_boundaries):
            chunk_path = self.extract_chunk(
                video_path=video_path,
                start=start,
                duration=end - start,
                chunk_index=i,
                output_dir=output_dir,
            )
            chunk_paths.append(chunk_path)
        
        return chunk_paths
    
    def cleanup_temp_files(self, paths: List[Path], remove_directory: bool = True) -> None:
        """
        Clean up temporary files after processing.
        
        Args:
            paths: List of file paths to remove
            remove_directory: Also remove the parent directory if empty
        """
        directories_to_check = set()
        
        for path in paths:
            if path.exists():
                directories_to_check.add(path.parent)
                path.unlink()
        
        if remove_directory:
            for directory in directories_to_check:
                try:
                    # Only remove if empty and is a temp directory
                    if directory.exists() and not any(directory.iterdir()):
                        if ".transcription_temp_" in directory.name:
                            directory.rmdir()
                except OSError:
                    pass  # Directory not empty or other error
    
    def cleanup_temp_directory(self, video_path: Path) -> None:
        """
        Remove the entire temporary directory for a video.
        
        Args:
            video_path: Path to the source video
        """
        temp_dir = video_path.parent / f".transcription_temp_{video_path.stem}"
        if temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def find_existing_chunks(self, video_path: Path) -> List[Path]:
        """
        Find existing chunk files for a video (for resume functionality).
        
        Args:
            video_path: Path to the source video
            
        Returns:
            List of existing chunk file paths, sorted by index
        """
        temp_dir = self.get_temp_directory(video_path)
        
        if not temp_dir.exists():
            return []
        
        chunks = list(temp_dir.glob(f"{TEMP_CHUNK_PREFIX}*.wav"))
        return sorted(chunks, key=lambda p: int(p.stem.split("_")[-1]))


if __name__ == "__main__":
    # Test file handler
    handler = FileHandler()
    print("FileHandler initialized successfully")
