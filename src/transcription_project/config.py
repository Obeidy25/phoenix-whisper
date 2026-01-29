# ==============================================================================
# config.py - Central Configuration Module
# ==============================================================================
"""
Central configuration containing defaults, RAM profiles, and constants.
This module provides all configurable parameters for the transcription application.
"""

from pathlib import Path
from typing import Dict, List

# ==============================================================================
# Chunk Duration Settings (in seconds)
# ==============================================================================
DEFAULT_CHUNK_DURATION: int = 600  # 10 minutes

# Adaptive Chunk Sizes (in seconds)
CHUNK_DURATION_GPU: int = 900        # 15 minutes - for GPU acceleration
CHUNK_DURATION_STRONG_CPU: int = 300 # 5 minutes - for 8+ cores
CHUNK_DURATION_WEAK_CPU: int = 120   # 2 minutes - for <8 cores

# RAM profiles: Adjust chunk size based on available system memory
RAM_PROFILES: Dict[str, int] = {
    "low": 300,      # 5 minutes - for systems with limited RAM
    "medium": 600,   # 10 minutes - balanced default
    "high": 1200,    # 20 minutes - for high-performance systems
}

# ==============================================================================
# Whisper Model Settings
# ==============================================================================
DEFAULT_WHISPER_MODEL: str = "small"

# Available Whisper models (from smallest to largest)
AVAILABLE_MODELS: List[str] = [
    "tiny",
    "base", 
    "small",
    "medium",
    "large",
    "large-v2",
    "large-v3",
]

# ==============================================================================
# Concurrency Settings
# ==============================================================================
DEFAULT_NUM_WORKERS: int = 1
MAX_NUM_WORKERS: int = 16  # Safety limit

# ==============================================================================
# Supported File Formats
# ==============================================================================
SUPPORTED_VIDEO_EXTENSIONS: List[str] = [
    ".mp4",
    ".mkv",
    ".avi",
    ".mov",
    ".webm",
    ".m4v",
    ".flv",
    ".wmv",
]

SUPPORTED_AUDIO_EXTENSIONS: List[str] = [
    ".mp3",
    ".wav",
    ".flac",
    ".aac",
    ".ogg",
    ".m4a",
]

# Combined supported extensions
SUPPORTED_EXTENSIONS: List[str] = SUPPORTED_VIDEO_EXTENSIONS + SUPPORTED_AUDIO_EXTENSIONS

# ==============================================================================
# Output Settings
# ==============================================================================
DEFAULT_OUTPUT_FORMAT: str = "srt"
TEMP_CHUNK_PREFIX: str = "chunk_"
PROGRESS_FILE_NAME: str = "progress.json"
PARTIAL_SUBTITLE_SUFFIX: str = "_IN_PROGRESS"  # v4.0: Single unified partial file

# ==============================================================================
# Application Metadata
# ==============================================================================
APP_NAME: str = "Video Transcription Tool"
APP_VERSION: str = "5.3.0"
APP_DESCRIPTION: str = "Ultimate Automation - Professional, resumable, and concurrent video transcription"


def get_chunk_duration(ram_profile: str) -> int:
    """
    Get chunk duration based on RAM profile.
    
    Args:
        ram_profile: One of 'low', 'medium', 'high'
        
    Returns:
        Chunk duration in seconds
    """
    return RAM_PROFILES.get(ram_profile.lower(), DEFAULT_CHUNK_DURATION)


def is_supported_file(file_path: Path) -> bool:
    """
    Check if a file has a supported extension.
    
    Args:
        file_path: Path to the file to check
        
    Returns:
        True if the file extension is supported
    """
    return file_path.suffix.lower() in SUPPORTED_EXTENSIONS
