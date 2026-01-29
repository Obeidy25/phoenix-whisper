# ==============================================================================
# cli.py - Command Line Interface Module
# ==============================================================================
"""
All argparse logic is isolated here for clean separation of concerns.
Provides argument parsing and validation for the transcription application.
"""

import argparse
from pathlib import Path
from typing import Optional

from .config import (
    APP_NAME,
    APP_VERSION,
    APP_DESCRIPTION,
    DEFAULT_WHISPER_MODEL,
    DEFAULT_NUM_WORKERS,
    MAX_NUM_WORKERS,
    AVAILABLE_MODELS,
    RAM_PROFILES,
    SUPPORTED_EXTENSIONS,
)


def create_parser() -> argparse.ArgumentParser:
    """
    Create and configure the argument parser.
    
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        prog=APP_NAME,
        description=APP_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  %(prog)s video.mp4                      # Basic transcription
  %(prog)s video.mp4 --model medium       # Use medium model
  %(prog)s video.mp4 --num-workers 4      # Use 4 concurrent workers
  %(prog)s video.mp4 --ram-profile low    # Use smaller chunks for low RAM
  %(prog)s video.mp4 --resume             # Resume interrupted transcription

Supported formats: {', '.join(SUPPORTED_EXTENSIONS)}
        """,
    )
    
    # Version
    parser.add_argument(
        "--version",
        action="version",
        version=f"{APP_NAME} v{APP_VERSION}",
    )
    
    # Positional argument: input file (optional when using --diagnose)
    parser.add_argument(
        "input",
        type=Path,
        nargs='?',  # Optional to allow --diagnose without input
        default=None,
        help="Path to the video or audio file to transcribe",
    )
    
    # Output directory
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=None,
        help="Output directory for transcription files (default: same as input)",
    )
    
    # Whisper model selection
    parser.add_argument(
        "-m", "--model",
        type=str,
        default=DEFAULT_WHISPER_MODEL,
        choices=AVAILABLE_MODELS,
        help=f"Whisper model to use (default: {DEFAULT_WHISPER_MODEL})",
    )
    
    # Language selection
    parser.add_argument(
        "-l", "--language",
        type=str,
        default=None,
        help="Source language code (e.g., 'en', 'es', 'fr'). Auto-detected if not specified",
    )
    
    # Concurrent workers
    parser.add_argument(
        "-w", "--num-workers",
        type=int,
        default=None,
        help="Number of concurrent workers (default: auto-detected based on GPU/CPU)",
    )
    
    # RAM profile
    parser.add_argument(
        "-r", "--ram-profile",
        type=str,
        default=None,
        choices=list(RAM_PROFILES.keys()),
        help="RAM profile for chunk sizing: low (5min), medium (10min), high (20min)",
    )
    
    # Resume from checkpoint
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume transcription from a previous checkpoint",
    )
    
    # Timeout
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Timeout in seconds for each chunk transcription (default: 3600)",
    )
    
    # Keep temporary files (for debugging)
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep temporary chunk files after completion",
    )
    
    # Verbose output
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    
    # Diagnostic mode
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Run diagnostic tests to measure performance (does not run transcription)",
    )
    
    return parser


def validate_args(args: argparse.Namespace) -> Optional[str]:
    """
    Validate parsed arguments.
    
    Args:
        args: Parsed arguments namespace
        
    Returns:
        Error message if validation fails, None if valid
    """
    # Skip input validation for diagnose mode
    if getattr(args, 'diagnose', False):
        return None
    
    # Input is required for normal operation
    if args.input is None:
        return "Input file is required. Use --diagnose for diagnostics without a file."
    
    # Check input file exists
    if not args.input.exists():
        return f"Input file not found: {args.input}"
    
    # Check input file is a file (not directory)
    if not args.input.is_file():
        return f"Input path is not a file: {args.input}"
    
    # Check file extension
    if args.input.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return f"Unsupported file format: {args.input.suffix}. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
    
    # Validate num_workers range
    if args.num_workers is not None:
        if args.num_workers < 1:
            return "Number of workers must be at least 1"
        if args.num_workers > MAX_NUM_WORKERS:
            return f"Number of workers cannot exceed {MAX_NUM_WORKERS}"
    
    # Validate output directory if specified
    if args.output_dir is not None:
        if args.output_dir.exists() and not args.output_dir.is_dir():
            return f"Output path exists but is not a directory: {args.output_dir}"
    
    return None


def parse_arguments() -> argparse.Namespace:
    """
    Parse and validate command line arguments.
    
    Returns:
        Validated arguments namespace
        
    Raises:
        SystemExit: If validation fails
    """
    parser = create_parser()
    args = parser.parse_args()
    
    # Set default output directory if not specified
    if args.output_dir is None:
        args.output_dir = args.input.parent
    
    # Validate arguments
    error = validate_args(args)
    if error:
        parser.error(error)
    
    return args


if __name__ == "__main__":
    # Test parsing
    args = parse_arguments()
    print(f"Input: {args.input}")
    print(f"Output: {args.output_dir}")
    print(f"Model: {args.model}")
    print(f"Workers: {args.num_workers}")
    print(f"RAM Profile: {args.ram_profile}")
