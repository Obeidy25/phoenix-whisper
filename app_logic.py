#!/usr/bin/env python3
# ==============================================================================
# app_logic.py - Application Logic with Pre-Flight Checks (v5.3)
# ==============================================================================
"""
User-friendly launcher with file browser dialog and comprehensive pre-flight checks.
Run via run.py for the best experience.
"""

import sys
import subprocess
import shutil
import os
from pathlib import Path

# Fix Windows console encoding for UTF-8
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except:
        pass

# Minimum Python version (v5.0: now requires 3.9+)
MIN_PYTHON = (3, 9)

# Required packages
REQUIRED_PACKAGES = [
    ("rich", "rich"),
    ("whisper", "openai-whisper"),
]

# Supported video extensions
VIDEO_EXTENSIONS = [
    ("Video Files", "*.mp4 *.mkv *.avi *.mov *.webm *.m4v *.flv *.wmv"),
    ("Audio Files", "*.mp3 *.wav *.flac *.aac *.ogg *.m4a"),
    ("All Files", "*.*"),
]


def safe_print(msg):
    """Print with fallback for encoding issues."""
    try:
        print(msg)
    except UnicodeEncodeError:
        import re
        clean = re.sub(r'[^\x00-\x7F]+', '*', msg)
        print(clean)


def print_banner():
    """Print application banner."""
    safe_print("\n" + "=" * 60)
    safe_print("  [*] Video Transcription Tool v5.3")
    safe_print("  Ultimate Automation - Professional, Resumable & Concurrent")
    safe_print("=" * 60 + "\n")


def print_error(msg):
    safe_print(f"\n[X] ERROR: {msg}\n")


def print_success(msg):
    safe_print(f"[OK] {msg}")


def print_warning(msg):
    safe_print(f"[!] {msg}")


def print_info(msg):
    safe_print(f"[i] {msg}")


def open_file_dialog():
    """Open a file dialog to select a video file."""
    try:
        import tkinter as tk
        from tkinter import filedialog
        
        # Create hidden root window
        root = tk.Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        
        # Configure file types
        filetypes = [
            ("Video/Audio Files", "*.mp4 *.mkv *.avi *.mov *.webm *.m4v *.flv *.wmv *.mp3 *.wav *.flac *.aac *.ogg *.m4a"),
            ("Video Files", "*.mp4 *.mkv *.avi *.mov *.webm *.m4v *.flv *.wmv"),
            ("Audio Files", "*.mp3 *.wav *.flac *.aac *.ogg *.m4a"),
            ("All Files", "*.*"),
        ]
        
        # Open file dialog
        file_path = filedialog.askopenfilename(
            title="Select Video/Audio File to Transcribe",
            filetypes=filetypes,
            initialdir=os.path.expanduser("~")
        )
        
        root.destroy()
        
        if file_path:
            return Path(file_path)
        return None
        
    except ImportError:
        print_warning("tkinter not available. Please provide file path manually.")
        return None
    except Exception as e:
        print_warning(f"Could not open file dialog: {e}")
        return None


def get_video_file_interactive():
    """Get video file path interactively."""
    safe_print("\n" + "-" * 50)
    safe_print("  Select a Video File")
    safe_print("-" * 50)
    safe_print("\nOptions:")
    safe_print("  [1] Browse for file (opens file dialog)")
    safe_print("  [2] Enter path manually")
    safe_print("  [3] Exit")
    safe_print("")
    
    while True:
        choice = input("Choose option [1/2/3]: ").strip()
        
        if choice == "1":
            safe_print("\n[i] Opening file browser...")
            file_path = open_file_dialog()
            if file_path:
                return file_path
            else:
                print_warning("No file selected. Try again.\n")
                
        elif choice == "2":
            path_str = input("\nEnter video file path: ").strip()
            if path_str:
                # Remove quotes if present
                path_str = path_str.strip('"').strip("'")
                file_path = Path(path_str)
                if file_path.exists():
                    return file_path
                else:
                    print_error(f"File not found: {path_str}\n")
            else:
                print_warning("No path entered.\n")
                
        elif choice == "3":
            safe_print("\nGoodbye!")
            sys.exit(0)
        else:
            print_warning("Invalid choice. Enter 1, 2, or 3.\n")


def check_python_version():
    if sys.version_info < MIN_PYTHON:
        print_error(f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ required.")
        return False
    print_success(f"Python {sys.version_info[0]}.{sys.version_info[1]}.{sys.version_info[2]}")
    return True


def check_ffmpeg():
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path is None:
        print_error("ffmpeg not found in PATH!")
        safe_print("\n[*] Installation instructions:")
        safe_print("   Windows: Download from https://ffmpeg.org/download.html")
        safe_print("            Extract and add 'bin' folder to System PATH")
        safe_print("   Linux:   sudo apt install ffmpeg")
        safe_print("   macOS:   brew install ffmpeg\n")
        return False
    print_success(f"ffmpeg found: {ffmpeg_path}")
    return True


def check_package(import_name, pip_name):
    try:
        __import__(import_name)
        return True
    except ImportError:
        return False


def install_package(pip_name):
    try:
        from rich.console import Console
        console = Console()
        status = console.status(f"[bold cyan]Installing {pip_name}...", spinner="bouncingBar")
        status.start()
    except:
        status = None
        print_info(f"Installing {pip_name}...")

    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", pip_name, "-q"],
            capture_output=True, text=True, timeout=300,
        )
        if status:
            status.stop()
        return result.returncode == 0
    except Exception as e:
        if status:
            status.stop()
        print_error(f"Failed to install {pip_name}: {e}")
        return False


def check_and_install_packages():
    all_ok = True
    for import_name, pip_name in REQUIRED_PACKAGES:
        if check_package(import_name, pip_name):
            print_success(f"{pip_name} is installed")
        else:
            print_warning(f"{pip_name} is missing")
            response = input(f"   Install {pip_name} now? [Y/n]: ").strip().lower()
            if response in ("", "y", "yes"):
                if install_package(pip_name):
                    print_success(f"{pip_name} installed successfully!")
                else:
                    print_error(f"Failed to install {pip_name}")
                    all_ok = False
            else:
                print_info(f"Skipped. Install manually: pip install {pip_name}")
                all_ok = False
    return all_ok


def run_transcription(video_path):
    """Run transcription on selected video."""
    try:
        # Add src to path
        src_path = Path(__file__).parent / "src"
        if str(src_path) not in sys.path:
             sys.path.insert(0, str(src_path))

        # Show loading status for heavy imports
        try:
            from rich.console import Console
            console = Console()
            loading_status = console.status("[bold cyan]Loading application components (Whisper, Torch, etc.)...", spinner="dots")
            loading_status.start()
        except:
            loading_status = None
            safe_print("[i] Loading application components...")

        try:
            from transcription_project.main import TranscriptionApp
            
            # Modify sys.argv to include the video path
            sys.argv = [sys.argv[0], str(video_path)]
            
            app = TranscriptionApp()
            if loading_status:
                loading_status.stop()
            return app.run()
        finally:
            if loading_status:
                loading_status.stop()

        
    except ImportError as e:
        print_error(f"Import error: {e}")
        return 1
    except KeyboardInterrupt:
        safe_print("\n\n[!] Interrupted by user")
        return 0
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        return 1


def main():
    """Main entry point with pre-flight checks and interactive file selection."""
    print_banner()
    
    # v5.0: Run comprehensive pre-flight checks
    safe_print("[*] Running Pre-Flight System Checks...\n")
    
    # Add src to path for preflight import
    src_path = Path(__file__).parent / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    
    try:
        from transcription_project.preflight import run_preflight_checks, print_preflight_report
        
        all_passed, results = run_preflight_checks()
        print_preflight_report(results, use_rich=True)
        
        if not all_passed:
            safe_print("\n")
            sys.exit(1)
        
        safe_print("")
    except ImportError as e:
        # Fallback to basic checks if preflight module not available
        safe_print(f"[!] Warning: Could not load preflight module: {e}")
        safe_print("[*] Falling back to basic checks...\n")
        if not check_python_version():
            sys.exit(1)
        if not check_ffmpeg():
            sys.exit(1)
    
    # Check and install Python packages
    if not check_and_install_packages():
        print_error("Missing dependencies.")
        sys.exit(1)
    
    safe_print("\n" + "-" * 40)
    safe_print("[OK] All pre-flight checks passed!")
    safe_print("-" * 40)
    
    # v3.2: Check for --diagnose flag first
    if "--diagnose" in sys.argv:
        safe_print("\n[i] Running diagnostics mode...\n")
        try:
            src_path = Path(__file__).parent / "src"
            if str(src_path) not in sys.path:
                sys.path.insert(0, str(src_path))
            from transcription_project.diagnostics import run_diagnostics
            from rich.console import Console
            sys.exit(run_diagnostics(Console()))
        except Exception as e:
            print_error(f"Failed to run diagnostics: {e}")
            sys.exit(1)
    
    # Check if video path was provided as argument
    if len(sys.argv) > 1 and sys.argv[1] not in ("-h", "--help", "--diagnose"):
        video_path = Path(sys.argv[1])
        if video_path.exists():
            safe_print(f"\n[i] Selected: {video_path.name}\n")
            sys.exit(run_transcription(video_path))
        else:
            print_error(f"File not found: {sys.argv[1]}")
    
    # Interactive file selection
    video_path = get_video_file_interactive()
    
    if video_path:
        safe_print(f"\n[i] Selected: {video_path.name}\n")
        sys.exit(run_transcription(video_path))
    else:
        print_error("No file selected.")
        sys.exit(1)


if __name__ == "__main__":
    main()
