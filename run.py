#!/usr/bin/env python3
# ==============================================================================
# run.py - Zero-Configuration Smart Launcher (v4.1)
# ==============================================================================
"""
The ONLY script users should run. Ensures proper UTF-8 encoding on Windows.

This launcher:
1. Checks if PYTHONUTF8 environment variable is set to '1'
2. If NOT set (first-run): Re-launches app_logic.py as a subprocess with 
   PYTHONUTF8='1' in the environment
3. If set (normal run): Imports and runs app_logic.main() directly
"""

import os
import sys
import subprocess
from pathlib import Path


def is_venv():
    """Check if the script is running inside any virtual environment."""
    # Standard Python 3 venv check
    in_venv = (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    # Legacy virtualenv check
    in_venv = in_venv or hasattr(sys, 'real_prefix')
    # Environment variables set by 'activate' scripts (venv, virtualenv, conda)
    in_venv = in_venv or bool(os.environ.get('VIRTUAL_ENV'))
    in_venv = in_venv or bool(os.environ.get('CONDA_PREFIX'))
    return in_venv


def get_venv_python():
    """Get the path to the python executable in the local .venv folder."""
    script_dir = Path(__file__).parent.resolve()
    # Priority: Currently running interpreter (if already in venv)
    if is_venv():
        return Path(sys.executable)
        
    # Fallback: Local .venv folder
    if sys.platform == "win32":
        return script_dir / ".venv" / "Scripts" / "python.exe"
    else:
        return script_dir / ".venv" / "bin" / "python"


def print_banner():
    """Print a professional ANSI-decorated banner."""
    print("\033[96m" + "=" * 60)
    print("  \033[93m[*] Video Transcription Tool - Setup Wizard")
    print("  \033[90mUltimate Automation - Zero Configuration")
    print("\033[96m" + "=" * 60 + "\033[0m")


def ensure_venv():
    """Ensure a virtual environment exists and we are running inside it."""
    if is_venv():
        return True

    venv_python = get_venv_python()
    script_dir = Path(__file__).parent.resolve()

    if not venv_python.exists():
        print_banner()
        print("\033[94m[i]\033[0m Isolated environment not found. Creating .venv...")
        try:
            subprocess.run([sys.executable, "-m", "venv", ".venv"], check=True, cwd=script_dir)
            print("\033[92m[✓]\033[0m Virtual environment created successfully.")
        except KeyboardInterrupt:
            print("\n\033[91m[X] Setup cancelled by user.\033[0m")
            sys.exit(1)
        except Exception as e:
            print(f"\n\033[91m[X] ERROR: Failed to create virtual environment: {e}\033[0m")
            return False

    # Re-launch this script using the venv's python
    try:
        # Pass through all command-line arguments
        result = subprocess.run([str(venv_python), __file__] + sys.argv[1:])
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        return False
    except Exception as e:
        print(f"\033[91m[X] ERROR: Failed to bootstrap from venv: {e}\033[0m")
        return False


def ensure_dependencies():
    """Ensure all required packages are installed in the current environment."""
    script_dir = Path(__file__).parent.resolve()
    
    # v5.8: Tie marker to the specific environment directory to prevent stale markers
    if is_venv():
        # Marker goes inside the actual venv directory
        marker_file = Path(sys.prefix) / ".dependencies_installed"
    else:
        # Fallback for system python (not recommended but safe)
        marker_file = script_dir / ".dependencies_installed"
    
    # 1. Quick bypass if already fully installed
    if marker_file.exists():
        return True
        
    # 2. Check if we can import key packages (failsafe)
    try:
        import whisper
        import rich
        # Create marker if it's missing but imports work
        marker_file.touch()
        return True
    except ImportError:
        print_banner()
        print("\033[94m[i]\033[0m Setting up dependencies. This only happens once...")
        
        req_file = script_dir / 'requirements.txt'
        if not req_file.exists():
            print(f"\033[91m[X] ERROR: requirements.txt not found at {req_file}\033[0m")
            return False
            
        try:
            # First, install rich for better UI in subsequent steps
            subprocess.run([sys.executable, "-m", "pip", "install", "rich"], check=True)
            
            # Now install the rest
            print("\033[94m[i]\033[0m Installing core components (Whisper, FFmpeg tools)...")
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(req_file)], check=True)
            
            # Create marker file to skip this next time
            marker_file.touch()
            
            print("\033[92m[✓]\033[0m All dependencies installed and verified.")
            print("\033[90m" + "-" * 60 + "\033[0m\n")
            return True
        except KeyboardInterrupt:
            print("\n\033[91m[X] Installation cancelled by user. Environment may be incomplete.\033[0m")
            sys.exit(1)
        except Exception as e:
            print(f"\n\033[91m[X] ERROR: Failed to install dependencies: {e}\033[0m")
            return False


def main():
    """Smart launcher entry point with isolation and auto-dep handling."""
    
    # 1. Isolation: Ensure we are in the .venv
    try:
        if not ensure_venv():
            return 1
    except KeyboardInterrupt:
        return 0
        
    # 2. Dependencies: Ensure requirements are met
    if not ensure_dependencies():
        return 1

    # 3. Environment: Configure UTF-8 for Windows
    if os.environ.get('PYTHONUTF8') == '1':
        script_dir = Path(__file__).parent.resolve()
        if str(script_dir) not in sys.path:
            sys.path.insert(0, str(script_dir))
        
        try:
            from app_logic import main as app_main
            return app_main()
        except KeyboardInterrupt:
            return 0
        except ImportError as e:
            print(f"\033[91m[X] ERROR: Could not import app_logic: {e}\033[0m")
            return 1
    else:
        # Restart with PYTHONUTF8 set
        env = os.environ.copy()
        env['PYTHONUTF8'] = '1'
        
        script_dir = Path(__file__).parent.resolve()
        app_logic_path = script_dir / 'app_logic.py'
        
        if not app_logic_path.exists():
            print(f"\033[91m[X] ERROR: app_logic.py not found at {app_logic_path}\033[0m")
            return 1
        
        try:
            result = subprocess.run(
                [sys.executable, str(app_logic_path)] + sys.argv[1:],
                env=env,
            )
            return result.returncode
        except KeyboardInterrupt:
            return 0
        except Exception as e:
            print(f"\033[91m[X] ERROR: Failed to launch application: {e}\033[0m")
            return 1


if __name__ == '__main__':
    sys.exit(main())
