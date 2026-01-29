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


def main():
    """Smart launcher entry point."""
    
    # Check if UTF-8 mode is already enabled in the environment
    if os.environ.get('PYTHONUTF8') == '1':
        # ======================================================================
        # NORMAL RUN: UTF-8 is configured, run directly
        # ======================================================================
        # Add current directory to path for app_logic import
        script_dir = Path(__file__).parent.resolve()
        if str(script_dir) not in sys.path:
            sys.path.insert(0, str(script_dir))
        
        try:
            from app_logic import main as app_main
            return app_main()
        except ImportError as e:
            print(f"[X] ERROR: Could not import app_logic: {e}")
            print("    Make sure app_logic.py exists in the same directory as run.py")
            return 1
    else:
        # ======================================================================
        # FIRST-RUN: Need to restart with PYTHONUTF8 set
        # ======================================================================
        print("[i] First-time setup: Configuring optimal encoding for Windows...")
        
        # Create a copy of the current environment
        env = os.environ.copy()
        
        # Set PYTHONUTF8 for the subprocess
        env['PYTHONUTF8'] = '1'
        
        # Get the path to app_logic.py (in the same directory as this script)
        script_dir = Path(__file__).parent.resolve()
        app_logic_path = script_dir / 'app_logic.py'
        
        if not app_logic_path.exists():
            print(f"[X] ERROR: app_logic.py not found at {app_logic_path}")
            return 1
        
        # Re-launch app_logic.py with the modified environment
        # Pass through all command-line arguments
        try:
            result = subprocess.run(
                [sys.executable, str(app_logic_path)] + sys.argv[1:],
                env=env,
            )
            return result.returncode
        except KeyboardInterrupt:
            # Graceful handling of Ctrl+C during subprocess
            return 0
        except Exception as e:
            print(f"[X] ERROR: Failed to launch application: {e}")
            return 1


if __name__ == '__main__':
    sys.exit(main())
