#!/usr/bin/env python3
import sys
from pathlib import Path

# Add src to path so we can import the package
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

try:
    from transcription_project.main import main
except ImportError as e:
    print(f"Error importing from transcription_project: {e}")
    sys.exit(1)

if __name__ == "__main__":
    sys.exit(main())
