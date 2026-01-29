# Video Transcription Tool v2.0

Professional, resumable, and concurrent video transcription application.

## Features

- 🚀 **Concurrent Processing** - Process multiple chunks in parallel
- 💾 **Stop/Resume** - Checkpoint progress and resume interrupted jobs  
- 📊 **RAM Profiles** - Dynamic chunk sizing (low/medium/high)
- 🎨 **Rich UI** - Beautiful progress bars and color-coded output
- 🛡️ **Graceful Exit** - Safe shutdown on Ctrl+C

## Prerequisites

1. **Python 3.10+** in a virtual environment
2. **FFmpeg** installed and in PATH
3. **openai-whisper** and **rich** packages

## Installation

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt
```

## Usage

This project supports multiple ways to run the application:

### Recommended
```bash
# Auto-setup and run (includes file browser)
python run.py
```

### Standard CLI
```bash
# Basic usage
python main.py video.mp4

# With options
python main.py video.mp4 --model medium --num-workers 4

# Run as module
python -m transcription_project.cli video.mp4
```

### Options

| Option | Description |
|--------|-------------|
| `--model` | Whisper model (tiny/base/small/medium/large) |
| `--language` | Source language code (auto-detected if not set) |
| `--num-workers` | Concurrent workers (default: 1) |
| `--ram-profile` | low/medium/high chunk sizes |
| `--resume` | Resume from checkpoint |
| `--output-dir` | Output directory |
| `--keep-temp` | Keep temporary files |
| `--verbose` | Verbose output |

## Project Structure

Refactored to `src` layout for better organization:

```
├── run.py                 # Smart launcher (root entry point)
├── main.py                # Shim for backward compatibility
├── cli.py                 # Shim for backward compatibility
├── setup.bat              # Windows setup script
├── transcribe.bat         # Drag-and-drop script
└── src/
    └── transcription_project/
        ├── main.py        # Core application logic
        ├── config.py      # Configuration
        ├── cli.py         # Argument parsing
        └── ...            # Other modules
```
