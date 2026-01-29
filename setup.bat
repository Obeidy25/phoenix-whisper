@echo off
REM ==============================================================================
REM setup.bat - Easy Setup Script for Windows
REM ==============================================================================
REM Double-click this file to automatically set up the transcription tool

echo.
echo ============================================================
echo   Video Transcription Tool v2.0 - Setup
echo ============================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found!
    echo.
    echo Please install Python 3.8+ from https://python.org
    echo Make sure to check "Add Python to PATH" during installation
    echo.
    pause
    exit /b 1
)

echo [OK] Python found
echo.

REM Install requirements
echo Installing required packages...
echo.
pip install rich openai-whisper --quiet

if errorlevel 1 (
    echo.
    echo [WARNING] Some packages may have failed to install.
    echo Try running: pip install rich openai-whisper
    echo.
) else (
    echo.
    echo [OK] Packages installed successfully!
)

echo.
echo ============================================================
echo   Setup Complete!
echo ============================================================
echo.
echo To transcribe a video, run:
echo   python run.py video.mp4
echo.
echo Or drag and drop a video onto run.py
echo.
pause
