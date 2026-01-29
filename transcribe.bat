@echo off
REM ==============================================================================
REM transcribe.bat - Easy Transcription Script
REM ==============================================================================
REM Drag and drop a video file onto this script to transcribe it

if "%~1"=="" (
    echo.
    echo ============================================================
    echo   Video Transcription Tool v2.0
    echo ============================================================
    echo.
    echo Usage: Drag and drop a video file onto this script
    echo    Or: transcribe.bat "path\to\video.mp4"
    echo.
    echo Options:
    echo   transcribe.bat video.mp4
    echo   transcribe.bat video.mp4 --model medium
    echo   transcribe.bat video.mp4 --num-workers 4
    echo.
    pause
    exit /b 0
)

echo.
echo Starting transcription for: %~nx1
echo.

python "%~dp0run.py" %*

if errorlevel 1 (
    echo.
    echo [!] Transcription encountered an error.
    echo.
)

pause
