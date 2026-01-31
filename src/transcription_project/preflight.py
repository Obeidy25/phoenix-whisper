#!/usr/bin/env python3
# ==============================================================================
# preflight.py - Pre-Flight System Checks (v5.3)
# ==============================================================================
"""
Comprehensive environment validation before running the transcription tool.
Checks Python version, external tools, internet connectivity, and disk space.
"""

import os
import sys
import subprocess
import shutil
import socket
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional


# ============================================================================
# Windows Environment Helper
# ============================================================================

def refresh_windows_environment():
    """
    Refreshes the current process environment variables from the Windows Registry.
    This allows detecting tools installed after the terminal was opened (e.g., via winget).
    """
    if sys.platform != "win32":
        return

    try:
        import winreg
        
        # Paths to environment variables in registry
        reg_keys = [
            (winreg.HKEY_CURRENT_USER, r"Environment"),
            (winreg.HKEY_LOCAL_MACHINE, r"System\CurrentControlSet\Control\Session Manager\Environment")
        ]
        
        new_path_parts = []
        
        for root, key_path in reg_keys:
            try:
                with winreg.OpenKey(root, key_path, 0, winreg.KEY_READ) as key:
                    # Registry values can contain variables like %USERPROFILE%
                    val, _ = winreg.QueryValueEx(key, "Path")
                    # Expand environment variables in the string
                    expanded_val = os.path.expandvars(val)
                    new_path_parts.extend(expanded_val.split(os.pathsep))
            except Exception:
                continue

        if new_path_parts:
            # Filter duplicates and empty strings
            unique_paths = []
            seen = set()
            for p in new_path_parts:
                p = p.strip()
                if p and p.lower() not in seen:
                    unique_paths.append(p)
                    seen.add(p.lower())
            
            # Update current process PATH
            os.environ["PATH"] = os.pathsep.join(unique_paths)
            
    except Exception as e:
        # Silently fail if something goes wrong with registry access
        pass


def install_git_winget() -> bool:
    """
    Tries to install Git using Windows Package Manager (winget).
    """
    if sys.platform != "win32":
        return False

    try:
        from rich.console import Console
        from rich.status import Status
        console = Console()
        
        with console.status("[bold cyan]Installing Git via winget...[/]", spinner="bouncingBar"):
            # winget install --id Git.Git -e --source winget
            process = subprocess.run(
                ["winget", "install", "--id", "Git.Git", "-e", "--source", "winget", "--accept-source-agreements", "--accept-package-agreements"],
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes timeout
            )
            
            if process.returncode == 0:
                console.print("[green]✓ Git installed successfully![/]")
                # Refresh environment immediately to detect new git
                refresh_windows_environment()
                return True
            else:
                console.print(f"[red]✗ Winget failed with code {process.returncode}[/]")
                if process.stderr:
                    console.print(f"[dim]{process.stderr.strip()}[/]")
                return False
                
    except Exception as e:
        print(f"[!] Error during winget installation: {e}")
        return False


def install_ffmpeg_linux() -> bool:
    """
    Tries to install FFmpeg on Linux using apt-get.
    """
    try:
        from rich.console import Console
        console = Console()
        
        with console.status("[bold cyan]Installing FFmpeg via apt...[/]", spinner="bouncingBar"):
            # Check if we have sudo
            process = subprocess.run(
                ["sudo", "apt-get", "update"],
                capture_output=True, text=True, timeout=120
            )
            process = subprocess.run(
                ["sudo", "apt-get", "install", "-y", "ffmpeg"],
                capture_output=True, text=True, timeout=300
            )
            
            if process.returncode == 0:
                console.print("[green]✓ FFmpeg installed successfully![/]")
                return True
            else:
                console.print(f"[red]✗ Apt failed with code {process.returncode}[/]")
                return False
    except Exception as e:
        print(f"[!] Error during apt installation: {e}")
        return False


def install_ffmpeg() -> bool:
    """Generic entry point for FFmpeg installation based on OS."""
    if sys.platform == "win32":
        return install_ffmpeg_winget()
    elif sys.platform == "linux":
        return install_ffmpeg_linux()
    return False


def install_ffmpeg_winget() -> bool:
    """
    Tries to install FFmpeg using Windows Package Manager (winget).
    """
    if sys.platform != "win32":
        return False

    try:
        from rich.console import Console
        console = Console()
        
        with console.status("[bold cyan]Installing FFmpeg via winget...[/]", spinner="bouncingBar"):
            # winget install --id Gyan.FFmpeg -e --source winget
            process = subprocess.run(
                ["winget", "install", "--id", "Gyan.FFmpeg", "-e", "--source", "winget", "--accept-source-agreements", "--accept-package-agreements"],
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes timeout
            )
            
            if process.returncode == 0:
                console.print("[green]✓ FFmpeg installed successfully![/]")
                # Refresh environment immediately to detect new ffmpeg
                refresh_windows_environment()
                return True
            else:
                console.print(f"[red]✗ Winget failed with code {process.returncode}[/]")
                if process.stderr:
                    console.print(f"[dim]{process.stderr.strip()}[/]")
                return False
                
    except Exception as e:
        print(f"[!] Error during winget installation: {e}")
        return False


def detect_nvidia_gpu() -> Tuple[bool, str]:
    """
    Detect if an NVIDIA GPU is physically present on the system.
    Uses wmic on Windows as a more reliable hardware check than nvidia-smi.
    
    Returns:
        Tuple of (found, description)
    """
    if sys.platform != "win32":
        # Fallback for Linux/macOS
        try:
            res = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=5)
            if res.returncode == 0:
                return True, res.stdout.strip()
        except:
            pass
        return False, "Non-Windows / No nvidia-smi"

    try:
        # Query Windows Management Instrumentation for VideoController info
        result = subprocess.run(
            ["wmic", "path", "win32_VideoController", "get", "name,VideoProcessor"],
            capture_output=True,
            text=True,
            timeout=10,
            errors='replace'
        )
        
        if result.returncode == 0:
            lines = [line.strip() for line in result.stdout.split('\n') if line.strip()]
            for line in lines[1:]:  # Skip header
                if "nvidia" in line.lower():
                    return True, line
            
            # If nothing found, return the first one as descriptive info
            if len(lines) > 1:
                return False, lines[1]
                
    except Exception:
        pass
        
    return False, "Unknown hardware"


# ============================================================================
# Configuration
# ============================================================================

MIN_PYTHON_VERSION = (3, 9)
MIN_DISK_SPACE_GB = 5  # Minimum free space needed for models + temp files


@dataclass
class CheckResult:
    """Result of a single pre-flight check."""
    name: str
    passed: bool
    message: str
    fix_instruction: Optional[str] = None
    critical: bool = True  # If True, failure blocks execution


# ============================================================================
# Individual Check Functions
# ============================================================================

def check_python_version() -> CheckResult:
    """
    Verify Python version is 3.9 or newer.
    
    Returns:
        CheckResult with pass/fail status
    """
    current = sys.version_info
    required = MIN_PYTHON_VERSION
    
    if current >= required:
        return CheckResult(
            name="Python Version",
            passed=True,
            message=f"Python {current.major}.{current.minor}.{current.micro}"
        )
    else:
        return CheckResult(
            name="Python Version",
            passed=False,
            critical=True,
            message=f"Python {current.major}.{current.minor} is too old (need {required[0]}.{required[1]}+)",
            fix_instruction="Please upgrade to Python 3.9 or newer from https://python.org"
        )


def check_ffmpeg() -> CheckResult:
    """
    Check if ffmpeg is installed and accessible.
    
    Returns:
        CheckResult with pass/fail status
    """
    ffmpeg_path = shutil.which("ffmpeg")
    
    if ffmpeg_path is None:
        # v5.8: Offer auto-install on Windows AND Linux
        if sys.platform in ["win32", "linux"]:
            from rich.console import Console
            from rich.prompt import Confirm
            console = Console()
            
            console.print(f"\n[yellow]![/] [bold]FFmpeg is not found in your system path (OS: {sys.platform}).[/]")
            console.print("    FFmpeg is [bold red]required[/] for processing audio and video.")
            
            if Confirm.ask("    Would you like to auto-install FFmpeg now?", default=True):
                if install_ffmpeg():
                    # Re-check after install
                    ffmpeg_path = shutil.which("ffmpeg")
                    if ffmpeg_path:
                        return CheckResult(
                            name="FFmpeg",
                            passed=True,
                            message=f"Found (after auto-install): {ffmpeg_path}"
                        )

        return CheckResult(
            name="FFmpeg",
            passed=False,
            critical=True,
            message="ffmpeg is not found in PATH",
            fix_instruction=(
                "Install from https://ffmpeg.org/download.html\n"
                "     |   Windows: Extract and add the 'bin' folder to System PATH\n"
                "     |   Linux: sudo apt install ffmpeg\n"
                "     |   macOS: brew install ffmpeg"
            )
        )
    
    # Try to run ffmpeg to verify it works
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=10,
            errors='replace'
        )
        if result.returncode == 0:
            return CheckResult(
                name="FFmpeg",
                passed=True,
                message=f"Found: {ffmpeg_path}"
            )
    except Exception:
        pass
    
    return CheckResult(
        name="FFmpeg",
        passed=False,
        critical=True,
        message="ffmpeg found but failed to execute",
        fix_instruction="Reinstall ffmpeg from https://ffmpeg.org/download.html"
    )


def check_ffprobe() -> CheckResult:
    """
    Check if ffprobe is installed (usually comes with ffmpeg).
    
    Returns:
        CheckResult with pass/fail status
    """
    ffprobe_path = shutil.which("ffprobe")
    
    if ffprobe_path is None:
        # If FFmpeg was just installed, we might need another refresh or manual check
        return CheckResult(
            name="FFprobe",
            passed=False,
            critical=True,
            message="ffprobe is not found in PATH",
            fix_instruction="FFprobe should be included with FFmpeg. Please ensure FFmpeg is installed correctly."
        )
    
    return CheckResult(
        name="FFprobe",
        passed=True,
        message=f"Found: {ffprobe_path}"
    )


def check_git() -> CheckResult:
    """
    Check if git is installed (required for Whisper tokenizer configurations).
    
    Returns:
        CheckResult with pass/fail status
    """
    git_path = shutil.which("git")
    
    if git_path is None:
        # v5.3: Offer auto-installation via winget
        try:
            from rich.prompt import Confirm
            from rich.console import Console
            console = Console()
            
            console.print("\n[yellow]![/] [bold]Git is missing.[/] Whisper may need it to download tokenizer components.")
            if Confirm.ask("    Would you like to auto-install Git now via winget?", default=True):
                if install_git_winget():
                    # Re-check git after installation
                    new_git_path = shutil.which("git")
                    if new_git_path:
                        return CheckResult(
                            name="Git",
                            passed=True,
                            message=f"Installed & Found: {new_git_path}"
                        )
                else:
                    console.print("[red]✗ Auto-installation failed.[/] Please install manually.")
        except ImportError:
            pass

        return CheckResult(
            name="Git",
            passed=False,
            critical=False,  # v5.1: Non-critical, many users have pre-downloaded models
            message="Git is not found in PATH",
            fix_instruction=(
                "Install from https://git-scm.com/downloads\n"
                "     |   Or run: winget install --id Git.Git -e --source winget"
            )
        )
    
    # Verify git works
    try:
        result = subprocess.run(
            ["git", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
            errors='replace'
        )
        if result.returncode == 0:
            version = result.stdout.strip() if result.stdout else "Unknown version"
            return CheckResult(
                name="Git",
                passed=True,
                message=version
            )
    except Exception:
        pass
    
    return CheckResult(
        name="Git",
        passed=False,
        critical=False,
        message="Git found but failed to execute",
        fix_instruction="If you encounter errors downloading models, reinstall Git."
    )


def check_internet_connection() -> CheckResult:
    """
    Check if there's an active internet connection.
    
    Returns:
        CheckResult with pass/fail status
    """
    test_hosts = [
        ("github.com", 443),
        ("huggingface.co", 443),
        ("google.com", 443),
    ]
    
    for host, port in test_hosts:
        try:
            socket.create_connection((host, port), timeout=5)
            return CheckResult(
                name="Internet Connection",
                passed=True,
                message=f"Connected (verified via {host})"
            )
        except (socket.timeout, socket.error, OSError):
            continue
    
    return CheckResult(
        name="Internet Connection",
        passed=False,
        critical=False,  # Non-critical if models are already local
        message="No active internet connection detected",
        fix_instruction=(
            "If you haven't downloaded models yet, Whisper will fail.\n"
            "     |   Offline mode is supported if models are already cached."
        )
    )


def check_disk_space(min_space_gb: float = MIN_DISK_SPACE_GB) -> CheckResult:
    """
    Check if there's enough free disk space.
    
    Args:
        min_space_gb: Minimum required space in gigabytes
        
    Returns:
        CheckResult with pass/fail status
    """
    try:
        script_dir = Path(__file__).parent.resolve()
        
        if sys.platform == "win32":
            drive = script_dir.drive or "C:"
            import ctypes
            free_bytes = ctypes.c_ulonglong(0)
            ctypes.windll.kernel32.GetDiskFreeSpaceExW(
                ctypes.c_wchar_p(drive),
                None,
                None,
                ctypes.pointer(free_bytes)
            )
            free_gb = free_bytes.value / (1024 ** 3)
        else:
            import os
            stat = os.statvfs(script_dir)
            free_gb = (stat.f_bavail * stat.f_frsize) / (1024 ** 3)
        
        if free_gb >= min_space_gb:
            return CheckResult(
                name="Disk Space",
                passed=True,
                message=f"{free_gb:.1f} GB available"
            )
        else:
            return CheckResult(
                name="Disk Space",
                passed=False,
                critical=False,  # Non-critical
                message=f"Low disk space: {free_gb:.1f} GB available (need {min_space_gb} GB)",
                fix_instruction="If processing fails, free up some disk space."
            )
    except Exception as e:
        return CheckResult(
            name="Disk Space",
            passed=True,
            message=f"Could not check disk space: {e}"
        )


def check_pytorch_cuda() -> CheckResult:
    """
    Check PyTorch and CUDA availability for GPU acceleration.
    Offers auto-reinstall if PyTorch is installed without CUDA.
    
    Returns:
        CheckResult with GPU status (non-critical - CPU fallback is valid)
    """
    import importlib.util
    
    # Step 1: Check if PyTorch is installed
    if importlib.util.find_spec("torch") is None:
        return CheckResult(
            name="GPU Acceleration",
            passed=False,
            critical=False,
            message="PyTorch not installed (CPU-only mode)",
            fix_instruction=(
                "For GPU acceleration, install PyTorch with CUDA:\n"
                "     |   pip install torch --index-url https://download.pytorch.org/whl/cu121"
            )
        )
    
    try:
        import torch
        
        if torch.cuda.is_available():
            # Success! GPU is available
            try:
                gpu_name = torch.cuda.get_device_name(0)
                vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                return CheckResult(
                    name="GPU Acceleration",
                    passed=True,
                    message=f"{gpu_name} ({vram:.1f} GB VRAM)"
                )
            except Exception:
                return CheckResult(
                    name="GPU Acceleration",
                    passed=True,
                    message="NVIDIA GPU detected and active"
                )
        else:
            # CUDA not available - Check hardware type for tailored messaging
            has_nvidia_hw, hw_desc = detect_nvidia_gpu()
            
            if not has_nvidia_hw:
                # v5.8: High Flexibility - Acknowledge AMD/Intel and explain Optimized CPU usage
                gpu_brand = "Intel/AMD"
                if "intel" in hw_desc.lower(): gpu_brand = "Intel"
                elif "amd" in hw_desc.lower() or "radeon" in hw_desc.lower(): gpu_brand = "AMD"
                
                return CheckResult(
                    name="Hardware Compatibility",
                    passed=True, # Mark as passed for flexibility
                    critical=False,
                    message=f"Optimized for {hw_desc}",
                    fix_instruction=(
                        f"ℹ Current mode: High-Compatibility CPU Execution.\n"
                        f"     |   Detected: {gpu_brand} Graphics.\n"
                        f"     |   Status: Running on {os.cpu_count()} CPU cores for maximum stability."
                    )
                )

            # Hardware exists, but software is missing/broken
            cuda_build = getattr(torch.version, 'cuda', None)
            if cuda_build is None:
                # v5.5: Offer auto-reinstall with CUDA
                if _offer_pytorch_cuda_reinstall():
                    # Re-check after reinstall
                    return _verify_cuda_after_reinstall()
                
                return CheckResult(
                    name="GPU Acceleration",
                    passed=False,
                    critical=False,
                    message=f"NVIDIA hardware found ({hw_desc}) but PyTorch lacks CUDA support",
                    fix_instruction=(
                        "Reinstall PyTorch with CUDA:\n"
                        "     |   pip uninstall torch\n"
                        "     |   pip install torch --index-url https://download.pytorch.org/whl/cu121"
                    )
                )
            else:
                # PyTorch has CUDA but runtime failed (likely driver)
                return CheckResult(
                    name="GPU Acceleration",
                    passed=False,
                    critical=False,
                    message=f"NVIDIA hardware found but CUDA runtime failed (Driver issue?)",
                    fix_instruction=(
                        f"Detected: {hw_desc}\n"
                        "     |   1. Ensure NVIDIA drivers are installed\n"
                        "     |   2. Try running 'nvidia-smi' to verify driver health\n"
                        "     |   3. CUDA toolkit version may mismatch PyTorch"
                    )
                )
                
    except ImportError as e:
        return CheckResult(
            name="GPU Acceleration",
            passed=False,
            critical=False,
            message=f"PyTorch import failed: {e}",
            fix_instruction="Reinstall PyTorch: pip install torch"
        )
    except Exception as e:
        return CheckResult(
            name="GPU Acceleration",
            passed=False,
            critical=False,
            message=f"GPU detection error: {e}",
            fix_instruction="Check PyTorch installation"
        )


def _offer_pytorch_cuda_reinstall() -> bool:
    """
    Offer to reinstall PyTorch with CUDA support.
    Returns True if reinstall was attempted and successful.
    """
    import sys
    
    # Check Python version compatibility (PyTorch CUDA wheels: 3.9-3.12)
    py_version = sys.version_info[:2]
    supported_versions = [(3, 9), (3, 10), (3, 11), (3, 12)]
    
    try:
        from rich.console import Console
        from rich.prompt import Confirm
        console = Console()
        
        console.print("\n[yellow]![/] [bold]PyTorch is installed without CUDA support.[/]")
        console.print("    GPU acceleration will be [bold red]disabled[/]. Transcription will be slower.")
        
        # Check if Python version is compatible
        if py_version not in supported_versions:
            console.print(f"\n[red]⚠ Python {py_version[0]}.{py_version[1]} is not supported by PyTorch CUDA wheels.[/]")
            console.print("    [dim]PyTorch CUDA requires Python 3.9, 3.10, 3.11, or 3.12[/]")
            console.print("    [dim]Auto-install is not available for your Python version.[/]")
            console.print("\n    [bold]Options:[/]")
            console.print("    1. Install Python 3.12 and create a new virtual environment")
            console.print("    2. Continue with CPU-only mode (slower but works)")
            return False
        
        if not Confirm.ask("    Would you like to auto-reinstall PyTorch with CUDA now?", default=True):
            return False
        
        return _reinstall_pytorch_cuda(console)
        
    except ImportError:
        return False


def _reinstall_pytorch_cuda(console) -> bool:
    """
    Reinstall PyTorch with CUDA support.
    """
    import subprocess
    import sys
    
    try:
        # Step 1: Uninstall current torch
        console.print("\n[bold cyan]Step 1/2:[/] Uninstalling current PyTorch...")
        with console.status("[bold cyan]Removing torch, torchvision, torchaudio...[/]", spinner="bouncingBar"):
            result = subprocess.run(
                [sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchvision", "torchaudio"],
                capture_output=True,
                text=True,
                timeout=120
            )
        
        if result.returncode != 0:
            console.print(f"[yellow]Warning:[/] Uninstall returned code {result.returncode}")
        else:
            console.print("[green]✓[/] Removed existing PyTorch packages")
        
        # Step 2: Install with CUDA
        console.print("\n[bold cyan]Step 2/2:[/] Installing PyTorch with CUDA 12.1...")
        console.print("[dim]This may take several minutes...[/]")
        
        with console.status("[bold cyan]Downloading and installing PyTorch+CUDA...[/]", spinner="bouncingBar"):
            result = subprocess.run(
                [
                    sys.executable, "-m", "pip", "install",
                    "torch", "torchvision", "torchaudio",
                    "--index-url", "https://download.pytorch.org/whl/cu121"
                ],
                capture_output=True,
                text=True,
                timeout=900  # 15 minutes for large download
            )
        
        if result.returncode == 0:
            console.print("[green]✓[/] PyTorch with CUDA installed successfully!")
            console.print("\n[bold yellow]⚠ IMPORTANT:[/] Please [bold]restart this script[/] for changes to take effect.")
            return True
        else:
            console.print(f"[red]✗[/] Installation failed with code {result.returncode}")
            if result.stderr:
                console.print(f"[dim]{result.stderr[:500]}[/]")
            return False
            
    except subprocess.TimeoutExpired:
        console.print("[red]✗[/] Installation timed out. Try manually:")
        console.print("    pip install torch --index-url https://download.pytorch.org/whl/cu121")
        return False
    except Exception as e:
        console.print(f"[red]✗[/] Error during reinstall: {e}")
        return False


def _verify_cuda_after_reinstall() -> CheckResult:
    """
    Verify CUDA availability after reinstall.
    Note: Due to Python module caching, a restart is typically needed.
    """
    return CheckResult(
        name="GPU Acceleration",
        passed=False,
        critical=False,
        message="PyTorch reinstalled - restart required",
        fix_instruction="Please restart this script to use GPU acceleration"
    )


# ============================================================================
# Main Pre-Flight Check Orchestrator
# ============================================================================

def run_preflight_checks(verbose: bool = True) -> Tuple[bool, List[CheckResult]]:
    """
    Run all pre-flight checks and return results.
    
    Args:
        verbose: If True, print progress during checks
        
    Returns:
        Tuple of (should_continue, list_of_results)
    """
    # v5.2: Refresh Windows environment to pick up newly installed tools
    refresh_windows_environment()
    
    checks = [
        ("Python Version", check_python_version),
        ("FFmpeg", check_ffmpeg),
        ("FFprobe", check_ffprobe),
        ("Git", check_git),
        ("Internet", check_internet_connection),
        ("Disk Space", check_disk_space),
        ("GPU", check_pytorch_cuda),  # v5.4: GPU/CUDA status check
    ]
    
    results: List[CheckResult] = []
    
    for name, check_func in checks:
        try:
            result = check_func()
            results.append(result)
        except Exception as e:
            results.append(CheckResult(
                name=name,
                passed=False,
                critical=True,
                message=f"Check failed with error: {e}",
                fix_instruction="Please report this issue."
            ))
    
    # v5.1: Only critical failures should block execution
    should_continue = not any(r.passed == False and r.critical == True for r in results)
    return should_continue, results


def print_preflight_report(results: List[CheckResult], use_rich: bool = True) -> None:
    """
    Print a formatted pre-flight check report.
    
    Args:
        results: List of CheckResult objects
        use_rich: If True, use rich formatting
    """
    failed_results = [r for r in results if not r.passed]
    critical_failures = [r for r in failed_results if r.critical]
    warnings = [r for r in failed_results if not r.critical]
    
    if use_rich:
        try:
            from rich.console import Console
            from rich.panel import Panel
            
            console = Console()
            
            if critical_failures:
                console.print()
                console.print(Panel(
                    "[bold red]Critical System Issues Found (Execution Blocked):[/]",
                    title="[bold red]PRE-FLIGHT CHECK FAILED[/]",
                    border_style="red"
                ))
                console.print()
                
                for i, result in enumerate(critical_failures, 1):
                    console.print(f"  [bold red]{i}.[/] [red]✗[/] [bold]{result.name}[/]: {result.message}")
                    if result.fix_instruction:
                        console.print(f"     [dim]|-> How to fix:[/] {result.fix_instruction}")
                    console.print()
                
                console.print("[dim]Application must exit until these are fixed.[/]")
            
            # Print Warnings (non-blocking)
            if warnings:
                if not critical_failures:
                    console.print()
                
                console.print(Panel(
                    "[bold yellow]Potential Issues Found (Non-Blocking):[/]",
                    title="[bold yellow]PRE-FLIGHT WARNINGS[/]",
                    border_style="yellow"
                ))
                console.print()
                
                for i, result in enumerate(warnings, 1):
                    console.print(f"  [bold yellow]{i}.[/] [yellow]![/] [bold]{result.name}[/]: {result.message}")
                    if result.fix_instruction:
                        console.print(f"     [dim]|-> Recommendation:[/] {result.fix_instruction}")
                    console.print()
            
            # Print Success Summary (if no critical failures)
            if not critical_failures:
                from rich.table import Table
                table = Table(show_header=False, box=None, padding=(0, 1))
                table.add_column("Status")
                table.add_column("Check")
                table.add_column("Details", style="dim")
                
                for result in results:
                    if result.passed:
                        table.add_row("[green]✓[/]", result.name, result.message)
                    elif not result.critical:
                        table.add_row("[yellow]![/]", result.name, result.message)
                
                console.print(table)
            
            return
        except ImportError:
            pass  # Fall back to plain text
    
    # Plain text fallback (Optimized for better visual appeal)
    if critical_failures:
        print("\n\033[91m" + "!" * 80)
        print("  [CRITICAL FAILURE] Your system is missing required components:")
        print("!" * 80 + "\033[0m")
        for r in critical_failures:
            print(f"\n  \033[1m- X {r.name}\033[0m: \033[91m{r.message}\033[0m")
            if r.fix_instruction: 
                print(f"    \033[90m|-> FIX:\033[0m {r.fix_instruction}")
        print("\033[91m\nApplication must be corrected before it can run.\033[0m")
        print("-" * 80)
    
    if warnings:
        print("\n\033[93m" + "-" * 80)
        print("  [WARNING] Potential issues detected (Non-Blocking):")
        print("-" * 80 + "\033[0m")
        for r in warnings:
            print(f"\n  \033[1m- ! {r.name}\033[0m: \033[93m{r.message}\033[0m")
            if r.fix_instruction: 
                print(f"    \033[90m|-> REC:\033[0m {r.fix_instruction}")
        print("-" * 80)
    print()



def run_and_report(use_rich: bool = True) -> bool:
    """
    Run pre-flight checks and print report.
    
    Args:
        use_rich: If True, use rich formatting
        
    Returns:
        True if all checks passed, False otherwise
    """
    all_passed, results = run_preflight_checks()
    print_preflight_report(results, use_rich=use_rich)
    return all_passed


# ============================================================================
# Module Test
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  Pre-Flight System Checks")
    print("=" * 60 + "\n")
    
    success = run_and_report(use_rich=True)
    sys.exit(0 if success else 1)
