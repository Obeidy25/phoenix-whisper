# ==============================================================================
# environment.py - Environment and Dependency Checking Module
# ==============================================================================
"""
Handles dependency checks for external tools (ffmpeg, whisper).
Validates the environment before starting any heavy processing.
"""

import subprocess
import shutil
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class DependencyStatus:
    """Status of a single dependency check."""
    name: str
    available: bool
    version: Optional[str] = None
    error_message: Optional[str] = None


class DependencyChecker:
    """
    Checks for required external dependencies.
    
    This class validates that ffmpeg and whisper are properly installed
    and accessible from the system PATH before starting transcription.
    """
    
    def __init__(self):
        self._cache: Dict[str, DependencyStatus] = {}
    
    def check_ffmpeg(self) -> DependencyStatus:
        """
        Check if ffmpeg is installed and accessible.
        
        Returns:
            DependencyStatus with availability and version info
        """
        if "ffmpeg" in self._cache:
            return self._cache["ffmpeg"]
        
        status = DependencyStatus(name="ffmpeg", available=False)
        
        # First, check if ffmpeg is in PATH
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is None:
            status.error_message = (
                "ffmpeg not found in system PATH.\n"
                "Please install ffmpeg and add it to your PATH:\n"
                "  - Windows: Download from https://ffmpeg.org/download.html\n"
                "             Add the 'bin' folder to your System PATH\n"
                "  - Linux: sudo apt install ffmpeg\n"
                "  - macOS: brew install ffmpeg"
            )
            self._cache["ffmpeg"] = status
            return status
        
        # Try to get version
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True,
                timeout=10,
                errors='replace',  # Handle encoding issues gracefully
            )
            if result.returncode == 0:
                # Extract version from first line
                first_line = result.stdout.split("\n")[0]
                status.available = True
                status.version = first_line
            else:
                status.error_message = f"ffmpeg returned error: {result.stderr}"
        except subprocess.TimeoutExpired:
            status.error_message = "ffmpeg check timed out"
        except FileNotFoundError:
            status.error_message = "ffmpeg executable not found"
        except Exception as e:
            status.error_message = f"Error checking ffmpeg: {str(e)}"
        
        self._cache["ffmpeg"] = status
        return status
    
    def check_whisper(self) -> DependencyStatus:
        """
        Check if whisper CLI is installed and accessible.
        
        Returns:
            DependencyStatus with availability info
        """
        if "whisper" in self._cache:
            return self._cache["whisper"]
        
        status = DependencyStatus(name="whisper", available=False)
        
        # Check if whisper is in PATH
        whisper_path = shutil.which("whisper")
        if whisper_path is None:
            status.error_message = (
                "whisper CLI not found in system PATH.\n"
                "Please install openai-whisper:\n"
                "  pip install openai-whisper\n"
                "\n"
                "Make sure your Python Scripts folder is in PATH."
            )
            self._cache["whisper"] = status
            return status
        
        # Try to run whisper help
        try:
            result = subprocess.run(
                ["whisper", "-h"],
                capture_output=True,
                text=True,
                timeout=30,  # Whisper can take a moment to load
                errors='replace',  # Handle encoding issues gracefully
            )
            if result.returncode == 0:
                status.available = True
                status.version = "Installed"
            else:
                status.error_message = f"whisper returned error: {result.stderr}"
        except subprocess.TimeoutExpired:
            status.error_message = "whisper check timed out"
        except FileNotFoundError:
            status.error_message = "whisper executable not found"
        except Exception as e:
            status.error_message = f"Error checking whisper: {str(e)}"
        
        self._cache["whisper"] = status
        return status
    
    def check_ffprobe(self) -> DependencyStatus:
        """
        Check if ffprobe is installed (usually comes with ffmpeg).
        
        Returns:
            DependencyStatus with availability info
        """
        if "ffprobe" in self._cache:
            return self._cache["ffprobe"]
        
        status = DependencyStatus(name="ffprobe", available=False)
        
        ffprobe_path = shutil.which("ffprobe")
        if ffprobe_path is None:
            status.error_message = (
                "ffprobe not found. It should be installed with ffmpeg.\n"
                "Please ensure ffmpeg is properly installed."
            )
            self._cache["ffprobe"] = status
            return status
        
        try:
            result = subprocess.run(
                ["ffprobe", "-version"],
                capture_output=True,
                text=True,
                timeout=10,
                errors='replace',  # Handle encoding issues gracefully
            )
            if result.returncode == 0:
                status.available = True
                status.version = result.stdout.split("\n")[0]
            else:
                status.error_message = f"ffprobe returned error: {result.stderr}"
        except Exception as e:
            status.error_message = f"Error checking ffprobe: {str(e)}"
        
        self._cache["ffprobe"] = status
        return status
    
    def validate_all(self) -> Tuple[bool, Dict[str, DependencyStatus]]:
        """
        Validate all required dependencies.
        
        Returns:
            Tuple of (all_valid, status_dict)
        """
        statuses = {
            "ffmpeg": self.check_ffmpeg(),
            "ffprobe": self.check_ffprobe(),
            "whisper": self.check_whisper(),
        }
        
        all_valid = all(s.available for s in statuses.values())
        return all_valid, statuses
    
    def get_missing_dependencies(self) -> list:
        """
        Get list of missing dependencies.
        
        Returns:
            List of DependencyStatus for missing dependencies
        """
        _, statuses = self.validate_all()
        return [s for s in statuses.values() if not s.available]

@dataclass
class HardwareInfo:
    """Hardware capabilities of the system."""
    cpu_cores: int
    has_nvidia_gpu: bool
    description: str
    gpu_reason: str = ""  # Detailed reason for GPU availability status


def get_hardware_info() -> HardwareInfo:
    """
    Detect system hardware capabilities (CPU/GPU).
    
    Returns:
        HardwareInfo object with detected capabilities and diagnostics
    """
    import os
    import importlib.util

    # Check CPU
    cpu_cores = os.cpu_count() or 4
    
    # Check GPU with detailed diagnostics
    has_gpu = False
    gpu_reason = ""
    gpu_name = None
    
    # Step 1: Check if PyTorch is installed
    torch_spec = importlib.util.find_spec("torch")
    if torch_spec is None:
        gpu_reason = "PyTorch not installed (pip install torch)"
    else:
        try:
            import torch
            
            # Step 2: Check if CUDA is available
            if not torch.cuda.is_available():
                # Provide detailed reason why CUDA is unavailable
                cuda_build = getattr(torch.version, 'cuda', None)
                if cuda_build is None:
                    gpu_reason = "PyTorch installed without CUDA support (reinstall with CUDA)"
                else:
                    # CUDA was compiled in but runtime failed
                    gpu_reason = f"CUDA runtime unavailable (PyTorch built for CUDA {cuda_build})"
                    
                    # Additional check: try to detect NVIDIA driver
                    try:
                        import subprocess
                        result = subprocess.run(
                            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                            capture_output=True, text=True, timeout=5
                        )
                        if result.returncode != 0:
                            gpu_reason = "No NVIDIA driver detected (install or update GPU drivers)"
                    except (FileNotFoundError, subprocess.TimeoutExpired):
                        gpu_reason = "nvidia-smi not found (no NVIDIA driver installed)"
                    except Exception:
                        pass  # Keep original cuda reason
            else:
                # GPU is available!
                has_gpu = True
                try:
                    gpu_name = torch.cuda.get_device_name(0)
                    vram_bytes = torch.cuda.get_device_properties(0).total_memory
                    vram_gb = vram_bytes / (1024 ** 3)
                    gpu_reason = f"{gpu_name} ({vram_gb:.1f} GB VRAM)"
                except Exception:
                    gpu_reason = "NVIDIA GPU detected"
                    
        except ImportError as e:
            gpu_reason = f"PyTorch import failed: {e}"
        except Exception as e:
            gpu_reason = f"GPU detection error: {e}"
    
    # Build description string
    desc = [f"CPU: {cpu_cores} cores"]
    if has_gpu and gpu_name:
        desc.append(f"GPU: {gpu_name}")
    elif has_gpu:
        desc.append("GPU: NVIDIA")
    else:
        desc.append("GPU: None (CPU mode)")
        
    return HardwareInfo(
        cpu_cores=cpu_cores,
        has_nvidia_gpu=has_gpu,
        description=", ".join(desc),
        gpu_reason=gpu_reason
    )


def check_environment() -> Tuple[bool, str]:
    """
    Convenience function to check all dependencies.
    
    Returns:
        Tuple of (success, message)
    """
    checker = DependencyChecker()
    all_valid, statuses = checker.validate_all()
    
    # Also get hardware info
    hw = get_hardware_info()
    
    if all_valid:
        versions = []
        for name, status in statuses.items():
            if status.version:
                versions.append(f"  ✓ {name}: {status.version}")
        
        msg = "All dependencies found:\n" + "\n".join(versions)
        msg += f"\n\nSystem Hardware:\n  ℹ {hw.description}"
        if hw.gpu_reason:
            msg += f"\n  ℹ GPU Status: {hw.gpu_reason}"
        return True, msg
    else:
        errors = []
        for name, status in statuses.items():
            if not status.available:
                errors.append(f"\n[{name}]\n{status.error_message}")
        return False, "Missing dependencies:" + "".join(errors)


if __name__ == "__main__":
    # Test dependency checking
    success, message = check_environment()
    print(message)
    exit(0 if success else 1)
