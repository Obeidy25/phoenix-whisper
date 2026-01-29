# ==============================================================================
# diagnostics.py - Performance Diagnostic Module
# ==============================================================================
"""
Diagnostic tests to measure and verify performance of key operations.
Run with: python run.py --diagnose
"""

import time
import tempfile
from pathlib import Path
from typing import Tuple, List

from .config import TEMP_CHUNK_PREFIX, APP_NAME, APP_VERSION


def format_time(seconds: float) -> str:
    """Format time in human-readable format."""
    if seconds < 0.001:
        return f"{seconds * 1000000:.0f} µs"
    elif seconds < 1:
        return f"{seconds * 1000:.1f} ms"
    else:
        return f"{seconds:.2f} s"


def test_chunk_existence_check(num_chunks: int = 106) -> Tuple[float, bool]:
    """
    Test performance of checking file existence for many chunks.
    
    Simulates the fast-path check in the producer task.
    Creates temporary dummy files and checks their existence.
    
    Returns:
        Tuple of (elapsed_time, passed)
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir)
        
        # Create dummy chunk files (just empty files)
        for i in range(num_chunks):
            chunk_file = temp_path / f"{TEMP_CHUNK_PREFIX}{i:04d}.wav"
            chunk_file.touch()
        
        # Now measure the time to check existence of all files
        start = time.perf_counter()
        
        for i in range(num_chunks):
            chunk_file = temp_path / f"{TEMP_CHUNK_PREFIX}{i:04d}.wav"
            exists = chunk_file.exists()
            # Simulate what producer does: if exists, would queue it
            if exists:
                _ = (i, chunk_file)  # Simulate tuple creation
        
        elapsed = time.perf_counter() - start
        
    # Pass if less than 1 second
    passed = elapsed < 1.0
    return elapsed, passed


def test_boundary_calculation(total_seconds: float = 6360.0, chunk_duration: int = 60) -> Tuple[float, bool]:
    """
    Test performance of calculating chunk boundaries.
    
    Args:
        total_seconds: Total video duration (default ~106 minutes)
        chunk_duration: Duration per chunk in seconds
        
    Returns:
        Tuple of (elapsed_time, passed)
    """
    start = time.perf_counter()
    
    boundaries = []
    current_pos = 0.0
    while current_pos < total_seconds:
        end_pos = min(current_pos + chunk_duration, total_seconds)
        boundaries.append((current_pos, end_pos))
        current_pos = end_pos
    
    elapsed = time.perf_counter() - start
    
    # Pass if less than 0.1 seconds
    passed = elapsed < 0.1
    return elapsed, passed, len(boundaries)


def test_set_operations(num_items: int = 106) -> Tuple[float, bool]:
    """
    Test performance of set operations used in resume logic.
    
    Returns:
        Tuple of (elapsed_time, passed)
    """
    # Simulate completed_indices set creation and lookups
    start = time.perf_counter()
    
    completed = list(range(0, num_items, 2))  # 53 completed
    completed_set = set(completed)
    
    # Simulate the producer loop checks
    for i in range(num_items):
        _ = i in completed_set
    
    elapsed = time.perf_counter() - start
    
    # Pass if less than 0.01 seconds
    passed = elapsed < 0.01
    return elapsed, passed


def test_path_operations(num_paths: int = 106) -> Tuple[float, bool]:
    """
    Test performance of Path object creation.
    
    Returns:
        Tuple of (elapsed_time, passed)
    """
    start = time.perf_counter()
    
    base = Path("C:/test/temp")
    for i in range(num_paths):
        _ = base / f"{TEMP_CHUNK_PREFIX}{i:04d}.wav"
    
    elapsed = time.perf_counter() - start
    
    # Pass if less than 0.01 seconds
    passed = elapsed < 0.01
    return elapsed, passed


def run_diagnostics(console) -> int:
    """
    Run all diagnostic tests and display results.
    
    Args:
        console: Rich console instance for output
        
    Returns:
        Exit code (0 = all passed, 1 = some failed)
    """
    from rich.panel import Panel
    from rich.text import Text
    from rich import box
    
    # Header
    header = Text()
    header.append(f"🔬 Running Diagnostics - {APP_NAME} v{APP_VERSION}", style="bold cyan")
    console.print(Panel(header, box=box.ROUNDED, border_style="cyan"))
    console.print()
    
    all_passed = True
    results = []
    
    # Test 1: Chunk existence check
    console.print("[bold yellow][TEST 1][/] Simulating chunk file existence check (106 files)...")
    elapsed, passed = test_chunk_existence_check(106)
    status = "[green]PASS[/]" if passed else "[red]FAIL[/]"
    console.print(f"         {'✓' if passed else '✗'} Time: {format_time(elapsed)} ({status}: expected < 1s)")
    results.append(("Chunk existence check", elapsed, passed))
    if not passed:
        all_passed = False
    console.print()
    
    # Test 2: Boundary calculation
    console.print("[bold yellow][TEST 2][/] Calculating chunk boundaries (106 chunks)...")
    elapsed, passed, num_chunks = test_boundary_calculation()
    status = "[green]PASS[/]" if passed else "[red]FAIL[/]"
    console.print(f"         {'✓' if passed else '✗'} Generated {num_chunks} chunks in {format_time(elapsed)} ({status}: expected < 0.1s)")
    results.append(("Boundary calculation", elapsed, passed))
    if not passed:
        all_passed = False
    console.print()
    
    # Test 3: Set operations
    console.print("[bold yellow][TEST 3][/] Testing set operations for resume logic...")
    elapsed, passed = test_set_operations(106)
    status = "[green]PASS[/]" if passed else "[red]FAIL[/]"
    console.print(f"         {'✓' if passed else '✗'} Time: {format_time(elapsed)} ({status}: expected < 0.01s)")
    results.append(("Set operations", elapsed, passed))
    if not passed:
        all_passed = False
    console.print()
    
    # Test 4: Path operations
    console.print("[bold yellow][TEST 4][/] Testing Path object creation (106 paths)...")
    elapsed, passed = test_path_operations(106)
    status = "[green]PASS[/]" if passed else "[red]FAIL[/]"
    console.print(f"         {'✓' if passed else '✗'} Time: {format_time(elapsed)} ({status}: expected < 0.01s)")
    results.append(("Path operations", elapsed, passed))
    if not passed:
        all_passed = False
    console.print()
    
    # Summary
    console.print("─" * 60)
    total_time = sum(r[1] for r in results)
    
    if all_passed:
        summary = Text()
        summary.append("\n📊 All Diagnostics Passed!\n\n", style="bold green")
        summary.append(f"  Total time: {format_time(total_time)}\n\n")
        summary.append("  If resume is still slow, the bottleneck is likely:\n", style="dim")
        summary.append("  • Heavy import loading (Whisper/Torch) - unavoidable\n", style="dim")
        summary.append("  • ffprobe video analysis - one-time per session\n", style="dim")
        summary.append("  • Actual transcription work - expected\n", style="dim")
        console.print(Panel(summary, box=box.ROUNDED, border_style="green"))
        return 0
    else:
        summary = Text()
        summary.append("\n⚠️ Some Diagnostics Failed\n\n", style="bold red")
        summary.append("  Please review the failed tests above.\n")
        console.print(Panel(summary, box=box.ROUNDED, border_style="red"))
        return 1


if __name__ == "__main__":
    from rich.console import Console
    console = Console()
    exit(run_diagnostics(console))
