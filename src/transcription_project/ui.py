# ==============================================================================
# ui.py - Rich Console Interface Module
# ==============================================================================
"""
Rich console interface with progress tracking and color-coded logging.
"""

from typing import Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TaskProgressColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.text import Text
from rich import box

from .config import APP_NAME, APP_VERSION


class TranscriptionUI:
    """Rich console interface for transcription progress."""
    
    def __init__(self, verbose: bool = False):
        self.console = Console()
        self.verbose = verbose
        self._progress: Optional[Progress] = None
        self._overall_task = None
        self._total_chunks = 0
        self._num_workers = 1
    
    def show_banner(self) -> None:
        """Display application banner."""
        banner = Text()
        banner.append(f"🎬 {APP_NAME} ", style="bold cyan")
        banner.append(f"v{APP_VERSION}", style="dim")
        self.console.print(Panel(banner, box=box.ROUNDED, border_style="cyan"))
    
    def log_info(self, message: str) -> None:
        """Log info message."""
        self.console.print(f"[blue]ℹ[/blue] {message}")
    
    def log_success(self, message: str) -> None:
        """Log success message."""
        self.console.print(f"[green]✓[/green] {message}")
    
    def log_warning(self, message: str) -> None:
        """Log warning message."""
        self.console.print(f"[yellow]⚠[/yellow] {message}")
    
    def log_error(self, message: str) -> None:
        """Log error message."""
        self.console.print(f"[red]✗[/red] {message}")
    
    def log_debug(self, message: str) -> None:
        """Log debug message (only in verbose mode)."""
        if self.verbose:
            self.console.print(f"[dim]  {message}[/dim]")
    
    def create_progress(self) -> Progress:
        """Create progress display."""
        self._progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=self.console,
        )
        return self._progress
    
    def start_progress(self, total_chunks: int, num_workers: int = 1) -> None:
        """Start progress tracking.
        
        Args:
            total_chunks: Total number of chunks to process
            num_workers: Number of concurrent workers (for display)
        """
        self._total_chunks = total_chunks
        self._num_workers = num_workers
        if self._progress:
            # v3.1: Show workers in progress description
            desc = f"[cyan]Overall Progress ({num_workers} worker{'s' if num_workers > 1 else ''})"
            self._overall_task = self._progress.add_task(desc, total=total_chunks)
    
    def update_progress(self, completed: int) -> None:
        """Update progress display.
        
        Args:
            completed: Number of completed chunks
        """
        if self._progress and self._overall_task is not None:
            self._progress.update(self._overall_task, completed=completed)
    
    def finish_progress(self) -> None:
        """Complete progress tracking."""
        # v3.1: Simplified - just mark the progress as done
        pass
    
    def show_summary(self, video_name: str, total_chunks: int, duration_mins: float, output_path: str) -> None:
        """Show transcription summary."""
        summary = Text()
        summary.append("\n📊 Transcription Complete!\n\n", style="bold green")
        summary.append(f"  Video: {video_name}\n")
        summary.append(f"  Chunks: {total_chunks}\n")
        summary.append(f"  Duration: {duration_mins:.1f} minutes\n")
        summary.append(f"  Output: {output_path}\n")
        self.console.print(Panel(summary, box=box.ROUNDED, border_style="green"))
    
    def show_error_panel(self, title: str, message: str) -> None:
        """Show error panel."""
        self.console.print(Panel(message, title=title, border_style="red"))
    
    def show_resume_info(self, completed: int, total: int) -> None:
        """Show resume information."""
        self.log_info(f"Resuming: {completed}/{total} chunks already completed ({completed/total*100:.1f}%)")

    def show_status(self, message: str):
        """Show a status spinner with a message."""
        return self.console.status(f"[bold cyan]{message}", spinner="dots")

