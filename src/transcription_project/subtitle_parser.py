# ==============================================================================
# subtitle_parser.py - SRT Subtitle Manipulation Module
# ==============================================================================
"""
Handles SRT file parsing, manipulation, and merging.
"""

import re
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class SubtitleEntry:
    """A single subtitle entry."""
    index: int
    start_time: float
    end_time: float
    text: str
    
    def __str__(self) -> str:
        return f"{self.index}\n{format_timestamp(self.start_time)} --> {format_timestamp(self.end_time)}\n{self.text}\n"


def format_timestamp(seconds: float) -> str:
    """Convert seconds to SRT timestamp (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def parse_timestamp(timestamp: str) -> float:
    """Parse SRT timestamp to seconds."""
    timestamp = timestamp.replace(",", ".")
    match = re.match(r"(\d{1,2}):(\d{2}):(\d{2})\.(\d{3})", timestamp)
    if match:
        h, m, s, ms = map(int, match.groups())
        return h * 3600 + m * 60 + s + ms / 1000
    raise ValueError(f"Invalid timestamp: {timestamp}")


class SubtitleParser:
    """Handles SRT file parsing and manipulation."""
    
    def parse_srt(self, srt_path: Path) -> List[SubtitleEntry]:
        """Parse an SRT file into subtitle entries."""
        with open(srt_path, "r", encoding="utf-8") as f:
            content = f.read()
        return self.parse_srt_content(content)
    
    def parse_srt_content(self, content: str) -> List[SubtitleEntry]:
        """Parse SRT content string into entries."""
        entries = []
        content = content.replace("\r\n", "\n").strip()
        blocks = re.split(r"\n\n+", content)
        
        for block in blocks:
            if not block.strip():
                continue
            lines = block.strip().split("\n")
            if len(lines) < 3:
                continue
            try:
                index = int(lines[0].strip())
                time_match = re.match(
                    r"(\d{1,2}:\d{2}:\d{2}[,\.]\d{3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[,\.]\d{3})",
                    lines[1].strip()
                )
                if not time_match:
                    continue
                start = parse_timestamp(time_match.group(1))
                end = parse_timestamp(time_match.group(2))
                text = "\n".join(lines[2:]).strip()
                entries.append(SubtitleEntry(index, start, end, text))
            except (ValueError, IndexError):
                continue
        return entries
    
    def write_srt(self, entries: List[SubtitleEntry], output_path: Path) -> None:
        """Write subtitle entries to an SRT file."""
        reindexed = self.reindex_entries(entries)
        with open(output_path, "w", encoding="utf-8") as f:
            for entry in reindexed:
                f.write(str(entry) + "\n")
    
    def adjust_timestamps(self, entries: List[SubtitleEntry], offset: float) -> List[SubtitleEntry]:
        """Adjust all timestamps by offset."""
        return [SubtitleEntry(e.index, e.start_time + offset, e.end_time + offset, e.text) for e in entries]
    
    def reindex_entries(self, entries: List[SubtitleEntry]) -> List[SubtitleEntry]:
        """Re-index entries starting from 1."""
        return [SubtitleEntry(i, e.start_time, e.end_time, e.text) for i, e in enumerate(entries, 1)]
    
    def merge_srts(self, srt_files: List[Path], chunk_duration: float) -> List[SubtitleEntry]:
        """Merge multiple SRT files with timestamp adjustment."""
        all_entries = []
        for i, srt_path in enumerate(srt_files):
            if not srt_path.exists():
                continue
            entries = self.parse_srt(srt_path)
            offset = i * chunk_duration
            adjusted = self.adjust_timestamps(entries, offset)
            all_entries.extend(adjusted)
        all_entries.sort(key=lambda e: e.start_time)
        return self.reindex_entries(all_entries)
    
    def merge_srt_files(self, srt_files: List[Path], chunk_duration: float, output_path: Path) -> None:
        """Merge SRT files and write to output."""
        merged = self.merge_srts(srt_files, chunk_duration)
        self.write_srt(merged, output_path)
    
    def merge_with_existing(
        self, 
        existing_path: Path, 
        new_entries: List[SubtitleEntry],
        tolerance: float = 0.1
    ) -> List[SubtitleEntry]:
        """
        Merge new entries with an existing SRT file.
        
        v4.0: Used for unified partial subtitle - combines old and new entries,
        removes duplicates (based on start time proximity), and sorts by time.
        
        Args:
            existing_path: Path to existing SRT file (may not exist)
            new_entries: New subtitle entries to merge
            tolerance: Time tolerance for deduplication (seconds)
            
        Returns:
            Combined, sorted, deduplicated list of entries
        """
        all_entries = []
        
        # Read existing entries if file exists
        if existing_path.exists():
            try:
                existing_entries = self.parse_srt(existing_path)
                all_entries.extend(existing_entries)
            except Exception:
                pass  # If parse fails, start fresh
        
        # Add new entries
        all_entries.extend(new_entries)
        
        # Sort by start time
        all_entries.sort(key=lambda e: e.start_time)
        
        # Deduplicate: Remove entries with nearly identical start times
        # Keep the later one (assumed to be more recent/accurate)
        if len(all_entries) > 1:
            deduplicated = []
            seen_times = set()
            
            for entry in all_entries:
                # Round to tolerance for comparison
                time_key = round(entry.start_time / tolerance) * tolerance
                if time_key not in seen_times:
                    deduplicated.append(entry)
                    seen_times.add(time_key)
            
            all_entries = deduplicated
        
        # Re-index starting from 1
        return self.reindex_entries(all_entries)
    
    def update_partial_file(
        self,
        partial_path: Path,
        srt_files: List[Path],
        chunk_duration: float,
        completed_indices: List[int]
    ) -> None:
        """
        Update the unified partial subtitle file with new chunks.
        
        v4.0: Merges chunk SRT files with existing partial file content.
        
        Args:
            partial_path: Path to _IN_PROGRESS.srt file
            srt_files: List of chunk SRT file paths (indexed by chunk number)
            chunk_duration: Duration of each chunk in seconds
            completed_indices: List of chunk indices to include
        """
        # Collect entries from newly completed chunks
        new_entries = []
        for idx in completed_indices:
            if idx < len(srt_files) and srt_files[idx].exists():
                chunk_entries = self.parse_srt(srt_files[idx])
                offset = idx * chunk_duration
                adjusted = self.adjust_timestamps(chunk_entries, offset)
                new_entries.extend(adjusted)
        
        # Merge with existing partial file
        merged = self.merge_with_existing(partial_path, new_entries)
        
        # Write back
        self.write_srt(merged, partial_path)
