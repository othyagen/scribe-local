"""Live transcript preview — lightweight console output.

Formats and prints transcript segments as they arrive, providing
real-time feedback during recording sessions.

No ML, no LLM — pure formatting and display.
"""

from __future__ import annotations

import sys
from typing import Optional, TextIO

from app.streaming_buffer import StreamingBuffer


def format_timestamp(seconds: float) -> str:
    """Format seconds as MM:SS."""
    if seconds < 0:
        seconds = 0.0
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def format_preview_line(segment: dict) -> str:
    """Format a single segment for console preview.

    Output::

        [00:12] spk_1: I've had a headache for three days
    """
    ts = format_timestamp(segment.get("t0", 0.0))
    speaker = segment.get("speaker_id", "???")
    text = segment.get("normalized_text", "")
    return f"[{ts}] {speaker}: {text}"


def format_status_line(buffer: StreamingBuffer) -> str:
    """Format a status summary line.

    Output::

        --- 42 words | 01:23 elapsed | 5 segments ---
    """
    words = buffer.get_word_count()
    elapsed = format_timestamp(buffer.get_elapsed_seconds())
    segs = buffer.get_segment_count()
    return f"--- {words} words | {elapsed} elapsed | {segs} segments ---"


class LivePreview:
    """Console preview that prints segments as they arrive.

    Args:
        buffer: the streaming buffer to read from.
        show_status: whether to print periodic status lines.
        output: writable stream (defaults to ``sys.stderr``).
    """

    def __init__(
        self,
        buffer: StreamingBuffer,
        show_status: bool = False,
        output: Optional[TextIO] = None,
    ) -> None:
        self._buffer = buffer
        self._show_status = show_status
        self._output = output or sys.stderr
        self._last_printed: int = 0

    def print_segment(self, segment: dict) -> str:
        """Format and print a single segment.  Returns the formatted line."""
        line = format_preview_line(segment)
        self._output.write(line + "\n")
        self._output.flush()
        self._last_printed += 1
        return line

    def print_new_segments(self) -> list[str]:
        """Print any segments added since the last call.

        Returns the list of formatted lines printed.
        """
        all_segs = self._buffer.get_full_transcript()
        new_segs = all_segs[self._last_printed:]
        lines: list[str] = []
        for seg in new_segs:
            lines.append(self.print_segment(seg))
        return lines

    def print_status(self) -> str:
        """Print a status summary line.  Returns the formatted string."""
        line = format_status_line(self._buffer)
        self._output.write(line + "\n")
        self._output.flush()
        return line

    @property
    def segments_printed(self) -> int:
        return self._last_printed
