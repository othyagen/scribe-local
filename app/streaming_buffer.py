"""Streaming transcript buffer — ordered, append-only segment store.

Maintains an ordered list of transcript segments produced by the ASR
pipeline.  Supports incremental extraction by tracking which segments
have already been processed.

Thread-safe: all mutations are protected by a lock so the buffer can
be written from the recording thread and read from a display or
extraction thread.

No ML, no LLM — pure data structure.
"""

from __future__ import annotations

import bisect
import threading
from typing import Optional


class StreamingBuffer:
    """Append-only, time-ordered transcript segment buffer.

    Segment schema (minimum required keys)::

        {
            "seg_id": str,
            "t0": float,
            "t1": float,
            "speaker_id": str,
            "normalized_text": str,
        }

    Extra keys (``raw_text``, ``paragraph_id``, …) are preserved.
    """

    def __init__(self) -> None:
        self._segments: list[dict] = []
        self._seg_ids: set[str] = set()
        self._lock = threading.Lock()
        self._processed_count: int = 0

    # ── mutation ────────────────────────────────────────────────────

    def append_segment(self, segment: dict) -> bool:
        """Append a segment, maintaining time order.

        Returns ``True`` if the segment was added, ``False`` if it was
        a duplicate (same ``seg_id``).

        Raises ``ValueError`` if required keys are missing.
        """
        _validate_segment(segment)

        with self._lock:
            seg_id = segment["seg_id"]
            if seg_id in self._seg_ids:
                return False

            self._seg_ids.add(seg_id)

            # Fast path: most segments arrive in order
            if not self._segments or segment["t0"] >= self._segments[-1]["t0"]:
                self._segments.append(segment)
            else:
                # Out-of-order: insert at correct position by t0
                keys = [s["t0"] for s in self._segments]
                idx = bisect.bisect_right(keys, segment["t0"])
                self._segments.insert(idx, segment)

            return True

    def clear(self) -> None:
        """Remove all segments and reset processed counter."""
        with self._lock:
            self._segments.clear()
            self._seg_ids.clear()
            self._processed_count = 0

    # ── queries ─────────────────────────────────────────────────────

    def get_recent_segments(self, n: int) -> list[dict]:
        """Return the *n* most recent segments (by time order).

        Returns fewer than *n* if the buffer contains fewer segments.
        """
        with self._lock:
            if n <= 0:
                return []
            return list(self._segments[-n:])

    def get_full_transcript(self) -> list[dict]:
        """Return a copy of all segments in time order."""
        with self._lock:
            return list(self._segments)

    def get_segment_count(self) -> int:
        """Return the total number of segments in the buffer."""
        with self._lock:
            return len(self._segments)

    def get_full_text(self) -> str:
        """Concatenate all normalized_text values into a single string."""
        with self._lock:
            return " ".join(
                s.get("normalized_text", "") for s in self._segments
            ).strip()

    def get_elapsed_seconds(self) -> float:
        """Return the time span from first t0 to last t1, or 0.0."""
        with self._lock:
            if not self._segments:
                return 0.0
            return self._segments[-1]["t1"] - self._segments[0]["t0"]

    def get_word_count(self) -> int:
        """Return total word count across all segments."""
        with self._lock:
            total = 0
            for s in self._segments:
                text = s.get("normalized_text", "")
                if text:
                    total += len(text.split())
            return total

    # ── incremental processing ──────────────────────────────────────

    def get_unprocessed_segments(self) -> list[dict]:
        """Return segments that have not yet been marked as processed.

        Does **not** advance the processed counter — call
        :meth:`mark_processed` after extraction succeeds.
        """
        with self._lock:
            return list(self._segments[self._processed_count:])

    def mark_processed(self, count: Optional[int] = None) -> None:
        """Advance the processed counter.

        Args:
            count: number of segments to mark.  If ``None``, marks all
                   current segments as processed.
        """
        with self._lock:
            if count is None:
                self._processed_count = len(self._segments)
            else:
                self._processed_count = min(
                    self._processed_count + count,
                    len(self._segments),
                )

    @property
    def processed_count(self) -> int:
        with self._lock:
            return self._processed_count


# ── helpers ─────────────────────────────────────────────────────────

_REQUIRED_KEYS = {"seg_id", "t0", "t1", "speaker_id", "normalized_text"}


def _validate_segment(segment: dict) -> None:
    """Raise ValueError if required keys are missing."""
    missing = _REQUIRED_KEYS - set(segment.keys())
    if missing:
        raise ValueError(f"Segment missing required keys: {missing}")
