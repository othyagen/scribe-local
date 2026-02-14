"""Segment commitment — creates immutable RAW segments.

A committed segment is the atomic unit of the RAW transcript.
Once created, a RawSegment must NEVER be modified.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass(frozen=True)
class RawSegment:
    """Immutable raw transcript segment."""
    seg_id: str
    t0: float
    t1: float
    speaker_id: str
    raw_text: str
    model_name: str
    language: str
    paragraph_id: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_txt_line(self) -> str:
        ts0 = _fmt_ts(self.t0)
        ts1 = _fmt_ts(self.t1)
        return f"[{ts0} - {ts1}] [{self.speaker_id}] {self.raw_text}"


class SegmentCommitter:
    """Creates RawSegments with sequential IDs and paragraph tracking."""

    def __init__(self, model_name: str, language: str) -> None:
        self.model_name: str = model_name
        self.language: str = language
        self._seg_counter: int = 0
        self._para_counter: int = 0

    def commit(
        self,
        t0: float,
        t1: float,
        speaker_id: str,
        text: str,
    ) -> RawSegment:
        """Create and return a new immutable RawSegment."""
        self._seg_counter += 1
        return RawSegment(
            seg_id=f"seg_{self._seg_counter:04d}",
            t0=round(t0, 3),
            t1=round(t1, 3),
            speaker_id=speaker_id,
            raw_text=text,
            model_name=self.model_name,
            language=self.language,
            paragraph_id=f"para_{self._para_counter:04d}",
        )

    def new_paragraph(self) -> None:
        """Advance the paragraph counter (called on long silence)."""
        self._para_counter += 1

    @property
    def seg_count(self) -> int:
        return self._seg_counter


def _fmt_ts(seconds: float) -> str:
    """Format seconds as HH:MM:SS.mmm."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"
