"""Live findings preview — incremental extractor output during recording.

Reads unprocessed segments from a :class:`StreamingBuffer`, runs existing
deterministic extractors and the qualifier engine on those segments only,
collects newly detected findings, and prints a concise console preview.

No ML, no LLM — calls the same extractors used in batch mode.
Does **not** modify extractor or qualifier logic.
"""

from __future__ import annotations

import sys
from typing import Optional, TextIO

from app.extractors import (
    extract_symptoms,
    extract_negations,
    extract_durations,
    extract_medications,
)
from app.qualifier_extraction import extract_qualifiers
from app.streaming_buffer import StreamingBuffer


# ── finding categories ─────────────────────────────────────────────

_EXTRACTOR_MAP: dict[str, callable] = {
    "symptom": extract_symptoms,
    "negation": extract_negations,
    "duration": extract_durations,
    "medication": extract_medications,
}


# ── formatting ─────────────────────────────────────────────────────

def format_finding_line(category: str, value: str) -> str:
    """Format a single finding for console output.

    Example::

        + symptom: headache
    """
    return f"  + {category}: {value}"


def format_qualifier_line(qualifier_type: str, value: str) -> str:
    """Format a qualifier for console output.

    Example::

        + severity: severe
    """
    return f"  + {qualifier_type}: {value}"


# ── incremental extraction ─────────────────────────────────────────

def extract_findings_from_text(text: str) -> list[tuple[str, str]]:
    """Run all extractors on *text* and return ``(category, value)`` pairs.

    The same extractor functions used in batch mode are called here.
    Results are returned in a stable order: symptoms, negations,
    durations, medications.
    """
    findings: list[tuple[str, str]] = []
    for category, extractor in _EXTRACTOR_MAP.items():
        for value in extractor(text):
            findings.append((category, value))
    return findings


def extract_qualifiers_from_segment(
    segment: dict,
    known_symptoms: set[str],
) -> list[tuple[str, str]]:
    """Run qualifier extraction on a single segment.

    Returns ``(qualifier_type, value)`` pairs — e.g.
    ``("severity", "severe")``, ``("aggravating_factor", "movement")``.
    """
    results: list[tuple[str, str]] = []
    if not known_symptoms:
        return results

    entries = extract_qualifiers(
        [segment], extracted_findings=list(known_symptoms),
    )
    for entry in entries:
        quals = entry.get("qualifiers", {})
        for qtype, qval in quals.items():
            if isinstance(qval, list):
                # aggravating_factors / relieving_factors
                # Use singular form for display
                singular = qtype.rstrip("s") if qtype.endswith("_factors") else qtype
                for v in qval:
                    results.append((singular, v))
            else:
                results.append((qtype, qval))
    return results


class LiveFindings:
    """Incremental findings preview backed by a :class:`StreamingBuffer`.

    Reads unprocessed segments, runs extractors and the qualifier engine,
    prints new findings, and marks segments as processed.  Tracks seen
    findings and qualifiers to suppress duplicates across segments.

    Args:
        buffer: the streaming buffer supplying segments.
        output: writable stream for findings output (defaults to
            ``sys.stderr``).
    """

    def __init__(
        self,
        buffer: StreamingBuffer,
        output: Optional[TextIO] = None,
    ) -> None:
        self._buffer = buffer
        self._output = output or sys.stderr
        self._seen: set[tuple[str, str]] = set()
        self._all_findings: list[tuple[str, str]] = []
        self._seen_qualifiers: set[tuple[str, str]] = set()
        self._all_qualifiers: list[tuple[str, str]] = []

    # ── core loop ──────────────────────────────────────────────────

    def process_new_segments(self) -> list[tuple[str, str]]:
        """Process unprocessed segments and return *new* findings.

        Steps:
          1. Read unprocessed segments from the buffer.
          2. Run extractors on each segment's text.
          3. Run qualifier engine on each segment.
          4. Filter out already-seen findings and qualifiers.
          5. Print new findings and qualifiers to the output stream.
          6. Mark segments as processed.

        Returns:
            list of ``(category, value)`` pairs for newly detected
            findings **and** qualifiers combined.
        """
        unprocessed = self._buffer.get_unprocessed_segments()
        if not unprocessed:
            return []

        new_items: list[tuple[str, str]] = []

        for seg in unprocessed:
            text = seg.get("normalized_text", "")
            if not text:
                continue

            # — extractor findings —
            seg_findings = extract_findings_from_text(text)
            new_symptoms: set[str] = set()

            for finding in seg_findings:
                key = (finding[0], finding[1].lower())
                if key not in self._seen:
                    self._seen.add(key)
                    self._all_findings.append(finding)
                    new_items.append(finding)
                    self._print_finding(finding)
                if finding[0] == "symptom":
                    new_symptoms.add(finding[1])

            # — qualifier extraction —
            # Use all known symptoms (accumulated + this segment's)
            known = {v for c, v in self._all_findings if c == "symptom"}
            known.update(new_symptoms)

            seg_quals = extract_qualifiers_from_segment(seg, known)
            for qual in seg_quals:
                qkey = (qual[0], qual[1].lower())
                if qkey not in self._seen_qualifiers:
                    self._seen_qualifiers.add(qkey)
                    self._all_qualifiers.append(qual)
                    new_items.append(qual)
                    self._print_qualifier(qual)

        self._buffer.mark_processed(len(unprocessed))
        return new_items

    # ── queries ────────────────────────────────────────────────────

    @property
    def all_findings(self) -> list[tuple[str, str]]:
        """All unique extractor findings detected so far, in discovery order."""
        return list(self._all_findings)

    @property
    def all_qualifiers(self) -> list[tuple[str, str]]:
        """All unique qualifiers detected so far, in discovery order."""
        return list(self._all_qualifiers)

    def get_findings_by_category(self, category: str) -> list[str]:
        """Return all values for a given finding category."""
        return [v for c, v in self._all_findings if c == category]

    def get_qualifiers_by_type(self, qualifier_type: str) -> list[str]:
        """Return all values for a given qualifier type."""
        return [v for t, v in self._all_qualifiers if t == qualifier_type]

    @property
    def finding_count(self) -> int:
        return len(self._all_findings)

    @property
    def qualifier_count(self) -> int:
        return len(self._all_qualifiers)

    # ── output ─────────────────────────────────────────────────────

    def _print_finding(self, finding: tuple[str, str]) -> None:
        line = format_finding_line(finding[0], finding[1])
        self._output.write(line + "\n")
        self._output.flush()

    def _print_qualifier(self, qualifier: tuple[str, str]) -> None:
        line = format_qualifier_line(qualifier[0], qualifier[1])
        self._output.write(line + "\n")
        self._output.flush()
