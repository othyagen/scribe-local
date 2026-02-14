"""File I/O — writes RAW, NORMALIZED, and audit-change outputs.

RAW files are append-only (JSON-Lines + TXT) and must NEVER be
rewritten after a segment is committed.

RAW file handles are opened at session start and flushed on every
commit so that data survives even if the process is killed.

NORMALIZED and CHANGES files are written/updated at session end.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import IO, List

from app.commit import RawSegment
from app.normalize import NormalizedSegment, NormalizationChange


class OutputWriter:
    """Manages all output files for a single recording session."""

    def __init__(self, output_dir: str, model_name: str) -> None:
        self._dir = Path(output_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        tag = model_name.replace("/", "-")

        self._raw_json_path = self._dir / f"raw_{ts}_{tag}.json"
        self._raw_txt_path = self._dir / f"raw_{ts}_{tag}.txt"
        self._norm_json_path = self._dir / f"normalized_{ts}_{tag}.json"
        self._norm_txt_path = self._dir / f"normalized_{ts}_{tag}.txt"
        self._changes_json_path = self._dir / f"changes_{ts}_{tag}.json"

        # Open RAW file handles immediately so files exist on disk
        self._raw_json_fh: IO[str] = open(
            self._raw_json_path, "a", encoding="utf-8"
        )
        self._raw_txt_fh: IO[str] = open(
            self._raw_txt_path, "a", encoding="utf-8"
        )

        # In-memory accumulators for session-end writes
        self._normalized: List[NormalizedSegment] = []
        self._changes: List[NormalizationChange] = []
        self._last_paragraph_id: str | None = None

    # ------------------------------------------------------------------
    # RAW — append-only, called per committed segment
    # ------------------------------------------------------------------

    def append_raw(self, segment: RawSegment) -> None:
        """Append a raw segment to the JSON-Lines and TXT files.

        Flushes to disk immediately so data survives process kill.
        """
        # JSON-Lines (one JSON object per line — append-safe)
        self._raw_json_fh.write(
            json.dumps(segment.to_dict(), ensure_ascii=False) + "\n"
        )
        self._raw_json_fh.flush()

        # Plain text
        if (
            self._last_paragraph_id is not None
            and segment.paragraph_id != self._last_paragraph_id
        ):
            self._raw_txt_fh.write("\n")  # paragraph break
        self._raw_txt_fh.write(segment.to_txt_line() + "\n")
        self._raw_txt_fh.flush()
        self._last_paragraph_id = segment.paragraph_id

    # ------------------------------------------------------------------
    # NORMALIZED + CHANGES — accumulated, written at session end
    # ------------------------------------------------------------------

    def add_normalized(
        self, segment: NormalizedSegment, changes: List[NormalizationChange]
    ) -> None:
        """Buffer a normalized segment and its changes for session-end write."""
        self._normalized.append(segment)
        self._changes.extend(changes)

    def _close_raw_handles(self) -> None:
        """Close RAW file handles, ignoring errors."""
        for fh in (self._raw_json_fh, self._raw_txt_fh):
            try:
                fh.close()
            except Exception:
                pass

    def _write_session_files(self) -> None:
        """Write normalized and changes JSON/TXT files."""
        # normalized JSON
        with open(self._norm_json_path, "w", encoding="utf-8") as f:
            json.dump(
                [s.to_dict() for s in self._normalized],
                f,
                ensure_ascii=False,
                indent=2,
            )

        # normalized TXT
        last_para: str | None = None
        with open(self._norm_txt_path, "w", encoding="utf-8") as f:
            for seg in self._normalized:
                if last_para is not None and seg.paragraph_id != last_para:
                    f.write("\n")
                f.write(seg.to_txt_line() + "\n")
                last_para = seg.paragraph_id

        # changes JSON
        with open(self._changes_json_path, "w", encoding="utf-8") as f:
            json.dump(
                [c.to_dict() for c in self._changes],
                f,
                ensure_ascii=False,
                indent=2,
            )

    def finalize(self, timeout: float = 5.0) -> None:
        """Write all session-end files with a bounded timeout.

        On timeout, still closes RAW file handles to avoid data loss.
        """
        self._close_raw_handles()

        t = threading.Thread(target=self._write_session_files, daemon=True)
        t.start()
        t.join(timeout=timeout)
        if t.is_alive():
            print(
                f"[shutdown] WARNING: writer.finalize() did not complete "
                f"within {timeout}s — RAW files are safe on disk"
            )

    # ------------------------------------------------------------------
    # Accessors (for display / diagnostics)
    # ------------------------------------------------------------------

    @property
    def raw_json_path(self) -> Path:
        return self._raw_json_path

    @property
    def normalized_json_path(self) -> Path:
        return self._norm_json_path

    @property
    def changes_json_path(self) -> Path:
        return self._changes_json_path
