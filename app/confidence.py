"""Confidence report — flags low-quality ASR segments without modifying outputs."""

from __future__ import annotations

import json
from pathlib import Path

# Conservative thresholds — flag obvious problems only
NO_SPEECH_THRESHOLD = 0.6
AVG_LOGPROB_THRESHOLD = -1.0
COMPRESSION_RATIO_HIGH = 2.4


def build_confidence_report(entries: list[dict]) -> dict:
    """Build a confidence report from per-segment ASR quality metrics.

    Each entry must have keys: seg_id, t0, t1, avg_logprob,
    no_speech_prob, compression_ratio.  Metric values may be None
    (missing); None metrics are not flagged.

    Returns a report dict with thresholds, per-segment flags,
    flagged_count, and total_count.
    """
    segments: list[dict] = []
    flagged_count = 0

    for entry in entries:
        flags: list[str] = []
        nsp = entry.get("no_speech_prob")
        alp = entry.get("avg_logprob")
        cr = entry.get("compression_ratio")

        if nsp is not None and nsp > NO_SPEECH_THRESHOLD:
            flags.append("no_speech")
        if alp is not None and alp < AVG_LOGPROB_THRESHOLD:
            flags.append("low_confidence")
        if cr is not None and cr > COMPRESSION_RATIO_HIGH:
            flags.append("repetitive")
        if nsp is None and alp is None and cr is None:
            flags.append("missing_metrics")

        if flags:
            flagged_count += 1

        segments.append({
            "seg_id": entry["seg_id"],
            "t0": entry["t0"],
            "t1": entry["t1"],
            "avg_logprob": alp,
            "no_speech_prob": nsp,
            "compression_ratio": cr,
            "flags": flags,
        })

    return {
        "thresholds": {
            "no_speech_prob": NO_SPEECH_THRESHOLD,
            "avg_logprob": AVG_LOGPROB_THRESHOLD,
            "compression_ratio_high": COMPRESSION_RATIO_HIGH,
        },
        "segments": segments,
        "flagged_count": flagged_count,
        "total_count": len(entries),
    }


def write_confidence_report(
    report: dict, output_dir: str, timestamp: str
) -> Path:
    """Write the confidence report JSON to the session output directory."""
    out = Path(output_dir) / f"confidence_report_{timestamp}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return out
