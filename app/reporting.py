"""Session report generation — consolidated per-session summary."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from app.config import AppConfig


def build_session_report(
    session_ts: str,
    config: AppConfig,
    segment_count: int,
    output_paths: dict[str, Optional[str]],
    diarization_stats: Optional[dict] = None,
    calibration_stats: Optional[dict] = None,
) -> dict:
    """Build a consolidated session report dict.

    Summarises config snapshot, feature flags, output file paths,
    and pipeline statistics for a single recording session.
    """
    diar = config.diarization
    report: dict = {
        "session_ts": session_ts,
        "config": {
            "language": config.language,
            "asr": {
                "model": config.asr.model,
                "device": config.asr.device,
                "compute_type": config.asr.compute_type,
            },
            "diarization": {
                "backend": diar.backend,
                "smoothing": diar.smoothing,
                "calibration_profile": diar.calibration_profile,
                "calibration_enabled": diar.calibration_enabled,
                "overlap_stabilizer_enabled": diar.overlap_stabilizer_enabled,
                "prototype_matching_enabled": diar.prototype_matching_enabled,
                "min_duration_filter_enabled": diar.min_duration_filter_enabled,
            },
            "reporting": {
                "session_report_enabled": config.reporting.session_report_enabled,
            },
        },
        "feature_flags": {
            "calibration_enabled": diar.calibration_enabled,
            "overlap_stabilizer_enabled": diar.overlap_stabilizer_enabled,
            "prototype_matching_enabled": diar.prototype_matching_enabled,
            "min_duration_filter_enabled": diar.min_duration_filter_enabled,
        },
        "outputs": output_paths,
        "stats": {
            "segment_count": segment_count,
        },
    }

    if diarization_stats is not None:
        report["stats"].update(diarization_stats)
    if calibration_stats is not None:
        report["stats"].update(calibration_stats)

    return report


def write_session_report(
    report: dict,
    output_dir: str,
    session_ts: str,
) -> Path:
    """Write session report to ``session_report_<session_ts>.json``."""
    p = Path(output_dir) / f"session_report_{session_ts}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return p
