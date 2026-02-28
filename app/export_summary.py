"""Render session report as human-readable Markdown summary."""

from __future__ import annotations

from pathlib import Path


def build_summary_markdown(report: dict) -> str:
    """Build a Markdown summary string from a session report dict."""
    lines: list[str] = []

    ts = report.get("session_ts", "unknown")
    lines.append(f"# Session Summary — {ts}")
    lines.append("")

    # Configuration
    cfg = report.get("config", {})
    asr = cfg.get("asr", {})
    diar = cfg.get("diarization", {})
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- **Language:** {cfg.get('language', 'n/a')}")
    lines.append(f"- **ASR model:** {asr.get('model', 'n/a')}")
    lines.append(f"- **Compute device:** {asr.get('device', 'n/a')}")
    lines.append(f"- **Compute type:** {asr.get('compute_type', 'n/a')}")
    lines.append(f"- **Diarization backend:** {diar.get('backend', 'n/a')}")
    if diar.get("calibration_profile"):
        lines.append(f"- **Calibration profile:** {diar['calibration_profile']}")
    lines.append("")

    # Feature flags
    flags = report.get("feature_flags", {})
    if flags:
        lines.append("## Feature Flags")
        lines.append("")
        for key, val in flags.items():
            status = "on" if val else "off"
            lines.append(f"- `{key}`: {status}")
        lines.append("")

    # Audio precheck
    precheck = report.get("audio_precheck")
    if precheck and precheck.get("enabled"):
        lines.append("## Audio Pre-check")
        lines.append("")
        lines.append(f"- **Duration:** {precheck.get('duration_sec', 0):.1f} s")
        lines.append(f"- **Peak:** {precheck.get('peak_dbfs', 0):.1f} dBFS")
        lines.append(f"- **RMS:** {precheck.get('rms_dbfs', 0):.1f} dBFS")
        lines.append(f"- **SNR estimate:** {precheck.get('snr_db_est', 0):.1f} dB")
        lines.append(f"- **Clipping rate:** {precheck.get('clipping_rate', 0):.4%}")
        passed = precheck.get("passed", True)
        lines.append(f"- **Passed:** {'yes' if passed else 'no'}")
        warnings = precheck.get("warnings", [])
        if warnings:
            lines.append(f"- **Warnings:**")
            for w in warnings:
                lines.append(f"  - {w}")
        lines.append("")

    # Stats
    stats = report.get("stats", {})
    lines.append("## Statistics")
    lines.append("")
    lines.append(f"- **Segments:** {stats.get('segment_count', 0)}")
    if stats.get("turns_before_smoothing") is not None:
        lines.append(
            f"- **Diarization turns:** {stats['turns_before_smoothing']}"
            f" → {stats.get('turns_after_smoothing', '?')} (after smoothing)"
        )
    if stats.get("clusters_total") is not None:
        lines.append(f"- **Clusters:** {stats['clusters_total']} total,"
                      f" {stats.get('clusters_assigned', 0)} assigned,"
                      f" {stats.get('clusters_unknown', 0)} unknown")
    if stats.get("overlaps_marked") is not None:
        lines.append(f"- **Overlaps marked:** {stats['overlaps_marked']}")
    if stats.get("embeddings_computed") is not None:
        lines.append(f"- **Embeddings computed:** {stats['embeddings_computed']}")
    lines.append("")

    # Output files
    outputs = report.get("outputs", {})
    present = {k: v for k, v in outputs.items() if v is not None}
    if present:
        lines.append("## Output Files")
        lines.append("")
        for key, path in present.items():
            lines.append(f"- **{key}:** `{path}`")
        lines.append("")

    return "\n".join(lines)


def write_summary(
    report: dict,
    output_dir: str,
    session_ts: str,
) -> Path:
    """Write Markdown summary to ``session_summary_<session_ts>.md``."""
    content = build_summary_markdown(report)
    p = Path(output_dir) / f"session_summary_{session_ts}.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p
