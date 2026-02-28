"""Export diarized segments as SRT or WebVTT subtitles."""

from __future__ import annotations

from pathlib import Path


def _clamp_time(t: float) -> float:
    return max(0.0, t)


def _fix_segment(seg: dict) -> dict | None:
    """Validate and fix a segment. Returns None if segment should be skipped."""
    text = seg.get("text", "")
    if not text or not text.strip():
        return None
    t0 = _clamp_time(seg.get("t0", 0.0))
    t1 = _clamp_time(seg.get("t1", 0.0))
    if t1 <= t0:
        t1 = t0 + 0.01
    return {
        "t0": t0,
        "t1": t1,
        "speaker": seg.get("speaker", ""),
        "text": text.strip(),
    }


def format_srt_time(seconds: float) -> str:
    """Format seconds as SRT timestamp: HH:MM:SS,mmm"""
    seconds = _clamp_time(seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def format_vtt_time(seconds: float) -> str:
    """Format seconds as WebVTT timestamp: HH:MM:SS.mmm"""
    seconds = _clamp_time(seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def build_srt_cues(segments: list[dict]) -> str:
    """Build SRT content from a list of segment dicts.

    Each segment must have keys: t0, t1, speaker, text.
    """
    lines: list[str] = []
    index = 0
    for seg in segments:
        fixed = _fix_segment(seg)
        if fixed is None:
            continue
        index += 1
        start = format_srt_time(fixed["t0"])
        end = format_srt_time(fixed["t1"])
        speaker = fixed["speaker"]
        text = fixed["text"]
        lines.append(str(index))
        lines.append(f"{start} --> {end}")
        lines.append(f"[{speaker}] {text}")
        lines.append("")
    return "\n".join(lines)


def build_vtt_cues(segments: list[dict]) -> str:
    """Build WebVTT content from a list of segment dicts.

    Each segment must have keys: t0, t1, speaker, text.
    """
    lines: list[str] = ["WEBVTT", ""]
    for seg in segments:
        fixed = _fix_segment(seg)
        if fixed is None:
            continue
        start = format_vtt_time(fixed["t0"])
        end = format_vtt_time(fixed["t1"])
        speaker = fixed["speaker"]
        text = fixed["text"]
        lines.append(f"{start} --> {end}")
        lines.append(f"[{speaker}] {text}")
        lines.append("")
    return "\n".join(lines)


def write_srt(
    segments: list[dict],
    output_dir: str,
    session_ts: str,
) -> Path:
    """Write SRT file and return path."""
    content = build_srt_cues(segments)
    p = Path(output_dir) / f"subtitles_{session_ts}.srt"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


def write_vtt(
    segments: list[dict],
    output_dir: str,
    session_ts: str,
) -> Path:
    """Write WebVTT file and return path."""
    content = build_vtt_cues(segments)
    p = Path(output_dir) / f"subtitles_{session_ts}.vtt"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p
