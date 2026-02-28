"""Session browser — read-only utilities for listing and inspecting sessions."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class SessionInfo:
    """Summary of a single recording session."""
    ts: str
    raw_path: Path
    segment_count: int
    duration_sec: float
    model_tag: str
    language: str
    has_audio: bool
    audio_parts_count: int
    has_normalized: bool
    has_diarization: bool
    has_tags: bool
    has_confidence: bool
    confidence_flagged_count: Optional[int]
    resume_possible: bool
    corrupt: bool = False


def scan_sessions(output_dir: Path) -> list[SessionInfo]:
    """Scan output_dir for sessions and return summaries (newest first).

    Corrupt RAW files are skipped with a warning printed to stderr.
    """
    results: list[SessionInfo] = []

    for raw_path in sorted(output_dir.glob("raw_*_*.json"), reverse=True):
        stem = raw_path.stem  # raw_<ts>_<model_tag>
        # Extract ts (YYYY-MM-DD_HH-MM-SS = 19 chars) starting at index 4
        if len(stem) < 24:  # "raw_" + 19 + "_" + at least 1
            continue
        ts = stem[4:23]
        model_tag = stem[24:]  # after "raw_<ts>_"

        info = _parse_raw_file(raw_path, ts, model_tag, output_dir)
        if info is None:
            continue
        results.append(info)

    return results


def show_session(output_dir: Path, ts: str) -> dict:
    """Build a detailed summary dict for a single session.

    Raises ValueError if the session cannot be found or is ambiguous.
    """
    candidates = list(output_dir.glob(f"raw_{ts}_*.json"))
    if not candidates:
        raise ValueError(f"no session found for timestamp {ts} in {output_dir}")
    if len(candidates) > 1:
        raise ValueError(
            f"multiple RAW files for timestamp {ts}: "
            + ", ".join(str(p) for p in candidates)
        )

    raw_path = candidates[0]
    model_tag = raw_path.stem[24:]

    info = _parse_raw_file(raw_path, ts, model_tag, output_dir)
    if info is None:
        raise ValueError(f"RAW file is corrupt: {raw_path}")

    result: dict = {
        "ts": info.ts,
        "raw_path": str(info.raw_path),
        "segment_count": info.segment_count,
        "duration_sec": info.duration_sec,
        "model_tag": info.model_tag,
        "language": info.language,
        "has_audio": info.has_audio,
        "audio_parts_count": info.audio_parts_count,
        "has_normalized": info.has_normalized,
        "has_diarization": info.has_diarization,
        "has_tags": info.has_tags,
        "has_confidence": info.has_confidence,
        "confidence_flagged_count": info.confidence_flagged_count,
        "resume_possible": info.resume_possible,
    }

    # Enrich with file paths
    files: dict[str, str] = {"raw": str(raw_path)}
    norm = list(output_dir.glob(f"normalized_{ts}_*.json"))
    if norm:
        files["normalized"] = str(norm[0])
    changes = list(output_dir.glob(f"changes_{ts}_*.json"))
    if changes:
        files["changes"] = str(changes[0])
    audio = output_dir / f"audio_{ts}.wav"
    if audio.exists():
        files["audio"] = str(audio)
    parts = sorted(output_dir.glob(f"audio_{ts}_part*.wav"))
    for p in parts:
        files[p.stem] = str(p)
    diar = output_dir / f"diarization_{ts}.json"
    if diar.exists():
        files["diarization"] = str(diar)
    tags = output_dir / f"speaker_tags_{ts}.json"
    if tags.exists():
        files["speaker_tags"] = str(tags)
    conf = output_dir / f"confidence_report_{ts}.json"
    if conf.exists():
        files["confidence_report"] = str(conf)
    diarized_txt = output_dir / f"diarized_{ts}.txt"
    if diarized_txt.exists():
        files["diarized_txt"] = str(diarized_txt)
    tagged_txt = output_dir / f"tag_labeled_{ts}.txt"
    if tagged_txt.exists():
        files["tag_labeled_txt"] = str(tagged_txt)
    result["files"] = files

    # Speaker count from diarization
    if info.has_diarization:
        try:
            with open(diar, encoding="utf-8") as f:
                diar_data = json.load(f)
            speakers = set(t["speaker"] for t in diar_data.get("turns", []))
            result["speaker_count"] = len(speakers)
        except Exception:
            result["speaker_count"] = None
    else:
        result["speaker_count"] = None

    # Confidence thresholds
    if info.has_confidence:
        try:
            with open(conf, encoding="utf-8") as f:
                conf_data = json.load(f)
            result["confidence_thresholds"] = conf_data.get("thresholds")
            result["confidence_total"] = conf_data.get("total_count")
        except Exception:
            pass

    # Resume reason
    if not info.resume_possible:
        result["resume_reason"] = "no audio file found"
    else:
        result["resume_reason"] = None

    return result


# ── internal helpers ─────────────────────────────────────────────────

def _parse_raw_file(
    raw_path: Path,
    ts: str,
    model_tag: str,
    output_dir: Path,
) -> SessionInfo | None:
    """Parse a RAW JSONL file and build SessionInfo.

    Returns None if the file is corrupt (prints warning to stderr).
    """
    segment_count = 0
    last_t1 = 0.0
    language = "unknown"

    try:
        with open(raw_path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    seg = json.loads(line)
                except json.JSONDecodeError:
                    print(
                        f"Warning: corrupt JSONL at line {line_num} in "
                        f"{raw_path.name} — skipping session {ts}",
                        file=sys.stderr,
                    )
                    return None
                t1 = seg.get("t1")
                if t1 is None:
                    print(
                        f"Warning: missing t1 at line {line_num} in "
                        f"{raw_path.name} — skipping session {ts}",
                        file=sys.stderr,
                    )
                    return None
                last_t1 = t1
                language = seg.get("language", language)
                segment_count += 1
    except OSError as e:
        print(f"Warning: cannot read {raw_path}: {e}", file=sys.stderr)
        return None

    if segment_count == 0:
        return None

    # Check companion files
    has_audio_main = (output_dir / f"audio_{ts}.wav").exists()
    audio_parts = list(output_dir.glob(f"audio_{ts}_part*.wav"))
    has_audio = has_audio_main or len(audio_parts) > 0
    audio_parts_count = (1 if has_audio_main else 0) + len(audio_parts)

    has_normalized = bool(list(output_dir.glob(f"normalized_{ts}_*.json")))
    has_diarization = (output_dir / f"diarization_{ts}.json").exists()
    has_tags = (output_dir / f"speaker_tags_{ts}.json").exists()

    conf_path = output_dir / f"confidence_report_{ts}.json"
    has_confidence = conf_path.exists()
    confidence_flagged: int | None = None
    if has_confidence:
        try:
            with open(conf_path, encoding="utf-8") as f:
                conf_data = json.load(f)
            confidence_flagged = conf_data.get("flagged_count")
        except Exception:
            pass

    return SessionInfo(
        ts=ts,
        raw_path=raw_path,
        segment_count=segment_count,
        duration_sec=last_t1,
        model_tag=model_tag,
        language=language,
        has_audio=has_audio,
        audio_parts_count=audio_parts_count,
        has_normalized=has_normalized,
        has_diarization=has_diarization,
        has_tags=has_tags,
        has_confidence=has_confidence,
        confidence_flagged_count=confidence_flagged,
        resume_possible=has_audio,
    )
