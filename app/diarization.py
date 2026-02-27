"""Speaker diarization — metadata-only, never modifies text.

The diarization layer determines *who* is speaking for a given audio
segment.  It runs BEFORE ASR and only attaches a ``speaker_id`` string.

Architecture:
    * ``Diarizer`` — abstract base class defining the interface.
    * ``DefaultDiarizer`` — assigns every segment to ``spk_0``.
    * ``create_diarizer()`` — factory that returns the correct backend
      based on config.  Drop-in replacements (e.g. pyannote) only need
      to subclass ``Diarizer`` and register in the factory.
"""

from __future__ import annotations

import json
import os
import wave
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import torch

from app.config import AppConfig


class Diarizer(ABC):
    """Base class for all diarization backends."""

    @abstractmethod
    def identify_speaker(self, audio: np.ndarray, sample_rate: int) -> str:
        """Return a speaker_id string for the given audio segment.

        Must NEVER inspect or modify transcribed text.
        Must NEVER assign roles (patient / clinician / etc.).
        """

    def detect_speaker_change(
        self,
        previous_audio: np.ndarray | None,
        current_audio: np.ndarray,
        sample_rate: int,
    ) -> bool:
        """Return True if the speaker changed between two audio segments.

        Default implementation always returns False (single speaker).
        """
        return False


class DefaultDiarizer(Diarizer):
    """Single-speaker stub — assigns spk_0 to all audio."""

    def identify_speaker(self, audio: np.ndarray, sample_rate: int) -> str:
        return "spk_0"

    def detect_speaker_change(
        self,
        previous_audio: np.ndarray | None,
        current_audio: np.ndarray,
        sample_rate: int,
    ) -> bool:
        return False


def create_diarizer(config: AppConfig) -> Diarizer:
    """Factory — returns the diarization backend specified in config.

    When backend is "pyannote", real-time diarization still uses the
    default single-speaker stub.  Post-session diarization on the full
    WAV is handled separately by ``run_pyannote_diarization()``.
    """
    backend = config.diarization.backend
    if backend in ("default", "pyannote"):
        return DefaultDiarizer()
    raise ValueError(
        f"Unknown diarization backend: {backend!r}. "
        "Available: 'default', 'pyannote'."
    )


def run_pyannote_diarization(wav_path: Path, output_dir: str) -> Path:
    """Run pyannote speaker diarization on a WAV file.

    Requires ``HF_TOKEN`` environment variable for Hugging Face auth.
    Writes ``diarization_<timestamp>.json`` to *output_dir*.

    Returns the path to the written JSON file.
    """
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise ValueError(
            "HF_TOKEN environment variable is required for pyannote diarization"
        )

    from pyannote.audio import Pipeline  # lazy import — heavy dependency

    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token,
    )

    # Load WAV as in-memory waveform to avoid torchcodec/FFmpeg dependency
    with wave.open(str(wav_path), "rb") as wf:
        sample_rate = wf.getframerate()
        pcm_bytes = wf.readframes(wf.getnframes())
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32767
    waveform = torch.from_numpy(samples).unsqueeze(0)  # (1, num_samples)

    result = pipeline({"waveform": waveform, "sample_rate": sample_rate})

    # pyannote 4.x returns DiarizeOutput; extract the Annotation
    annotation = getattr(result, "speaker_diarization", result)

    # Map pyannote labels to spk_0, spk_1, ... (ordered by first appearance)
    label_map: dict[str, str] = {}
    turns: list[dict] = []

    for turn, _, label in annotation.itertracks(yield_label=True):
        if label not in label_map:
            label_map[label] = f"spk_{len(label_map)}"
        turns.append({
            "start": round(turn.start, 3),
            "end": round(turn.end, 3),
            "speaker": label_map[label],
        })

    # Extract timestamp from WAV filename: audio_<timestamp>.wav
    ts = wav_path.stem.removeprefix("audio_")
    out = Path(output_dir) / f"diarization_{ts}.json"

    with open(out, "w", encoding="utf-8") as f:
        json.dump({"turns": turns}, f, ensure_ascii=False, indent=2)

    return out


def smooth_turns(
    turns: list[dict],
    min_turn_sec: float = 0.7,
    gap_merge_sec: float = 0.3,
) -> list[dict]:
    """Smooth diarization turns to reduce short backchannel flips.

    Pass 1 — merge short turns (< min_turn_sec) into the best neighbor:
        * Same-speaker neighbor preferred (prev first, then next).
        * Otherwise merge into the longer adjacent turn.
    Pass 2 — fill tiny same-speaker gaps (<= gap_merge_sec).

    Returns a new list; does not mutate the input.
    """
    if not turns:
        return []

    # Deep copy to avoid mutating input
    result = [dict(t) for t in turns]

    # ── Pass 1: merge short turns ─────────────────────────────────
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(result):
            t = result[i]
            duration = t["end"] - t["start"]
            if duration >= min_turn_sec:
                i += 1
                continue

            prev = result[i - 1] if i > 0 else None
            nxt = result[i + 1] if i < len(result) - 1 else None

            if prev and prev["speaker"] == t["speaker"]:
                prev["end"] = t["end"]
            elif nxt and nxt["speaker"] == t["speaker"]:
                nxt["start"] = t["start"]
            elif prev and nxt:
                prev_dur = prev["end"] - prev["start"]
                nxt_dur = nxt["end"] - nxt["start"]
                if prev_dur >= nxt_dur:
                    prev["end"] = t["end"]
                else:
                    nxt["start"] = t["start"]
            elif prev:
                prev["end"] = t["end"]
            elif nxt:
                nxt["start"] = t["start"]
            else:
                # Single turn shorter than threshold — keep it
                i += 1
                continue

            result.pop(i)
            changed = True
            # Don't increment i — re-check at same position

    # ── Pass 2: fill same-speaker gaps ────────────────────────────
    i = 0
    while i < len(result) - 1:
        curr = result[i]
        nxt = result[i + 1]
        gap = nxt["start"] - curr["end"]
        if curr["speaker"] == nxt["speaker"] and gap <= gap_merge_sec:
            curr["end"] = nxt["end"]
            result.pop(i + 1)
        else:
            i += 1

    return result


# ── speaker merge ────────────────────────────────────────────────────

def load_merge_map(output_dir: str, timestamp: str) -> dict[str, str]:
    """Load speaker merge map, or return empty dict if none exists."""
    path = Path(output_dir) / f"speaker_merge_{timestamp}.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_merge_map(
    merge_map: dict[str, str], output_dir: str, timestamp: str
) -> Path:
    """Write speaker merge map to ``speaker_merge_<timestamp>.json``."""
    path = Path(output_dir) / f"speaker_merge_{timestamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(merge_map, f, ensure_ascii=False, indent=2)
    return path


def resolve_merge_chains(merge_map: dict[str, str]) -> dict[str, str]:
    """Resolve transitive merge chains to final roots.

    Example: ``{spk_3: spk_2, spk_2: spk_0}`` → ``{spk_3: spk_0, spk_2: spk_0}``

    Raises ``ValueError`` on cycles or self-references.
    """
    if not merge_map:
        return {}

    resolved: dict[str, str] = {}
    for key in merge_map:
        visited: list[str] = [key]
        current = key
        while current in merge_map:
            current = merge_map[current]
            if current in visited:
                cycle = " → ".join(visited + [current])
                raise ValueError(
                    f"Cycle detected in speaker merge map: {cycle}"
                )
            visited.append(current)
        resolved[key] = current

    return resolved


def apply_merge_map(
    turns: list[dict], merge_map: dict[str, str]
) -> list[dict]:
    """Apply a resolved merge map to diarization turns.

    Replaces speaker IDs, then merges adjacent turns with the same
    speaker (touching or overlapping).  Returns a new list.
    """
    if not turns or not merge_map:
        return [dict(t) for t in turns]

    # Replace speaker IDs
    result = []
    for t in turns:
        t_copy = dict(t)
        t_copy["speaker"] = merge_map.get(t_copy["speaker"], t_copy["speaker"])
        result.append(t_copy)

    # Merge adjacent same-speaker turns
    i = 0
    while i < len(result) - 1:
        curr = result[i]
        nxt = result[i + 1]
        if curr["speaker"] == nxt["speaker"]:
            curr["end"] = max(curr["end"], nxt["end"])
            result.pop(i + 1)
        else:
            i += 1

    return result


_MIN_SEGMENT_SEC = 0.12


def clean_diarized_segments(segments: list[dict]) -> list[dict]:
    """Post-process relabeled segments: sort, dedup, merge, resolve overlaps.

    Input dicts must have keys: seg_id, t0, t1, old_speaker_id,
    new_speaker_id, normalized_text, paragraph_id.

    Returns a new list (does not mutate input).
    """
    import copy

    if not segments:
        return []

    segs = sorted(copy.deepcopy(segments), key=lambda s: (s["t0"], s["t1"]))

    # 1. Deduplicate near-identical consecutive segments
    deduped: list[dict] = [segs[0]]
    for s in segs[1:]:
        prev = deduped[-1]
        if (
            s["new_speaker_id"] == prev["new_speaker_id"]
            and abs(s["t0"] - prev["t0"]) < 0.1
            and abs(s["t1"] - prev["t1"]) < 0.1
            and s["normalized_text"] == prev["normalized_text"]
        ):
            continue
        deduped.append(s)

    # 2. Merge adjacent same-speaker segments (gap <= 0.4s, total <= 30s)
    merged: list[dict] = [deduped[0]]
    for s in deduped[1:]:
        prev = merged[-1]
        gap = s["t0"] - prev["t1"]
        combined_dur = s["t1"] - prev["t0"]
        if (
            s["new_speaker_id"] == prev["new_speaker_id"]
            and s.get("paragraph_id", "") == prev.get("paragraph_id", "")
            and gap <= 0.4
            and gap >= 0.0
            and combined_dur <= 30.0
        ):
            prev["t1"] = s["t1"]
            prev["normalized_text"] += " " + s["normalized_text"]
        else:
            merged.append(s)

    # 3. Resolve overlaps (strictly non-overlapping output)
    for i in range(len(merged) - 1):
        cur = merged[i]
        nxt = merged[i + 1]
        if nxt["t0"] < cur["t1"]:
            if cur["new_speaker_id"] == nxt["new_speaker_id"]:
                cur["t1"] = nxt["t0"]
            else:
                overlap = cur["t1"] - nxt["t0"]
                if overlap < 0.2:
                    cur["t1"] = nxt["t0"]
                else:
                    cur_dur = cur["t1"] - cur["t0"]
                    nxt_dur = nxt["t1"] - nxt["t0"]
                    if cur_dur <= nxt_dur:
                        cur["t1"] = nxt["t0"]
                    else:
                        nxt["t0"] = cur["t1"]

    # 4. Drop micro-segments (duration < _MIN_SEGMENT_SEC)
    return [s for s in merged if (s["t1"] - s["t0"]) >= _MIN_SEGMENT_SEC]


def relabel_segments(
    normalized_json_path: Path,
    diarization_json_path: Path,
    output_dir: str,
) -> tuple[Path, Path]:
    """Relabel ASR segments with speaker_ids from diarization overlap.

    For each normalized segment, computes time overlap with every
    diarization turn and assigns the speaker with the largest total
    overlap.  If no overlap exists, the original speaker_id is kept.

    Writes:
        diarized_segments_<timestamp>.json — relabeling map
        diarized_<timestamp>.txt — transcript with new speaker_ids

    Returns (json_path, txt_path).
    """
    from app.commit import _fmt_ts

    with open(normalized_json_path, encoding="utf-8") as f:
        segments = json.load(f)

    with open(diarization_json_path, encoding="utf-8") as f:
        turns = json.load(f)["turns"]

    relabeled: list[dict] = []

    for seg in segments:
        seg_t0 = seg["t0"]
        seg_t1 = seg["t1"]

        # Accumulate overlap per speaker
        speaker_overlap: dict[str, float] = {}
        for turn in turns:
            overlap = max(0.0, min(seg_t1, turn["end"]) - max(seg_t0, turn["start"]))
            if overlap > 0:
                speaker = turn["speaker"]
                speaker_overlap[speaker] = speaker_overlap.get(speaker, 0.0) + overlap

        if speaker_overlap:
            new_speaker = max(speaker_overlap, key=speaker_overlap.get)
        else:
            new_speaker = seg["speaker_id"]

        relabeled.append({
            "seg_id": seg["seg_id"],
            "t0": seg_t0,
            "t1": seg_t1,
            "old_speaker_id": seg["speaker_id"],
            "new_speaker_id": new_speaker,
            "normalized_text": seg["normalized_text"],
            "paragraph_id": seg.get("paragraph_id", ""),
        })

    relabeled = clean_diarized_segments(relabeled)

    # Extract timestamp from diarization filename
    ts = diarization_json_path.stem.removeprefix("diarization_")

    # Write JSON (strip internal fields so output format is unchanged)
    json_out = Path(output_dir) / f"diarized_segments_{ts}.json"
    json_data = [
        {k: v for k, v in r.items() if k not in ("normalized_text", "paragraph_id")}
        for r in relabeled
    ]
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    txt_out = Path(output_dir) / f"diarized_{ts}.txt"
    last_para: str | None = None
    with open(txt_out, "w", encoding="utf-8") as f:
        for rel in relabeled:
            para = rel.get("paragraph_id", "")
            if last_para is not None and para != last_para:
                f.write("\n")
            ts0 = _fmt_ts(rel["t0"])
            ts1 = _fmt_ts(rel["t1"])
            f.write(f"[{ts0} - {ts1}] [{rel['new_speaker_id']}] {rel['normalized_text']}\n")
            last_para = para

    return json_out, txt_out
