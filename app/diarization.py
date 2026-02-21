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
        })

    # Extract timestamp from diarization filename
    ts = diarization_json_path.stem.removeprefix("diarization_")

    json_out = Path(output_dir) / f"diarized_segments_{ts}.json"
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(relabeled, f, ensure_ascii=False, indent=2)

    txt_out = Path(output_dir) / f"diarized_{ts}.txt"
    last_para: str | None = None
    with open(txt_out, "w", encoding="utf-8") as f:
        for seg, rel in zip(segments, relabeled):
            para = seg.get("paragraph_id")
            if last_para is not None and para != last_para:
                f.write("\n")
            ts0 = _fmt_ts(seg["t0"])
            ts1 = _fmt_ts(seg["t1"])
            f.write(f"[{ts0} - {ts1}] [{rel['new_speaker_id']}] {seg['normalized_text']}\n")
            last_para = para

    return json_out, txt_out
