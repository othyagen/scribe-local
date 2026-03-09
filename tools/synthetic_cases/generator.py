"""Synthetic case generator — assembles audio, transcript, and ground truth.

Orchestrates TTS synthesis, audio environment simulation, and file output
for each scenario.  All generation is deterministic given the same inputs.

Output structure per case:
  test_data/synthetic/<case_id>/
    audio.wav          16 kHz mono 16-bit PCM
    transcript.txt     speaker-labeled transcript
    ground_truth.json  expected clinical facts
    meta.json          generation metadata
"""

from __future__ import annotations

import json
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tools.synthetic_cases.audio_env import (
    AudioEnvConfig,
    TARGET_RATE,
    apply_environment,
)
from tools.synthetic_cases.tts_engine import (
    VoiceConfig,
    synthesize_utterance,
    voice_config_from_hint,
)


@dataclass
class GeneratorConfig:
    """Configuration for the case generator."""

    output_dir: str = "test_data/synthetic"

    # Audio environment
    audio_env: AudioEnvConfig = field(default_factory=AudioEnvConfig)

    # Per-speaker voice overrides (speaker_id → VoiceConfig)
    voice_overrides: dict[str, VoiceConfig] = field(default_factory=dict)

    # Timing
    pause_between_turns_sec: float = 0.8
    pause_variation_sec: float = 0.0  # deterministic by default

    # Per-speaker environment overrides (e.g. patient is far from mic)
    speaker_env_overrides: dict[str, AudioEnvConfig] = field(default_factory=dict)

    # Random seed for reproducibility
    seed: int = 42


def generate_case(
    scenario: dict,
    config: GeneratorConfig | None = None,
) -> Path:
    """Generate a complete synthetic case from a scenario definition.

    Args:
        scenario: scenario dict from :mod:`scenarios`.
        config: generator configuration.  ``None`` uses defaults.

    Returns:
        Path to the output directory containing the generated files.
    """
    if config is None:
        config = GeneratorConfig()

    case_id = scenario["case_id"]
    out_dir = Path(config.output_dir) / case_id
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(config.seed)

    # Build voice configs from participant hints + overrides
    voice_map = _build_voice_map(scenario, config)

    # Synthesize and assemble audio
    segments = _synthesize_dialogue(scenario, voice_map, config, rng)
    audio = _assemble_audio(segments, config, rng)

    # Apply global environment
    audio = apply_environment(audio, config.audio_env, rng)

    # Write outputs
    _write_wav(out_dir / "audio.wav", audio)
    _write_transcript(out_dir / "transcript.txt", scenario)
    _write_ground_truth(out_dir / "ground_truth.json", scenario)
    _write_meta(out_dir / "meta.json", scenario, config, segments)

    return out_dir


# ── internal helpers ─────────────────────────────────────────────


@dataclass
class _SynthSegment:
    """A synthesized dialogue turn."""
    speaker_id: str
    text: str
    audio: np.ndarray
    t0: float = 0.0
    t1: float = 0.0


def _build_voice_map(
    scenario: dict,
    config: GeneratorConfig,
) -> dict[str, VoiceConfig]:
    """Map speaker_id to VoiceConfig from hints + overrides."""
    voice_map: dict[str, VoiceConfig] = {}
    for p in scenario.get("participants", []):
        sid = p["speaker_id"]
        if sid in config.voice_overrides:
            voice_map[sid] = config.voice_overrides[sid]
        else:
            hint = p.get("voice_hint", "male")
            voice_map[sid] = voice_config_from_hint(hint)
    return voice_map


def _synthesize_dialogue(
    scenario: dict,
    voice_map: dict[str, VoiceConfig],
    config: GeneratorConfig,
    rng: np.random.Generator,
) -> list[_SynthSegment]:
    """Synthesize each dialogue turn into audio segments."""
    segments: list[_SynthSegment] = []

    for turn in scenario.get("dialogue", []):
        sid = turn["speaker_id"]
        text = turn["text"]
        voice = voice_map.get(sid, VoiceConfig())

        audio = synthesize_utterance(text, voice)

        # Apply per-speaker environment if configured
        speaker_env = config.speaker_env_overrides.get(sid)
        if speaker_env is not None:
            audio = apply_environment(audio, speaker_env, rng)

        segments.append(_SynthSegment(speaker_id=sid, text=text, audio=audio))

    return segments


def _assemble_audio(
    segments: list[_SynthSegment],
    config: GeneratorConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Concatenate segments with pauses, recording timestamps."""
    parts: list[np.ndarray] = []
    cursor: float = 0.0

    for i, seg in enumerate(segments):
        # Add inter-turn pause (not before first turn)
        if i > 0:
            pause_sec = config.pause_between_turns_sec
            if config.pause_variation_sec > 0:
                pause_sec += rng.uniform(
                    -config.pause_variation_sec,
                    config.pause_variation_sec,
                )
                pause_sec = max(0.1, pause_sec)

            pause_samples = int(pause_sec * TARGET_RATE)
            parts.append(np.zeros(pause_samples, dtype=np.float32))
            cursor += pause_sec

        seg.t0 = cursor
        duration = len(seg.audio) / TARGET_RATE
        seg.t1 = cursor + duration
        parts.append(seg.audio)
        cursor += duration

    if not parts:
        return np.zeros(0, dtype=np.float32)

    return np.concatenate(parts)


def _write_wav(path: Path, audio: np.ndarray) -> None:
    """Write a 16 kHz mono 16-bit PCM WAV file."""
    # Clip and convert to int16
    clipped = np.clip(audio, -1.0, 1.0)
    pcm = (clipped * 32767).astype(np.int16)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(TARGET_RATE)
        wf.writeframes(pcm.tobytes())


def _write_transcript(path: Path, scenario: dict) -> None:
    """Write the speaker-labeled transcript."""
    # Build speaker label map
    label_map: dict[str, str] = {}
    for p in scenario.get("participants", []):
        label_map[p["speaker_id"]] = p.get("label", p["speaker_id"])

    lines: list[str] = []
    for turn in scenario.get("dialogue", []):
        label = label_map.get(turn["speaker_id"], turn["speaker_id"])
        lines.append(f"[{label}] {turn['text']}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_ground_truth(path: Path, scenario: dict) -> None:
    """Write the ground truth JSON."""
    gt = {
        "case_id": scenario["case_id"],
        "encounter_type": scenario["encounter_type"],
        "theme": scenario["theme"],
        "participants": scenario["participants"],
        "dialogue": scenario["dialogue"],
        **scenario["ground_truth"],
    }
    path.write_text(
        json.dumps(gt, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_meta(
    path: Path,
    scenario: dict,
    config: GeneratorConfig,
    segments: list[_SynthSegment],
) -> None:
    """Write generation metadata JSON."""
    meta = {
        "case_id": scenario["case_id"],
        "encounter_type": scenario["encounter_type"],
        "theme": scenario["theme"],
        "generator_version": "1.0",
        "audio_env_mode": config.audio_env.mode,
        "seed": config.seed,
        "sample_rate": TARGET_RATE,
        "total_duration_sec": round(
            segments[-1].t1 if segments else 0.0, 3
        ),
        "num_turns": len(segments),
        "segments": [
            {
                "speaker_id": s.speaker_id,
                "t0": round(s.t0, 3),
                "t1": round(s.t1, 3),
                "text": s.text,
            }
            for s in segments
        ],
    }
    path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
