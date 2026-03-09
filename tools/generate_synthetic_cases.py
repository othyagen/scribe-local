#!/usr/bin/env python
"""CLI entry point for synthetic clinical case generation.

Usage:
    # Generate all scenarios with default settings (clean audio):
    python -m tools.generate_synthetic_cases

    # Generate a specific scenario:
    python -m tools.generate_synthetic_cases --case chest_pain_consultation

    # Generate with telephone simulation:
    python -m tools.generate_synthetic_cases --env telephone

    # Generate with noisy room:
    python -m tools.generate_synthetic_cases --env noisy --noise-level 0.01

    # Generate with distance simulation (patient far from mic):
    python -m tools.generate_synthetic_cases --patient-distance far

    # List available scenarios:
    python -m tools.generate_synthetic_cases --list

    # Preview playback (requires sounddevice):
    python -m tools.generate_synthetic_cases --case chest_pain_consultation --play
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tools.synthetic_cases.audio_env import AudioEnvConfig
from tools.synthetic_cases.generator import GeneratorConfig, generate_case
from tools.synthetic_cases.scenarios import SCENARIOS, list_scenarios


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate synthetic clinical cases for SCRIBE testing.",
    )
    parser.add_argument(
        "--case", type=str, default=None,
        help="Generate a specific case ID (default: all)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available scenario IDs and exit",
    )
    parser.add_argument(
        "--output-dir", type=str, default="test_data/synthetic",
        help="Output directory (default: test_data/synthetic)",
    )
    parser.add_argument(
        "--env", type=str, default="clean",
        choices=["clean", "telephone", "noisy", "distance_near", "distance_far"],
        help="Audio environment mode (default: clean)",
    )
    parser.add_argument(
        "--noise-level", type=float, default=0.005,
        help="Noise level for noisy mode (default: 0.005)",
    )
    parser.add_argument(
        "--patient-distance", type=str, default=None,
        choices=["near", "far"],
        help="Simulate patient distance from microphone",
    )
    parser.add_argument(
        "--rate", type=int, default=160,
        help="TTS speaking rate in WPM (default: 160)",
    )
    parser.add_argument(
        "--pause", type=float, default=0.8,
        help="Pause between turns in seconds (default: 0.8)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--play", action="store_true",
        help="Play back generated audio after generation",
    )

    args = parser.parse_args(argv)

    if args.list:
        print("Available scenarios:")
        for cid in list_scenarios():
            s = SCENARIOS[cid]
            print(f"  {cid}  ({s['encounter_type']}, theme={s['theme']})")
        return 0

    # Build config
    audio_env = AudioEnvConfig(mode=args.env, noise_level=args.noise_level)

    speaker_env: dict[str, AudioEnvConfig] = {}
    if args.patient_distance == "near":
        speaker_env["spk_1"] = AudioEnvConfig(mode="distance_near")
    elif args.patient_distance == "far":
        speaker_env["spk_1"] = AudioEnvConfig(mode="distance_far")

    config = GeneratorConfig(
        output_dir=args.output_dir,
        audio_env=audio_env,
        pause_between_turns_sec=args.pause,
        speaker_env_overrides=speaker_env,
        seed=args.seed,
    )

    # Select scenarios
    if args.case:
        if args.case not in SCENARIOS:
            print(f"ERROR: Unknown case '{args.case}'", file=sys.stderr)
            print(f"Available: {', '.join(list_scenarios())}", file=sys.stderr)
            return 1
        cases = [args.case]
    else:
        cases = list_scenarios()

    # Generate
    for case_id in cases:
        scenario = SCENARIOS[case_id]
        print(f"Generating: {case_id} ...", end=" ", flush=True)
        out_dir = generate_case(scenario, config)
        print(f"→ {out_dir}")

        if args.play:
            _play_audio(out_dir / "audio.wav")

    print(f"\nDone. Generated {len(cases)} case(s) in {config.output_dir}/")
    return 0


def _play_audio(wav_path: Path) -> None:
    """Preview playback using sounddevice."""
    try:
        import sounddevice as sd
        import wave
        import numpy as np

        with wave.open(str(wav_path), "rb") as wf:
            rate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

        print(f"  Playing {wav_path.name} ({len(audio)/rate:.1f}s) ...")
        sd.play(audio, samplerate=rate)
        sd.wait()
    except ImportError:
        print("  (sounddevice not available for playback)")
    except Exception as exc:
        print(f"  Playback error: {exc}")


if __name__ == "__main__":
    sys.exit(main())
