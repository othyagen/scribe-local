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

    # Show case folder location:
    python -m tools.generate_synthetic_cases --case chest_pain_consultation --show

    # Open case folder in file browser:
    python -m tools.generate_synthetic_cases --case chest_pain_consultation --open

    # Step-by-step explanation:
    python -m tools.generate_synthetic_cases --explain
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import wave
from pathlib import Path

from tools.synthetic_cases.audio_env import AudioEnvConfig
from tools.synthetic_cases.generator import GeneratorConfig, generate_case
from tools.synthetic_cases.scenarios import SCENARIOS, list_scenarios


# ── explain text ─────────────────────────────────────────────────

_EXPLAIN_TEXT = """\
Synthetic Case Generator — How It Works
========================================

The synthetic case generator creates realistic clinical audio scenarios
for testing SCRIBE's full pipeline: ASR, diarization, normalization,
extraction, classification, and export.

How it works:
  1. Scenarios are defined in tools/synthetic_cases/scenarios.py
     Each has a dialogue, participants, and ground truth clinical facts.
  2. Text-to-speech (pyttsx3) converts each dialogue turn into audio
     using different voices for doctor vs patient.
  3. Audio environment effects are optionally applied (telephone
     bandpass, background noise, distance simulation).
  4. Four files are written per case:
       audio.wav          16 kHz mono PCM — ready for SCRIBE
       transcript.txt     Speaker-labeled reference transcript
       ground_truth.json  Expected symptoms, medications, negations, etc.
       meta.json          Timestamps, config, segment boundaries

Where files are stored:
  Default: test_data/synthetic/<case_id>/
  Override: --output-dir <path>

Workflow: generate, listen, run SCRIBE, compare
------------------------------------------------

Step 1 — Generate a case:
  python -m tools.generate_synthetic_cases --case chest_pain_consultation

Step 2 — Listen to the audio:
  python -m tools.generate_synthetic_cases --case chest_pain_consultation --play

  Or open the WAV file directly:
  test_data/synthetic/chest_pain_consultation/audio.wav

Step 3 — Read the reference transcript:
  test_data/synthetic/chest_pain_consultation/transcript.txt

Step 4 — Inspect ground truth (expected clinical facts):
  test_data/synthetic/chest_pain_consultation/ground_truth.json

Step 5 — Run SCRIBE on the generated audio:
  (Feed the audio.wav through SCRIBE's ASR pipeline, or use the
  transcript segments directly for extraction testing.)

  For extraction testing without ASR:
    python -c "
    import json
    from app.clinical_state import build_clinical_state
    meta = json.loads(open('test_data/synthetic/chest_pain_consultation/meta.json').read())
    segments = [
        {'seg_id': f'seg_{i:04d}', 't0': s['t0'], 't1': s['t1'],
         'speaker_id': s['speaker_id'], 'normalized_text': s['text']}
        for i, s in enumerate(meta['segments'])
    ]
    state = build_clinical_state(segments)
    print('Symptoms:', state['symptoms'])
    print('Negations:', state['negations'])
    print('Medications:', state['medications'])
    "

Step 6 — Compare expected vs actual:
  Compare SCRIBE's extracted symptoms/negations/medications against
  the ground_truth.json values. A future benchmark scorer will
  automate this comparison.

Audio environment modes:
  --env clean           No processing (default)
  --env telephone       Bandpass 300-3400 Hz
  --env noisy           Additive Gaussian noise (--noise-level 0.005)
  --env distance_near   Mild attenuation
  --env distance_far    Strong attenuation + low-pass filter

  Per-speaker override:
  --patient-distance far   Patient sounds far from microphone
"""


# ── case summary printer ─────────────────────────────────────────


def _print_case_summary(
    case_id: str,
    scenario: dict,
    out_dir: Path,
    env_mode: str,
) -> None:
    """Print a structured summary after case generation."""
    duration = ""
    meta_path = out_dir / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            duration = f" ({meta['total_duration_sec']:.1f}s audio, {meta['num_turns']} turns)"
        except Exception:
            pass

    n_symptoms = len(scenario.get("ground_truth", {}).get("symptoms", []))
    n_negations = len(scenario.get("ground_truth", {}).get("negations", []))
    n_meds = len(scenario.get("ground_truth", {}).get("medications", []))

    path_str = str(out_dir).replace("\\", "/")

    print(f"""
Synthetic case generated:

  Case ID:      {case_id}
  Scenario:     {scenario['encounter_type']}
  Theme:        {scenario['theme']}
  Environment:  {env_mode}{duration}
  Ground truth: {n_symptoms} symptoms, {n_negations} negations, {n_meds} medications

  Files created:
    audio.wav          16 kHz mono PCM
    transcript.txt     Speaker-labeled transcript
    ground_truth.json  Expected clinical facts
    meta.json          Generation metadata

  Location:
    {path_str}/

  Next steps:
    1. Listen:     python -m tools.generate_synthetic_cases --case {case_id} --play
    2. Transcript: type {path_str}/transcript.txt
    3. Ground truth: type {path_str}/ground_truth.json""")


def _print_batch_summary(
    case_dirs: list[tuple[str, Path]],
    output_dir: str,
    env_mode: str,
) -> None:
    """Print summary after generating multiple cases."""
    print(f"\nGenerated {len(case_dirs)} case(s)  [env={env_mode}]\n")
    for case_id, out_dir in case_dirs:
        path_str = str(out_dir).replace("\\", "/")
        meta_path = out_dir / "meta.json"
        duration = ""
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                duration = f"  {meta['total_duration_sec']:.1f}s"
            except Exception:
                pass
        print(f"  {case_id:<35}{duration}  {path_str}/")

    print(f"\nTo play a case:   python -m tools.generate_synthetic_cases --case <case_id> --play")
    print(f"To inspect:       python -m tools.generate_synthetic_cases --case <case_id> --show")


# ── show command ─────────────────────────────────────────────────


def _show_case(case_id: str, output_dir: str) -> int:
    """Show case folder contents and ground truth summary."""
    case_dir = Path(output_dir) / case_id
    if not case_dir.exists():
        print(f"Case not found: {case_dir}", file=sys.stderr)
        print(f"Run generation first:", file=sys.stderr)
        print(f"  python -m tools.generate_synthetic_cases --case {case_id}", file=sys.stderr)
        return 1

    path_str = str(case_dir).replace("\\", "/")
    print(f"\nCase: {case_id}")
    print(f"Location: {path_str}/\n")

    print("Files:")
    for f in sorted(case_dir.iterdir()):
        size = f.stat().st_size
        if size > 1024:
            size_str = f"{size / 1024:.1f} KB"
        else:
            size_str = f"{size} B"
        print(f"  {f.name:<25} {size_str:>10}")

    # Show ground truth summary
    gt_path = case_dir / "ground_truth.json"
    if gt_path.exists():
        try:
            gt = json.loads(gt_path.read_text(encoding="utf-8"))
            print(f"\nGround truth:")
            print(f"  Encounter:   {gt.get('encounter_type', '?')}")
            print(f"  Theme:       {gt.get('theme', '?')}")
            print(f"  Symptoms:    {', '.join(gt.get('symptoms', []))}")
            print(f"  Negations:   {', '.join(gt.get('negations', []))}")
            print(f"  Medications: {', '.join(gt.get('medications', []))}")
            print(f"  Durations:   {', '.join(gt.get('durations', []))}")
        except Exception:
            pass

    # Show meta summary
    meta_path = case_dir / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            print(f"\nAudio:")
            print(f"  Duration:    {meta['total_duration_sec']:.1f}s")
            print(f"  Turns:       {meta['num_turns']}")
            print(f"  Environment: {meta['audio_env_mode']}")
            print(f"  Seed:        {meta['seed']}")
        except Exception:
            pass

    return 0


# ── open folder helper ───────────────────────────────────────────


def _open_folder(path: Path) -> None:
    """Open a folder in the system file browser."""
    path_str = str(path)
    try:
        if platform.system() == "Windows":
            os.startfile(path_str)
        elif platform.system() == "Darwin":
            subprocess.run(["open", path_str], check=True)
        else:
            subprocess.run(["xdg-open", path_str], check=True)
    except Exception:
        print(f"  Could not open file browser. Path:\n  {path_str}")


# ── playback ─────────────────────────────────────────────────────


def _play_audio(wav_path: Path) -> None:
    """Preview playback using sounddevice."""
    try:
        import sounddevice as sd
        import numpy as np

        with wave.open(str(wav_path), "rb") as wf:
            rate = wf.getframerate()
            frames = wf.readframes(wf.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0

        print(f"  Playing {wav_path.name} ({len(audio)/rate:.1f}s) ...")
        sd.play(audio, samplerate=rate)
        sd.wait()
        print("  Done.")
    except ImportError:
        print("  Playback unavailable (sounddevice not installed).")
        print(f"  Open the file directly: {wav_path}")
    except Exception as exc:
        print(f"  Playback error: {exc}")
        print(f"  Open the file directly: {wav_path}")


# ── main ─────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate synthetic clinical cases for SCRIBE testing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  Generate all cases:           python -m tools.generate_synthetic_cases
  Generate one case:            python -m tools.generate_synthetic_cases --case chest_pain_consultation
  Generate + play:              python -m tools.generate_synthetic_cases --case chest_pain_consultation --play
  Telephone simulation:         python -m tools.generate_synthetic_cases --env telephone
  List scenarios:               python -m tools.generate_synthetic_cases --list
  Inspect a generated case:     python -m tools.generate_synthetic_cases --case chest_pain_consultation --show
  How it works:                 python -m tools.generate_synthetic_cases --explain
""",
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
        "--show", action="store_true",
        help="Show case folder contents and ground truth summary",
    )
    parser.add_argument(
        "--explain", action="store_true",
        help="Print step-by-step explanation of the generator workflow",
    )
    parser.add_argument(
        "--open", action="store_true",
        help="Open the case folder in the system file browser",
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

    # ── explain ──────────────────────────────────────────────
    if args.explain:
        print(_EXPLAIN_TEXT)
        return 0

    # ── list ─────────────────────────────────────────────────
    if args.list:
        print("\nAvailable scenarios:\n")
        for cid in list_scenarios():
            s = SCENARIOS[cid]
            n_turns = len(s.get("dialogue", []))
            n_symp = len(s.get("ground_truth", {}).get("symptoms", []))
            print(f"  {cid:<35} {s['encounter_type']:<25} "
                  f"theme={s['theme']:<15} {n_turns} turns, {n_symp} symptoms")
        print(f"\nGenerate:  python -m tools.generate_synthetic_cases --case <case_id>")
        return 0

    # ── show ─────────────────────────────────────────────────
    if args.show:
        if not args.case:
            print("ERROR: --show requires --case <case_id>", file=sys.stderr)
            return 1
        return _show_case(args.case, args.output_dir)

    # ── open ─────────────────────────────────────────────────
    if args.open:
        if not args.case:
            print("ERROR: --open requires --case <case_id>", file=sys.stderr)
            return 1
        case_dir = Path(args.output_dir) / args.case
        if not case_dir.exists():
            print(f"Case not found: {case_dir}", file=sys.stderr)
            print(f"Run generation first:", file=sys.stderr)
            print(f"  python -m tools.generate_synthetic_cases --case {args.case}",
                  file=sys.stderr)
            return 1
        _open_folder(case_dir)
        return 0

    # ── build config ─────────────────────────────────────────
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

    # ── select scenarios ─────────────────────────────────────
    if args.case:
        if args.case not in SCENARIOS:
            print(f"ERROR: Unknown case '{args.case}'", file=sys.stderr)
            print(f"Available: {', '.join(list_scenarios())}", file=sys.stderr)
            return 1
        cases = [args.case]
    else:
        cases = list_scenarios()

    # ── generate ─────────────────────────────────────────────
    generated: list[tuple[str, Path]] = []

    for case_id in cases:
        scenario = SCENARIOS[case_id]
        print(f"Generating: {case_id} ...", end=" ", flush=True)
        out_dir = generate_case(scenario, config)
        print("done.")
        generated.append((case_id, out_dir))

    # ── output ───────────────────────────────────────────────
    if len(generated) == 1:
        case_id, out_dir = generated[0]
        _print_case_summary(case_id, SCENARIOS[case_id], out_dir, args.env)
    else:
        _print_batch_summary(generated, config.output_dir, args.env)

    # ── play ─────────────────────────────────────────────────
    if args.play:
        for case_id, out_dir in generated:
            wav_path = out_dir / "audio.wav"
            if wav_path.exists():
                print(f"\n  [{case_id}]")
                _play_audio(wav_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
