"""scribe-local — privacy-first streaming ASR pipeline.

Run with:
    python -m app.main --config config.yaml

Pipeline:
    Audio → VAD → Diarization → ASR → RAW commit → Normalization → Output
"""

from __future__ import annotations

import json
import os
import queue
import select
import signal
import sys
import threading
import time
import wave
from datetime import datetime
from pathlib import Path

import numpy as np

from app.config import (
    AppConfig,
    apply_cli_overrides,
    build_arg_parser,
    load_config,
)
from app.audio import AudioCapture, list_devices
from app.vad import VoiceActivityDetector, VadResult
from app.diarization import (
    apply_merge_map,
    create_diarizer,
    Diarizer,
    load_merge_map,
    relabel_segments,
    resolve_merge_chains,
    run_pyannote_diarization,
    save_merge_map,
    smooth_turns,
)
from app.asr import ASREngine
from app.commit import SegmentCommitter, RawSegment, _fmt_ts
from app.normalize import Normalizer
from app.io import OutputWriter


# ── pipeline processing ──────────────────────────────────────────────

def process_speech_buffer(
    speech_buffer: list[np.ndarray],
    buffer_start_sample: int,
    sample_rate: int,
    diarizer: Diarizer,
    asr_engine: ASREngine,
    committer: SegmentCommitter,
    normalizer: Normalizer,
    writer: OutputWriter,
    is_paragraph: bool,
) -> None:
    """Transcribe a speech buffer and commit all resulting segments."""
    audio = np.concatenate(speech_buffer)
    chunk_offset_sec = buffer_start_sample / sample_rate

    # 1. Diarization — who is speaking?
    speaker_id = diarizer.identify_speaker(audio, sample_rate)

    # 2. ASR — what did they say?
    asr_results = asr_engine.transcribe(audio)

    for result in asr_results:
        t0 = chunk_offset_sec + result.start
        t1 = chunk_offset_sec + result.end

        # 3. Commit immutable RAW segment
        raw_seg = committer.commit(t0, t1, speaker_id, result.text)
        writer.append_raw(raw_seg)

        # 4. Normalize
        norm_seg, changes = normalizer.normalize(raw_seg)
        writer.add_normalized(norm_seg, changes)

        # 5. Real-time display
        _display_segment(norm_seg.to_txt_line(), changes)

    if is_paragraph:
        committer.new_paragraph()


def _display_segment(line: str, changes: list) -> None:
    """Print a segment to the console."""
    suffix = f"  ({len(changes)} corrections)" if changes else ""
    print(f"  {line}{suffix}")


def _write_session_wav(
    chunks: list[np.ndarray],
    sample_rate: int,
    output_dir: str,
) -> Path | None:
    """Write all captured audio chunks to a single 16-bit PCM WAV file.

    Returns the path on success, None if there is no audio to write.
    """
    if not chunks:
        return None

    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out = Path(output_dir) / f"audio_{ts}.wav"
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"[{_ts()}] Writing session WAV...")
    audio = np.concatenate(chunks)
    pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)

    with wave.open(str(out), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())

    print(f"[{_ts()}] Session WAV written.")
    return out


# ── cross-platform quit listener ─────────────────────────────────────

def _start_quit_listener(stop_event: threading.Event) -> None:
    """Start a daemon thread that watches for 'q' to quit.

    Windows: uses msvcrt for non-blocking keypress detection.
    POSIX:   uses select() on sys.stdin for non-blocking read.
    """
    if os.name == "nt":
        try:
            import msvcrt
        except ImportError:
            return

        def _poll() -> None:
            while not stop_event.is_set():
                if msvcrt.kbhit():
                    ch = msvcrt.getwch()
                    if ch.lower() == "q":
                        print('\n\nStopping (pressed "q")...')
                        stop_event.set()
                        return
                time.sleep(0.05)

    else:
        # POSIX: select-based stdin listener
        def _poll() -> None:
            while not stop_event.is_set():
                try:
                    ready, _, _ = select.select([sys.stdin], [], [], 0.1)
                except (ValueError, OSError):
                    # stdin closed or not selectable
                    return
                if ready:
                    try:
                        line = sys.stdin.readline()
                    except (ValueError, OSError):
                        return
                    if line.strip().lower() == "q":
                        print('\n\nStopping (typed "q")...')
                        stop_event.set()
                        return

    t = threading.Thread(target=_poll, daemon=True)
    t.start()


def _ts() -> str:
    """Compact timestamp for shutdown instrumentation."""
    return time.strftime("%H:%M:%S")


# ── main loop ────────────────────────────────────────────────────────

def run(config: AppConfig, args: object = None) -> None:
    """Run the full streaming pipeline until stopped."""
    sample_rate = config.audio.sample_rate
    short_silence_samples = int(config.vad.short_silence_sec * sample_rate)
    long_silence_samples = int(config.vad.long_silence_sec * sample_rate)
    min_speech_samples = int(config.vad.min_speech_sec * sample_rate)

    # ── stop event + signal handler ─────────────────────────────
    stop_event = threading.Event()

    def _sigint_handler(signum: int, frame: object) -> None:
        print("\n\nStopping (Ctrl+C)...")
        stop_event.set()

    signal.signal(signal.SIGINT, _sigint_handler)

    # ── init components ──────────────────────────────────────────
    print(f"Language        : {config.language}")
    print(f"ASR model       : {config.asr.model}")
    print(f"Compute device  : {config.asr.device}")
    print(f"Output dir      : {config.output_dir}")
    print()
    print("Loading ASR model...")

    asr_engine = ASREngine(config)
    print(f"ASR device      : {asr_engine.device}")

    vad = VoiceActivityDetector(config)
    diarizer = create_diarizer(config)
    committer = SegmentCommitter(asr_engine.model_name, config.language)
    normalizer = Normalizer(config)
    writer = OutputWriter(config.output_dir, asr_engine.model_name)

    audio = AudioCapture(config)

    # ── state ────────────────────────────────────────────────────
    speech_buffer: list[np.ndarray] = []
    session_audio: list[np.ndarray] = []
    buffer_start_sample: int = 0
    total_samples: int = 0
    silence_samples: int = 0
    currently_speaking: bool = False
    sentence_committed: bool = False

    # ── start ────────────────────────────────────────────────────
    audio.start()
    _start_quit_listener(stop_event)

    print()
    stop_hint = 'Press Ctrl+C to stop.'
    if os.name == "nt":
        stop_hint += ' Or press "q".'
    else:
        stop_hint += ' Or type "q" + Enter.'
    print(f"Recording... {stop_hint}")
    print("-" * 60)

    try:
        while not stop_event.is_set():
            # Get next audio chunk (short timeout keeps loop responsive)
            try:
                chunk = audio.get_chunk(timeout=0.1)
            except queue.Empty:
                continue

            session_audio.append(chunk)
            result = vad.process(chunk)
            total_samples += len(chunk)

            if result == VadResult.SPEECH:
                if not currently_speaking:
                    currently_speaking = True
                    silence_samples = 0
                    buffer_start_sample = total_samples - len(chunk)
                speech_buffer.append(chunk)
                silence_samples = 0
                sentence_committed = False

            else:  # SILENCE
                if currently_speaking:
                    silence_samples += len(chunk)
                    # Include trailing silence in buffer (aids ASR accuracy)
                    if not sentence_committed:
                        speech_buffer.append(chunk)

                    # Short silence → sentence commit
                    if (
                        silence_samples >= short_silence_samples
                        and not sentence_committed
                    ):
                        speech_samples = sum(len(c) for c in speech_buffer)
                        if speech_samples >= min_speech_samples:
                            process_speech_buffer(
                                speech_buffer,
                                buffer_start_sample,
                                sample_rate,
                                diarizer,
                                asr_engine,
                                committer,
                                normalizer,
                                writer,
                                is_paragraph=False,
                            )
                        speech_buffer = []
                        sentence_committed = True

                    # Long silence → paragraph break
                    if silence_samples >= long_silence_samples:
                        committer.new_paragraph()
                        currently_speaking = False
                        silence_samples = 0
                        sentence_committed = False

    except KeyboardInterrupt:
        # Belt-and-suspenders: in case signal handler didn't fire
        print("\n\nStopping...")

    finally:
        # ── bounded-time shutdown ────────────────────────────────
        print(f"[{_ts()}] Flushing speech buffer...")

        # Flush remaining speech buffer
        if speech_buffer:
            speech_samples = sum(len(c) for c in speech_buffer)
            if speech_samples >= min_speech_samples:
                process_speech_buffer(
                    speech_buffer,
                    buffer_start_sample,
                    sample_rate,
                    diarizer,
                    asr_engine,
                    committer,
                    normalizer,
                    writer,
                    is_paragraph=True,
                )

        # Shut down audio stream (bounded timeout — won't hang)
        print(f"[{_ts()}] Stopping audio stream...")
        audio.stop(timeout=3.0)
        print(f"[{_ts()}] Audio stream stopped.")

        # Write session-end files (bounded timeout)
        print(f"[{_ts()}] Writing output files...")
        writer.finalize(timeout=5.0)
        print(f"[{_ts()}] Output files written.")

        # Write session WAV
        wav_path = _write_session_wav(session_audio, sample_rate, config.output_dir)

        # Post-session pyannote diarization
        diar_path = None
        diarized_json = None
        diarized_txt = None
        if wav_path and config.diarization.backend == "pyannote":
            print(f"[{_ts()}] Running pyannote diarization...")
            try:
                diar_path = run_pyannote_diarization(
                    wav_path, config.output_dir
                )
                print(f"[{_ts()}] Diarization complete.")
            except Exception as e:
                print(f"[{_ts()}] Diarization failed: {e}")

        # Smooth diarization turns (reduce backchannel flips)
        if diar_path and config.diarization.smoothing:
            print(f"[{_ts()}] Smoothing diarization turns...")
            with open(diar_path, encoding="utf-8") as f:
                diar_data = json.load(f)
            original_count = len(diar_data["turns"])
            diar_data["turns"] = smooth_turns(
                diar_data["turns"],
                config.diarization.min_turn_sec,
                config.diarization.gap_merge_sec,
            )
            with open(diar_path, "w", encoding="utf-8") as f:
                json.dump(diar_data, f, ensure_ascii=False, indent=2)
            print(f"[{_ts()}] Smoothed: {original_count} → {len(diar_data['turns'])} turns.")

        # Apply calibration profile (override speaker IDs from embeddings)
        calibrated_path = None
        if diar_path and config.diarization.calibration_profile:
            from app.calibration import (
                embed_turns, load_profile,
                build_cluster_embeddings, assign_clusters_to_profile,
                apply_cluster_override,
            )
            profile_path = os.path.join(
                "profiles",
                f"{config.diarization.calibration_profile}.json",
            )
            try:
                profile = load_profile(profile_path)
                with open(diar_path, encoding="utf-8") as f:
                    diar_data = json.load(f)
                print(f"[{_ts()}] Extracting per-turn embeddings...")
                embed_turns(diar_data["turns"], wav_path)
                cluster_embs = build_cluster_embeddings(diar_data["turns"])
                mapping = assign_clusters_to_profile(
                    cluster_embs,
                    profile,
                    config.diarization.calibration_similarity_threshold,
                    config.diarization.calibration_similarity_margin,
                )
                diar_data["turns"] = apply_cluster_override(
                    diar_data["turns"], mapping
                )
                # Strip embeddings before writing
                for t in diar_data["turns"]:
                    t.pop("embedding", None)
                calibrated_path = diar_path.with_suffix(".calibrated.json")
                with open(calibrated_path, "w", encoding="utf-8") as f:
                    json.dump(diar_data, f, ensure_ascii=False, indent=2)
                print(f"[{_ts()}] Applied calibration profile: {config.diarization.calibration_profile}")
            except FileNotFoundError as e:
                print(f"[{_ts()}] Calibration profile not found: {e}")
            except Exception as e:
                print(f"[{_ts()}] Calibration failed: {e}")

        # Prefer calibrated diarization if available, otherwise original
        active_diar_path = (
            calibrated_path
            if calibrated_path and calibrated_path.exists()
            else diar_path
        )

        # Apply speaker merge map (if exists)
        if active_diar_path:
            diar_ts = diar_path.stem.removeprefix("diarization_")
            merge_map = load_merge_map(config.output_dir, diar_ts)
            if merge_map:
                print(f"[{_ts()}] Applying speaker merge map...")
                try:
                    resolved = resolve_merge_chains(merge_map)
                    with open(active_diar_path, encoding="utf-8") as f:
                        diar_data = json.load(f)
                    original_count = len(diar_data["turns"])
                    diar_data["turns"] = apply_merge_map(
                        diar_data["turns"], resolved
                    )
                    with open(active_diar_path, "w", encoding="utf-8") as f:
                        json.dump(diar_data, f, ensure_ascii=False, indent=2)
                    print(f"[{_ts()}] Merged: {original_count} → {len(diar_data['turns'])} turns.")
                except ValueError as e:
                    print(f"[{_ts()}] Speaker merge failed: {e}")

        # Relabel segments with diarization speaker_ids
        if active_diar_path:
            print(f"[{_ts()}] Relabeling segments...")
            try:
                diarized_json, diarized_txt = relabel_segments(
                    writer.normalized_json_path, active_diar_path, config.output_dir
                )
                print(f"[{_ts()}] Segment relabeling complete.")
            except Exception as e:
                print(f"[{_ts()}] Segment relabeling failed: {e}")

        # Speaker tagging (after relabeling)
        tagged_txt = None
        if diarized_txt:
            from app.tagging import (
                load_or_create_tags, apply_auto_tags, save_tags,
                generate_tag_labeled_txt,
            )
            diar_ts = diarized_txt.stem.removeprefix("diarized_")
            tags = load_or_create_tags(config.output_dir, diar_ts)

            auto_tags = getattr(args, "auto_tags", "none") or "none"
            if auto_tags != "none" and diarized_json:
                speakers = sorted(set(
                    s["new_speaker_id"]
                    for s in json.loads(diarized_json.read_text("utf-8"))
                ))
                apply_auto_tags(tags, auto_tags, speakers)

            save_tags(tags, config.output_dir, diar_ts)
            tagged_txt = generate_tag_labeled_txt(
                diarized_txt, tags, config.output_dir, diar_ts
            )

        print("-" * 60)
        print(f"Segments committed : {committer.seg_count}")
        print(f"RAW output         : {writer.raw_json_path}")
        print(f"Normalized output  : {writer.normalized_json_path}")
        print(f"Change log         : {writer.changes_json_path}")
        if wav_path:
            print(f"Session audio      : {wav_path}")
        if diar_path:
            print(f"Diarization        : {diar_path}")
        if diarized_json:
            print(f"Diarized segments  : {diarized_json}")
        if diarized_txt:
            print(f"Diarized transcript: {diarized_txt}")
        if tagged_txt:
            print(f"Tagged transcript  : {tagged_txt}")
        print("Session ended.")


# ── entry point ──────────────────────────────────────────────────────

def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    # --list-audio-devices: print and exit
    if args.list_audio_devices:
        list_devices()
        sys.exit(0)

    # --create-profile: record voice samples and exit
    if args.create_profile:
        # Load config early for audio settings
        config_path = Path(args.config)
        if config_path.exists():
            config = load_config(str(config_path))
        elif args.config != "config.yaml":
            print(f"Error: config file not found: {args.config}")
            sys.exit(1)
        else:
            config = AppConfig()
        config = apply_cli_overrides(config, args)

        from app.calibration import record_and_build_profile, save_profile
        profile_path = os.path.join("profiles", f"{args.create_profile}.json")
        if os.path.exists(profile_path):
            print(f"Error: profile already exists: {profile_path}")
            print("Delete it manually or choose a different name.")
            sys.exit(1)
        profile = record_and_build_profile(
            config, args.profile_speakers, args.profile_duration,
        )
        path = save_profile(profile_path, profile)
        print(f"Profile saved: {path}")
        sys.exit(0)

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        config = load_config(str(config_path))
    elif args.config != "config.yaml":
        print(f"Error: config file not found: {args.config}")
        sys.exit(1)
    else:
        config = AppConfig()

    config = apply_cli_overrides(config, args)

    # Validate language
    if config.language not in ("da", "sv", "en"):
        print(f"Error: unsupported language '{config.language}'. Use da, sv, or en.")
        sys.exit(1)

    # Standalone session mode (tagging + merge)
    if args.session:
        from app.tagging import (
            load_or_create_tags, apply_auto_tags, set_tag, set_label,
            save_tags, generate_tag_labeled_txt,
        )
        ts = args.session

        # ── speaker merge ─────────────────────────────────────────
        diar_path = Path(config.output_dir) / f"diarization_{ts}.json"
        if args.merge:
            merge_map = load_merge_map(config.output_dir, ts)
            for entry in args.merge:
                spk, target = entry.split("=", 1)
                merge_map[spk] = target
            try:
                merge_map = resolve_merge_chains(merge_map)
            except ValueError as e:
                print(f"Error: {e}")
                sys.exit(1)
            merge_path = save_merge_map(merge_map, config.output_dir, ts)
            print(f"Speaker merges   : {merge_path}")

            if diar_path.exists():
                with open(diar_path, encoding="utf-8") as f:
                    diar_data = json.load(f)
                diar_data["turns"] = apply_merge_map(
                    diar_data["turns"], merge_map
                )
                with open(diar_path, "w", encoding="utf-8") as f:
                    json.dump(diar_data, f, ensure_ascii=False, indent=2)

                # Find normalized JSON to re-run relabeling
                norm_files = list(Path(config.output_dir).glob(
                    f"normalized_*_{ts.split('_')[0]}*.json"
                ))
                if not norm_files:
                    norm_files = list(Path(config.output_dir).glob(
                        "normalized_*.json"
                    ))
                if norm_files:
                    relabel_segments(
                        norm_files[-1], diar_path, config.output_dir
                    )

        # ── speaker tags ──────────────────────────────────────────
        tags = load_or_create_tags(config.output_dir, ts)

        for entry in args.set_tag:
            spk, val = entry.split("=", 1)
            set_tag(tags, spk, val)
        for entry in args.set_label:
            spk, val = entry.split("=", 1)
            set_label(tags, spk, val)

        if args.auto_tags != "none":
            diarized_seg_path = Path(config.output_dir) / f"diarized_segments_{ts}.json"
            if diarized_seg_path.exists():
                speakers = sorted(set(
                    s["new_speaker_id"]
                    for s in json.loads(
                        diarized_seg_path.read_text(encoding="utf-8")
                    )
                ))
                apply_auto_tags(tags, args.auto_tags, speakers)

        tags_path = save_tags(tags, config.output_dir, ts)
        print(f"Speaker tags     : {tags_path}")

        diarized_txt = Path(config.output_dir) / f"diarized_{ts}.txt"
        if diarized_txt.exists():
            labeled = generate_tag_labeled_txt(
                diarized_txt, tags, config.output_dir, ts
            )
            print(f"Tagged transcript: {labeled}")

        sys.exit(0)

    run(config, args)


if __name__ == "__main__":
    main()
