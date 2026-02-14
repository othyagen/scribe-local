"""scribe-local — privacy-first streaming ASR pipeline.

Run with:
    python -m app.main --config config.yaml

Pipeline:
    Audio → VAD → Diarization → ASR → RAW commit → Normalization → Output
"""

from __future__ import annotations

import os
import queue
import select
import signal
import sys
import threading
import time
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
from app.diarization import create_diarizer, Diarizer
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

def run(config: AppConfig) -> None:
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

        print("-" * 60)
        print(f"Segments committed : {committer.seg_count}")
        print(f"RAW output         : {writer.raw_json_path}")
        print(f"Normalized output  : {writer.normalized_json_path}")
        print(f"Change log         : {writer.changes_json_path}")
        print("Session ended.")


# ── entry point ──────────────────────────────────────────────────────

def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    # --list-audio-devices: print and exit
    if args.list_audio_devices:
        list_devices()
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

    run(config)


if __name__ == "__main__":
    main()
