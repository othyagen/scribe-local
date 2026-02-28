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


# ── resume support ───────────────────────────────────────────────────

class ResumeError(ValueError):
    """Raised when session resume validation fails."""


def load_resume_state(
    output_dir: str,
    session_ts: str,
    expected_model: str,
) -> dict:
    """Validate and load state from an existing session for resume.

    Returns a dict with model_tag, last_t1, last_seg_num, last_para_num,
    and wav_parts.  Raises ResumeError on any validation failure.
    """
    out = Path(output_dir)

    # Find raw JSONL file matching session_ts
    raw_candidates = list(out.glob(f"raw_{session_ts}_*.json"))
    if not raw_candidates:
        raise ResumeError(
            f"no RAW file found for session {session_ts} in {output_dir}"
        )
    if len(raw_candidates) > 1:
        raise ResumeError(
            f"multiple RAW files for session {session_ts}: {raw_candidates}"
        )

    raw_path = raw_candidates[0]

    # Extract model_tag from filename: raw_<ts>_<model_tag>.json
    prefix = f"raw_{session_ts}_"
    model_tag = raw_path.stem[len(prefix):]

    # Verify model matches
    expected_tag = expected_model.replace("/", "-")
    if model_tag != expected_tag:
        raise ResumeError(
            f"model mismatch — session used '{model_tag}' "
            f"but current config specifies '{expected_tag}'"
        )

    # Parse all segments from JSONL
    segments: list[dict] = []
    with open(raw_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                segments.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ResumeError(
                    f"corrupt JSONL at line {line_num} in {raw_path}: {e}"
                )

    if not segments:
        raise ResumeError(f"RAW file has no segments: {raw_path}")

    # Validate timestamps
    last_seg = segments[-1]
    last_t1 = last_seg["t1"]
    if last_t1 <= 0:
        raise ResumeError(f"last segment t1={last_t1} is not positive")

    prev_t0 = 0.0
    prev_t1 = 0.0
    for seg in segments:
        t0, t1 = seg["t0"], seg["t1"]
        if t0 > t1:
            raise ResumeError(
                f"segment {seg['seg_id']} has t0={t0} > t1={t1}"
            )
        if t0 < prev_t0:
            raise ResumeError(
                f"non-monotonic t0 — segment {seg['seg_id']} "
                f"t0={t0} < previous t0={prev_t0}"
            )
        if t1 < prev_t1:
            raise ResumeError(
                f"non-monotonic t1 — segment {seg['seg_id']} "
                f"t1={t1} < previous t1={prev_t1}"
            )
        prev_t0, prev_t1 = t0, t1

    # Extract counters from last segment
    last_seg_num = int(last_seg["seg_id"].split("_")[1])
    last_para_num = int(last_seg["paragraph_id"].split("_")[1])

    # Find existing WAV parts
    wav_parts: list[Path] = []
    original_wav = out / f"audio_{session_ts}.wav"
    if original_wav.exists():
        wav_parts.append(original_wav)
    wav_parts.extend(sorted(out.glob(f"audio_{session_ts}_part*.wav")))

    return {
        "model_tag": model_tag,
        "last_t1": last_t1,
        "last_seg_num": last_seg_num,
        "last_para_num": last_para_num,
        "wav_parts": wav_parts,
    }


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
    confidence_entries: list[dict] | None = None,
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

        # 6. Accumulate confidence metrics (if tracking)
        if confidence_entries is not None:
            confidence_entries.append({
                "seg_id": raw_seg.seg_id,
                "t0": raw_seg.t0,
                "t1": raw_seg.t1,
                "avg_logprob": result.avg_logprob,
                "no_speech_prob": result.no_speech_prob,
                "compression_ratio": result.compression_ratio,
            })

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
    filename_override: str | None = None,
) -> Path | None:
    """Write all captured audio chunks to a single 16-bit PCM WAV file.

    Returns the path on success, None if there is no audio to write.
    """
    if not chunks:
        return None

    if filename_override:
        out = Path(output_dir) / filename_override
    else:
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


def _concatenate_wavs(
    wav_paths: list[Path],
    sample_rate: int,
    output_dir: str,
    filename: str,
) -> Path:
    """Concatenate multiple WAV files into a single combined WAV.

    Does NOT delete the part files (safety: never destroy data).
    """
    out = Path(output_dir) / filename
    all_samples: list[np.ndarray] = []
    for wp in wav_paths:
        with wave.open(str(wp), "rb") as wf:
            pcm_bytes = wf.readframes(wf.getnframes())
            samples = np.frombuffer(pcm_bytes, dtype=np.int16)
            all_samples.append(samples)

    combined = np.concatenate(all_samples)
    with wave.open(str(out), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(combined.tobytes())

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

    # ── resume state (if applicable) ────────────────────────────
    resume_ts = getattr(args, "resume", None)
    resume_state = None
    if resume_ts:
        resume_state = load_resume_state(
            config.output_dir, resume_ts, asr_engine.model_name
        )
        print(f"Resuming session  : {resume_ts}")
        print(f"Last segment      : seg_{resume_state['last_seg_num']:04d} "
              f"(t1={resume_state['last_t1']:.3f}s)")

    vad = VoiceActivityDetector(config)
    diarizer = create_diarizer(config)
    normalizer = Normalizer(config)

    if resume_state:
        committer = SegmentCommitter(
            asr_engine.model_name, config.language,
            start_seg=resume_state["last_seg_num"],
            start_para=resume_state["last_para_num"],
        )
        writer = OutputWriter(
            config.output_dir, asr_engine.model_name,
            session_ts=resume_ts,
            model_tag=resume_state["model_tag"],
        )
    else:
        committer = SegmentCommitter(asr_engine.model_name, config.language)
        writer = OutputWriter(config.output_dir, asr_engine.model_name)

    audio = AudioCapture(config)

    # ── state ────────────────────────────────────────────────────
    speech_buffer: list[np.ndarray] = []
    session_audio: list[np.ndarray] = []
    buffer_start_sample: int = 0
    total_samples: int = (
        int(resume_state["last_t1"] * sample_rate)
        if resume_state else 0
    )
    silence_samples: int = 0
    currently_speaking: bool = False
    sentence_committed: bool = False
    confidence_entries: list[dict] = []

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
                                confidence_entries=confidence_entries,
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
                    confidence_entries=confidence_entries,
                )

        # Shut down audio stream (bounded timeout — won't hang)
        print(f"[{_ts()}] Stopping audio stream...")
        audio.stop(timeout=3.0)
        print(f"[{_ts()}] Audio stream stopped.")

        # Write session-end files (bounded timeout)
        print(f"[{_ts()}] Writing output files...")
        if resume_state:
            writer.finalize_from_raw(normalizer, timeout=5.0)
        else:
            writer.finalize(timeout=5.0)
        print(f"[{_ts()}] Output files written.")

        # Write session WAV
        if resume_state:
            part_num = len(resume_state["wav_parts"]) + 1
            wav_path = _write_session_wav(
                session_audio, sample_rate, config.output_dir,
                filename_override=f"audio_{resume_ts}_part{part_num}.wav",
            )
            # Concatenate all parts into a single WAV for pyannote
            all_wav_parts = list(resume_state["wav_parts"])
            if wav_path:
                all_wav_parts.append(wav_path)
            if len(all_wav_parts) > 1:
                wav_path = _concatenate_wavs(
                    all_wav_parts, sample_rate, config.output_dir,
                    f"audio_{resume_ts}.wav",
                )
            elif all_wav_parts:
                wav_path = all_wav_parts[0]
            else:
                wav_path = None
        else:
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
        turns_before_smoothing = None
        turns_after_smoothing = None
        if diar_path and config.diarization.smoothing:
            print(f"[{_ts()}] Smoothing diarization turns...")
            with open(diar_path, encoding="utf-8") as f:
                diar_data = json.load(f)
            turns_before_smoothing = len(diar_data["turns"])
            diar_data["turns"] = smooth_turns(
                diar_data["turns"],
                config.diarization.min_turn_sec,
                config.diarization.gap_merge_sec,
            )
            turns_after_smoothing = len(diar_data["turns"])
            with open(diar_path, "w", encoding="utf-8") as f:
                json.dump(diar_data, f, ensure_ascii=False, indent=2)
            print(f"[{_ts()}] Smoothed: {turns_before_smoothing} → {turns_after_smoothing} turns.")

        # Apply calibration profile (override speaker IDs from embeddings)
        calibrated_path = None
        cal_report_path = None
        cal_stats: dict | None = None
        cal_enabled = (
            diar_path
            and config.diarization.calibration_profile
            and config.diarization.calibration_enabled
        )
        if cal_enabled:
            from app.calibration import (
                embed_turns, load_profile,
                detect_and_mark_overlap, MIN_PROTOTYPE_DURATION_SEC,
                build_cluster_embeddings_with_stats,
                filter_eligible_clusters,
                assign_clusters_to_profile,
                apply_cluster_override,
                build_calibration_report, print_calibration_debug,
            )
            profile_path = os.path.join(
                "profiles",
                f"{config.diarization.calibration_profile}.json",
            )
            try:
                profile = load_profile(profile_path)
                with open(diar_path, encoding="utf-8") as f:
                    diar_data = json.load(f)

                # Overlap detection — gated
                if config.diarization.overlap_stabilizer_enabled:
                    diar_data["turns"] = detect_and_mark_overlap(diar_data["turns"])

                # Embedding min duration — gated
                if config.diarization.min_duration_filter_enabled:
                    effective_min = max(
                        config.diarization.calibration_min_turn_duration_sec,
                        MIN_PROTOTYPE_DURATION_SEC,
                    )
                else:
                    effective_min = config.diarization.calibration_min_turn_duration_sec

                print(f"[{_ts()}] Extracting per-turn embeddings...")
                embed_turns(
                    diar_data["turns"], wav_path,
                    min_duration_sec=effective_min,
                )

                proto_min = (
                    MIN_PROTOTYPE_DURATION_SEC
                    if config.diarization.min_duration_filter_enabled
                    else 0.0
                )
                cluster_embs, cluster_stats = build_cluster_embeddings_with_stats(
                    diar_data["turns"],
                    min_duration_sec=proto_min,
                )
                eligible_embs, ineligible_reasons = filter_eligible_clusters(
                    cluster_embs, cluster_stats,
                    min_cluster_turns=config.diarization.calibration_min_cluster_turns,
                    min_cluster_voiced_sec=config.diarization.calibration_min_cluster_voiced_sec,
                )

                # Collect calibration stats for session report
                overlap_count = sum(
                    1 for t in diar_data["turns"] if t.get("overlap")
                )
                emb_count = sum(
                    1 for t in diar_data["turns"] if "embedding" in t
                )
                eligible_ids = set(eligible_embs.keys())

                # Prototype matching — gated
                if config.diarization.prototype_matching_enabled:
                    mapping = assign_clusters_to_profile(
                        eligible_embs,
                        profile,
                        config.diarization.calibration_similarity_threshold,
                        config.diarization.calibration_similarity_margin,
                    )
                    allow_partial = config.diarization.calibration_allow_partial_assignment
                    partial_applied = None
                    if not allow_partial and eligible_ids:
                        unassigned = eligible_ids - set(mapping.keys())
                        partial_applied = len(unassigned) == 0
                else:
                    mapping = {}
                    allow_partial = True
                    partial_applied = None

                # Diagnostics report (always written)
                cal_report = build_calibration_report(
                    cluster_embs, profile,
                    config.diarization.calibration_similarity_threshold,
                    config.diarization.calibration_similarity_margin,
                    mapping, config.diarization.calibration_profile,
                    cluster_stats=cluster_stats,
                    ineligible_reasons=ineligible_reasons,
                    partial_assignment_applied=partial_applied,
                )
                if not config.diarization.prototype_matching_enabled:
                    cal_report["reason"] = "prototype_matching_disabled"
                if config.diarization.calibration_debug:
                    print_calibration_debug(cal_report)
                cal_report_path = diar_path.with_suffix(".calibration_report.json")
                with open(cal_report_path, "w", encoding="utf-8") as f:
                    json.dump(cal_report, f, ensure_ascii=False, indent=2)

                # Apply overrides only when prototype matching is enabled
                if config.diarization.prototype_matching_enabled:
                    diar_data["turns"] = apply_cluster_override(
                        diar_data["turns"], mapping,
                        eligible_cluster_ids=eligible_ids,
                        allow_partial=allow_partial,
                    )

                # Strip embeddings and overlap markers before writing
                for t in diar_data["turns"]:
                    t.pop("embedding", None)
                    t.pop("overlap", None)
                    t.pop("overlap_with", None)
                calibrated_path = diar_path.with_suffix(".calibrated.json")
                with open(calibrated_path, "w", encoding="utf-8") as f:
                    json.dump(diar_data, f, ensure_ascii=False, indent=2)
                print(f"[{_ts()}] Applied calibration profile: {config.diarization.calibration_profile}")

                # Stats for session report
                assigned_count = len(mapping)
                unknown_count = len(eligible_ids) - assigned_count
                cal_stats = {
                    "overlaps_marked": overlap_count,
                    "embeddings_computed": emb_count,
                    "clusters_total": len(cluster_embs),
                    "clusters_assigned": assigned_count,
                    "clusters_unknown": unknown_count,
                }
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

        # Confidence report
        confidence_report_path = None
        if confidence_entries:
            from app.confidence import build_confidence_report, write_confidence_report
            # raw filename: raw_YYYY-MM-DD_HH-MM-SS_<model>.json
            # timestamp is chars 4..23 of the stem
            session_ts = writer.raw_json_path.stem[4:23]
            report = build_confidence_report(confidence_entries)
            confidence_report_path = write_confidence_report(
                report, config.output_dir, session_ts
            )
            print(f"[{_ts()}] Confidence report: {report['flagged_count']}/{report['total_count']} flagged.")

        print("-" * 60)
        print(f"Segments committed : {committer.seg_count}")
        print(f"RAW output         : {writer.raw_json_path}")
        print(f"Normalized output  : {writer.normalized_json_path}")
        print(f"Change log         : {writer.changes_json_path}")
        if wav_path:
            print(f"Session audio      : {wav_path}")
        if diar_path:
            print(f"Diarization        : {diar_path}")
        if cal_report_path:
            print(f"Calibration report : {cal_report_path}")
        if diarized_json:
            print(f"Diarized segments  : {diarized_json}")
        if diarized_txt:
            print(f"Diarized transcript: {diarized_txt}")
        if tagged_txt:
            print(f"Tagged transcript  : {tagged_txt}")
        if confidence_report_path:
            print(f"Confidence report  : {confidence_report_path}")

        # Session report (consolidated summary)
        session_report_path = None
        if config.reporting.session_report_enabled:
            from app.reporting import build_session_report, write_session_report
            session_ts = writer.raw_json_path.stem[4:23]
            output_paths = {
                "raw": str(writer.raw_json_path),
                "normalized": str(writer.normalized_json_path),
                "changes": str(writer.changes_json_path),
                "audio": str(wav_path) if wav_path else None,
                "diarization": str(diar_path) if diar_path else None,
                "calibration_report": str(cal_report_path) if cal_report_path else None,
                "calibrated": str(calibrated_path) if calibrated_path else None,
                "diarized_segments": str(diarized_json) if diarized_json else None,
                "diarized_txt": str(diarized_txt) if diarized_txt else None,
                "tagged_txt": str(tagged_txt) if tagged_txt else None,
                "confidence_report": str(confidence_report_path) if confidence_report_path else None,
            }
            diar_stats = {
                "turns_before_smoothing": turns_before_smoothing,
                "turns_after_smoothing": turns_after_smoothing,
            }
            sr = build_session_report(
                session_ts=session_ts,
                config=config,
                segment_count=committer.seg_count,
                output_paths=output_paths,
                diarization_stats=diar_stats,
                calibration_stats=cal_stats,
            )
            session_report_path = write_session_report(
                sr, config.output_dir, session_ts
            )
            print(f"Session report     : {session_report_path}")

        print("Session ended.")


# ── session browser formatting ────────────────────────────────────────

def _fmt_duration(seconds: float) -> str:
    """Format seconds as mm:ss."""
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def _print_session_list(sessions: list) -> None:
    """Print a formatted table of sessions."""
    if not sessions:
        print("No sessions found.")
        return

    # Header
    print(
        f"{'Timestamp':<21} {'Duration':>8} {'Segs':>5} "
        f"{'Model':<12} {'Lang':>4} {'Audio':>8} "
        f"{'Diar':>4} {'Conf':>12} {'Resume':>6}"
    )
    print("-" * 90)

    for s in sessions:
        dur = _fmt_duration(s.duration_sec)
        audio = f"{s.audio_parts_count} part" if s.audio_parts_count == 1 else (
            f"{s.audio_parts_count} parts" if s.audio_parts_count > 0 else "NO"
        )
        diar = "YES" if s.has_diarization else "NO"
        if s.has_confidence and s.confidence_flagged_count is not None:
            conf = f"{s.confidence_flagged_count} flagged"
        elif s.has_confidence:
            conf = "YES"
        else:
            conf = "-"
        resume = "YES" if s.resume_possible else "NO"

        print(
            f"{s.ts:<21} {dur:>8} {s.segment_count:>5} "
            f"{s.model_tag:<12} {s.language:>4} {audio:>8} "
            f"{diar:>4} {conf:>12} {resume:>6}"
        )


def _print_session_detail(info: dict) -> None:
    """Print detailed session info."""
    print(f"Session: {info['ts']}")
    print("-" * 50)
    print(f"  Model         : {info['model_tag']}")
    print(f"  Language      : {info['language']}")
    print(f"  Segments      : {info['segment_count']}")
    print(f"  Duration      : {_fmt_duration(info['duration_sec'])} ({info['duration_sec']:.1f}s)")
    print(f"  Audio         : {'YES' if info['has_audio'] else 'NO'} ({info['audio_parts_count']} parts)")
    print(f"  Normalized    : {'YES' if info['has_normalized'] else 'NO'}")
    print(f"  Diarization   : {'YES' if info['has_diarization'] else 'NO'}")
    if info.get("speaker_count") is not None:
        print(f"  Speakers      : {info['speaker_count']}")
    print(f"  Speaker tags  : {'YES' if info['has_tags'] else 'NO'}")
    if info['has_confidence']:
        flagged = info.get('confidence_flagged_count')
        total = info.get('confidence_total', '?')
        print(f"  Confidence    : {flagged}/{total} flagged")
    else:
        print(f"  Confidence    : NO")
    print(f"  Resume        : {'YES' if info['resume_possible'] else 'NO'}")
    if info.get("resume_reason"):
        print(f"                  ({info['resume_reason']})")
    print()
    print("Files:")
    for label, path in info.get("files", {}).items():
        print(f"  {label:<20}: {path}")


# ── entry point ──────────────────────────────────────────────────────

def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    # --list-audio-devices: print and exit
    if args.list_audio_devices:
        list_devices()
        sys.exit(0)

    # --list-sessions / --show-session: session browser (early exit)
    if args.list_sessions or args.show_session:
        # Mutual exclusion
        conflicts = []
        if args.list_sessions and args.show_session:
            conflicts.append("--list-sessions and --show-session")
        if getattr(args, "resume", None):
            conflicts.append("--resume")
        if args.session:
            conflicts.append("--session")
        if conflicts:
            print(f"Error: {' and '.join(conflicts)} cannot be used together")
            sys.exit(1)

        # Load config for output_dir
        config_path = Path(args.config)
        if config_path.exists():
            config = load_config(str(config_path))
        elif args.config != "config.yaml":
            print(f"Error: config file not found: {args.config}")
            sys.exit(1)
        else:
            config = AppConfig()
        config = apply_cli_overrides(config, args)

        from app.session_browser import scan_sessions, show_session
        out_dir = Path(config.output_dir)

        if args.list_sessions:
            _print_session_list(scan_sessions(out_dir))
        else:
            try:
                info = show_session(out_dir, args.show_session)
            except ValueError as e:
                print(f"Error: {e}")
                sys.exit(1)
            _print_session_detail(info)
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

    # Mutual exclusion: --resume and --session
    if getattr(args, "resume", None) and args.session:
        print("Error: --resume and --session are mutually exclusive")
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

    try:
        run(config, args)
    except ResumeError as e:
        print(f"Error: resume failed — {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
