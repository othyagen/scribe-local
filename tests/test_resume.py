"""Tests for session resume / append mode."""

from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from app.commit import RawSegment, SegmentCommitter
from app.config import build_arg_parser
from app.io import OutputWriter
from app.main import (
    ResumeError,
    _concatenate_wavs,
    _write_session_wav,
    load_resume_state,
)


# ── helpers ──────────────────────────────────────────────────────────

def _seg_dict(
    seg_id: str = "seg_0001",
    t0: float = 0.0,
    t1: float = 1.0,
    speaker_id: str = "spk_0",
    raw_text: str = "hello",
    model_name: str = "large-v3",
    language: str = "en",
    paragraph_id: str = "para_0000",
) -> dict:
    return {
        "seg_id": seg_id, "t0": t0, "t1": t1,
        "speaker_id": speaker_id, "raw_text": raw_text,
        "model_name": model_name, "language": language,
        "paragraph_id": paragraph_id,
    }


def _write_raw_jsonl(tmp_path: Path, segments: list[dict],
                     ts: str = "2026-01-01_12-00-00",
                     model_tag: str = "large-v3") -> Path:
    p = tmp_path / f"raw_{ts}_{model_tag}.json"
    with open(p, "w", encoding="utf-8") as f:
        for seg in segments:
            f.write(json.dumps(seg, ensure_ascii=False) + "\n")
    return p


def _write_wav(path: Path, n_samples: int = 16000, sample_rate: int = 16000) -> Path:
    """Write a valid WAV file with silence."""
    pcm = np.zeros(n_samples, dtype=np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())
    return path


# ── load_resume_state — validation ───────────────────────────────────

class TestLoadResumeStateValid:
    def test_returns_correct_fields(self, tmp_path):
        segs = [
            _seg_dict("seg_0001", 0.0, 1.5, paragraph_id="para_0000"),
            _seg_dict("seg_0002", 2.0, 3.5, paragraph_id="para_0000"),
            _seg_dict("seg_0003", 5.0, 7.0, paragraph_id="para_0001"),
        ]
        _write_raw_jsonl(tmp_path, segs)

        state = load_resume_state(str(tmp_path), "2026-01-01_12-00-00", "large-v3")
        assert state["model_tag"] == "large-v3"
        assert state["last_t1"] == 7.0
        assert state["last_seg_num"] == 3
        assert state["last_para_num"] == 1
        assert state["wav_parts"] == []

    def test_finds_existing_wav_parts(self, tmp_path):
        _write_raw_jsonl(tmp_path, [_seg_dict()])
        _write_wav(tmp_path / "audio_2026-01-01_12-00-00.wav")
        _write_wav(tmp_path / "audio_2026-01-01_12-00-00_part2.wav")

        state = load_resume_state(str(tmp_path), "2026-01-01_12-00-00", "large-v3")
        assert len(state["wav_parts"]) == 2


class TestLoadResumeStateErrors:
    def test_missing_file(self, tmp_path):
        with pytest.raises(ResumeError, match="no RAW file found"):
            load_resume_state(str(tmp_path), "2026-99-99_00-00-00", "large-v3")

    def test_empty_file(self, tmp_path):
        _write_raw_jsonl(tmp_path, [])
        with pytest.raises(ResumeError, match="no segments"):
            load_resume_state(str(tmp_path), "2026-01-01_12-00-00", "large-v3")

    def test_model_mismatch(self, tmp_path):
        _write_raw_jsonl(tmp_path, [_seg_dict()])
        with pytest.raises(ResumeError, match="model mismatch"):
            load_resume_state(str(tmp_path), "2026-01-01_12-00-00", "small")

    def test_non_monotonic_t1(self, tmp_path):
        segs = [
            _seg_dict("seg_0001", 0.0, 3.0),
            _seg_dict("seg_0002", 1.0, 2.0),
        ]
        _write_raw_jsonl(tmp_path, segs)
        with pytest.raises(ResumeError, match="non-monotonic t1"):
            load_resume_state(str(tmp_path), "2026-01-01_12-00-00", "large-v3")

    def test_t0_exceeds_t1(self, tmp_path):
        segs = [_seg_dict("seg_0001", 5.0, 3.0)]
        _write_raw_jsonl(tmp_path, segs)
        with pytest.raises(ResumeError, match="t0=5.0 > t1=3.0"):
            load_resume_state(str(tmp_path), "2026-01-01_12-00-00", "large-v3")

    def test_corrupt_json(self, tmp_path):
        p = tmp_path / "raw_2026-01-01_12-00-00_large-v3.json"
        p.write_text('{"seg_id":"seg_0001"}\n{bad json\n', encoding="utf-8")
        with pytest.raises(ResumeError, match="corrupt JSONL at line 2"):
            load_resume_state(str(tmp_path), "2026-01-01_12-00-00", "large-v3")


# ── component resume ─────────────────────────────────────────────────

class TestCommitterResume:
    def test_resume_counters(self):
        c = SegmentCommitter("test", "en", start_seg=5, start_para=2)
        seg = c.commit(10.0, 11.0, "spk_0", "hello")
        assert seg.seg_id == "seg_0006"
        assert seg.paragraph_id == "para_0002"
        assert c.seg_count == 6


class TestRawSegmentFromDict:
    def test_roundtrip(self):
        original = RawSegment(
            seg_id="seg_0001", t0=1.5, t1=3.2, speaker_id="spk_0",
            raw_text="hello world", model_name="large-v3",
            language="en", paragraph_id="para_0000",
        )
        d = original.to_dict()
        restored = RawSegment.from_dict(d)
        assert restored == original


class TestOutputWriterResume:
    def test_reuses_paths(self, tmp_path):
        writer = OutputWriter(
            str(tmp_path), "large-v3",
            session_ts="2026-01-01_12-00-00", model_tag="large-v3",
        )
        assert writer.raw_json_path.name == "raw_2026-01-01_12-00-00_large-v3.json"
        assert writer.normalized_json_path.name == "normalized_2026-01-01_12-00-00_large-v3.json"
        writer._close_raw_handles()

    def test_appends_to_existing(self, tmp_path):
        # Write initial segment
        segs = [_seg_dict("seg_0001", 0.0, 1.0)]
        _write_raw_jsonl(tmp_path, segs)
        # Also create the txt file
        (tmp_path / "raw_2026-01-01_12-00-00_large-v3.txt").write_text(
            "", encoding="utf-8"
        )

        # Open in resume mode and append
        writer = OutputWriter(
            str(tmp_path), "large-v3",
            session_ts="2026-01-01_12-00-00", model_tag="large-v3",
        )
        seg2 = RawSegment(
            seg_id="seg_0002", t0=2.0, t1=3.0, speaker_id="spk_0",
            raw_text="world", model_name="large-v3",
            language="en", paragraph_id="para_0000",
        )
        writer.append_raw(seg2)
        writer._close_raw_handles()

        # Read back — should have both segments
        lines = writer.raw_json_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["seg_id"] == "seg_0001"
        assert json.loads(lines[1])["seg_id"] == "seg_0002"


class TestFinalizeFromRaw:
    def test_renormalizes_all_segments(self, tmp_path):
        from app.normalize import Normalizer
        from app.config import AppConfig

        segs = [
            _seg_dict("seg_0001", 0.0, 1.0, raw_text="hello"),
            _seg_dict("seg_0002", 2.0, 3.0, raw_text="world"),
            _seg_dict("seg_0003", 4.0, 5.0, raw_text="test"),
        ]
        _write_raw_jsonl(tmp_path, segs)
        # Create txt file so append mode works
        (tmp_path / "raw_2026-01-01_12-00-00_large-v3.txt").write_text(
            "", encoding="utf-8"
        )

        writer = OutputWriter(
            str(tmp_path), "large-v3",
            session_ts="2026-01-01_12-00-00", model_tag="large-v3",
        )
        normalizer = Normalizer(AppConfig(normalization=AppConfig().normalization))
        writer.finalize_from_raw(normalizer, timeout=5.0)

        # Normalized JSON should contain all 3 segments
        norm_data = json.loads(
            writer.normalized_json_path.read_text(encoding="utf-8")
        )
        assert len(norm_data) == 3
        assert norm_data[0]["seg_id"] == "seg_0001"
        assert norm_data[2]["seg_id"] == "seg_0003"


# ── WAV handling ─────────────────────────────────────────────────────

class TestWriteSessionWavOverride:
    def test_filename_override(self, tmp_path):
        chunks = [np.zeros(1600, dtype=np.float32)]
        path = _write_session_wav(
            chunks, 16000, str(tmp_path),
            filename_override="audio_2026-01-01_12-00-00_part2.wav",
        )
        assert path is not None
        assert path.name == "audio_2026-01-01_12-00-00_part2.wav"


class TestConcatenateWavs:
    def test_concatenates_and_preserves_parts(self, tmp_path):
        wav1 = _write_wav(tmp_path / "part1.wav", n_samples=8000)
        wav2 = _write_wav(tmp_path / "part2.wav", n_samples=16000)

        combined = _concatenate_wavs(
            [wav1, wav2], 16000, str(tmp_path), "combined.wav"
        )

        # Verify combined frame count
        with wave.open(str(combined), "rb") as wf:
            assert wf.getnframes() == 24000

        # Verify parts still exist
        assert wav1.exists()
        assert wav2.exists()


# ── CLI ──────────────────────────────────────────────────────────────

class TestResumeCli:
    def test_flag_parsed(self):
        parser = build_arg_parser()
        args = parser.parse_args(["--resume", "2026-01-01_12-00-00"])
        assert args.resume == "2026-01-01_12-00-00"

    def test_default_none(self):
        parser = build_arg_parser()
        args = parser.parse_args([])
        assert args.resume is None

    def test_resume_and_session_mutually_exclusive(self):
        """Both flags parsed but main() would reject the combination."""
        parser = build_arg_parser()
        args = parser.parse_args([
            "--resume", "2026-01-01_12-00-00",
            "--session", "2026-01-01_12-00-00",
        ])
        # Both parse fine — the mutual exclusion is enforced in main()
        assert args.resume is not None
        assert args.session is not None
