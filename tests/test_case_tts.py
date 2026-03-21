"""Tests for TTS case execution."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.case_tts import case_to_audio, run_case_tts
from app.case_system import run_case


# ── helpers ────────────────────────────────────────────────────────


def _minimal_case(**overrides) -> dict:
    case = {
        "case_id": "test_01",
        "segments": [
            {
                "seg_id": "seg_0001",
                "t0": 0.0,
                "t1": 3.0,
                "speaker_id": "spk_0",
                "normalized_text": "Patient has headache and fever for 3 days.",
            },
        ],
        "ground_truth": {"key_findings": ["headache", "fever"]},
    }
    case.update(overrides)
    return case


def _mock_tts_provider(success: bool = True, audio_path: str = ""):
    """Return a mock TTSProvider."""
    provider = MagicMock()
    provider.name = "mock"
    provider.synthesize.return_value = {
        "audio_path": audio_path,
        "provider": "mock",
        "voice": "test-voice",
        "text": "Patient has headache and fever for 3 days.",
        "success": success,
        "error": None if success else "mock error",
        "synthetic": True,
    }
    return provider


# ── case_to_audio ─────────────────────────────────────────────────


class TestCaseToAudio:
    def test_joins_text_and_calls_provider(self, tmp_path):
        mock_provider = _mock_tts_provider(success=True, audio_path=str(tmp_path / "out.wav"))
        with patch("app.case_tts.get_tts_provider", return_value=mock_provider):
            result = case_to_audio(_minimal_case(), tmp_path, provider="mock")

        assert result["synthetic"] is True
        mock_provider.synthesize.assert_called_once()
        call_args = mock_provider.synthesize.call_args
        # Text should be the joined segment text.
        assert "headache" in call_args[0][0]
        assert "fever" in call_args[0][0]

    def test_empty_text_returns_failure(self, tmp_path):
        case = _minimal_case(segments=[{
            "seg_id": "seg_0001", "t0": 0.0, "t1": 1.0,
            "speaker_id": "spk_0", "normalized_text": "",
        }])
        with patch("app.case_tts.get_tts_provider") as mock_get:
            result = case_to_audio(case, tmp_path)

        assert result["success"] is False
        assert "No text" in result["error"]
        assert result["synthetic"] is True
        mock_get.assert_not_called()

    def test_output_path_uses_case_id(self, tmp_path):
        mock_provider = _mock_tts_provider(success=True)
        with patch("app.case_tts.get_tts_provider", return_value=mock_provider):
            case_to_audio(_minimal_case(case_id="chest_pain"), tmp_path)

        call_args = mock_provider.synthesize.call_args
        output_path = call_args[0][1]
        assert "chest_pain" in str(output_path)

    def test_provider_name_passed_through(self, tmp_path):
        mock_provider = _mock_tts_provider(success=True)
        with patch("app.case_tts.get_tts_provider", return_value=mock_provider) as mock_get:
            case_to_audio(_minimal_case(), tmp_path, provider="edge")

        mock_get.assert_called_once_with("edge")


# ── run_case_tts ──────────────────────────────────────────────────


class TestRunCaseTts:
    def _mock_asr_engine(self, results=None):
        """Build a mock ASR engine returning configurable results."""
        from app.asr import AsrResult

        if results is None:
            results = [
                AsrResult(start=0.0, end=3.0,
                          text="Patient has headache and fever for 3 days."),
            ]
        engine = MagicMock()
        engine.transcribe.return_value = results
        return engine

    def test_returns_same_keys_as_run_case(self, tmp_path):
        """Result bundle shape matches run_case() plus tts_result."""
        case = _minimal_case()
        text_result = run_case(case)
        text_keys = set(text_result.keys())

        mock_provider = _mock_tts_provider(
            success=True, audio_path=str(tmp_path / "out.wav"),
        )
        # Create a minimal WAV for read_wav_float32.
        import struct
        import wave

        wav_path = tmp_path / "out.wav"
        samples = struct.pack("<1600h", *([0] * 1600))
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(samples)

        with patch("app.case_tts.get_tts_provider", return_value=mock_provider):
            tts_result = run_case_tts(
                case, tmp_path, provider="mock",
                asr_engine=self._mock_asr_engine(),
            )

        tts_keys = set(tts_result.keys())
        # TTS result has all text-mode keys plus tts_result.
        assert text_keys.issubset(tts_keys)
        assert "tts_result" in tts_keys

    def test_result_contains_clinical_state(self, tmp_path):
        import struct
        import wave

        wav_path = tmp_path / "out.wav"
        samples = struct.pack("<1600h", *([0] * 1600))
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(samples)

        mock_provider = _mock_tts_provider(
            success=True, audio_path=str(wav_path),
        )
        with patch("app.case_tts.get_tts_provider", return_value=mock_provider):
            result = run_case_tts(
                _minimal_case(), tmp_path, provider="mock",
                asr_engine=self._mock_asr_engine(),
            )

        state = result["session"]["clinical_state"]
        assert "symptoms" in state
        assert "encounter_output" in state
        assert "hypotheses" in state

    def test_ground_truth_preserved(self, tmp_path):
        import struct
        import wave

        wav_path = tmp_path / "out.wav"
        samples = struct.pack("<1600h", *([0] * 1600))
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(samples)

        mock_provider = _mock_tts_provider(
            success=True, audio_path=str(wav_path),
        )
        case = _minimal_case(ground_truth={"key_findings": ["headache"]})
        with patch("app.case_tts.get_tts_provider", return_value=mock_provider):
            result = run_case_tts(
                case, tmp_path, provider="mock",
                asr_engine=self._mock_asr_engine(),
            )

        assert result["ground_truth"] == {"key_findings": ["headache"]}

    def test_tts_failure_returns_empty_session(self, tmp_path):
        mock_provider = _mock_tts_provider(success=False)
        with patch("app.case_tts.get_tts_provider", return_value=mock_provider):
            result = run_case_tts(
                _minimal_case(), tmp_path, provider="mock",
            )

        assert result["session"] == {}
        assert result["tts_result"]["success"] is False

    def test_invalid_case_returns_early(self, tmp_path):
        bad_case = {"case_id": "bad"}  # missing segments
        result = run_case_tts(bad_case, tmp_path)
        assert result["validation"]["valid"] is False
        assert result["session"] == {}
        assert result["tts_result"] is None

    def test_empty_asr_produces_empty_state(self, tmp_path):
        import struct
        import wave

        wav_path = tmp_path / "out.wav"
        samples = struct.pack("<1600h", *([0] * 1600))
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(samples)

        mock_provider = _mock_tts_provider(
            success=True, audio_path=str(wav_path),
        )
        with patch("app.case_tts.get_tts_provider", return_value=mock_provider):
            result = run_case_tts(
                _minimal_case(), tmp_path, provider="mock",
                asr_engine=self._mock_asr_engine(results=[]),
            )

        # Should still produce a valid result with empty clinical state.
        assert result["session"] != {}
        state = result["session"]["clinical_state"]
        assert state["symptoms"] == []


# ── text mode preserved ───────────────────────────────────────────


class TestTextModePreserved:
    def test_run_case_unchanged(self):
        """Text-mode run_case still works identically."""
        case = _minimal_case()
        result = run_case(case)
        assert result["validation"]["valid"]
        state = result["session"]["clinical_state"]
        assert "headache" in state["symptoms"]
        assert "tts_result" not in result


# ── synthetic safety ──────────────────────────────────────────────


class TestSyntheticSafety:
    def test_tts_result_marked_synthetic(self, tmp_path):
        import struct
        import wave

        wav_path = tmp_path / "out.wav"
        samples = struct.pack("<1600h", *([0] * 1600))
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(samples)

        mock_provider = _mock_tts_provider(
            success=True, audio_path=str(wav_path),
        )
        with patch("app.case_tts.get_tts_provider", return_value=mock_provider):
            result = run_case_tts(
                _minimal_case(), tmp_path, provider="mock",
                asr_engine=MagicMock(transcribe=MagicMock(return_value=[])),
            )

        assert result["tts_result"]["synthetic"] is True

    def test_provider_must_be_explicit(self):
        """No silent fallback — provider name is always required."""
        from app.tts_provider import get_tts_provider

        # Empty string is not a valid provider.
        with pytest.raises(ValueError):
            get_tts_provider("")


# ── input metadata ───────────────────────────────────────────────


class TestInputMetadata:
    def test_text_mode_metadata(self):
        result = run_case(_minimal_case())
        meta = result["input_metadata"]
        assert meta["mode"] == "text"
        assert meta["synthetic"] is False
        assert meta["tts"] is None

    def test_text_mode_invalid_case_has_metadata(self):
        result = run_case({"case_id": "bad"})
        meta = result["input_metadata"]
        assert meta["mode"] == "text"
        assert meta["synthetic"] is False

    def test_tts_mode_metadata_on_success(self, tmp_path):
        import struct
        import wave

        wav_path = tmp_path / "out.wav"
        samples = struct.pack("<1600h", *([0] * 1600))
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(samples)

        mock_provider = _mock_tts_provider(
            success=True, audio_path=str(wav_path),
        )
        with patch("app.case_tts.get_tts_provider", return_value=mock_provider):
            result = run_case_tts(
                _minimal_case(), tmp_path, provider="mock",
                asr_engine=MagicMock(transcribe=MagicMock(return_value=[])),
            )

        meta = result["input_metadata"]
        assert meta["mode"] == "tts"
        assert meta["synthetic"] is True
        assert meta["tts"]["provider"] == "mock"
        assert meta["tts"]["voice"] == "test-voice"

    def test_tts_mode_metadata_on_failure(self, tmp_path):
        mock_provider = _mock_tts_provider(success=False)
        with patch("app.case_tts.get_tts_provider", return_value=mock_provider):
            result = run_case_tts(_minimal_case(), tmp_path, provider="mock")

        meta = result["input_metadata"]
        assert meta["mode"] == "tts"
        assert meta["synthetic"] is True
        assert meta["tts"] is None  # no provider info on failure

    def test_tts_mode_invalid_case_has_metadata(self, tmp_path):
        result = run_case_tts({"case_id": "bad"}, tmp_path)
        meta = result["input_metadata"]
        assert meta["mode"] == "tts"
        assert meta["synthetic"] is True
        assert meta["tts"] is None
