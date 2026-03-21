"""Tests for TTS provider abstraction."""

from __future__ import annotations

import struct
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.tts_provider import (
    TTSProvider,
    EdgeTTSProvider,
    get_tts_provider,
    list_providers,
    write_pcm16_wav,
    read_wav_float32,
    _TTS_RESULT_KEYS,
    _tts_result,
)


# ── provider factory ──────────────────────────────────────────────


class TestProviderFactory:
    def test_get_edge_provider(self):
        provider = get_tts_provider("edge")
        assert isinstance(provider, EdgeTTSProvider)
        assert isinstance(provider, TTSProvider)

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown TTS provider"):
            get_tts_provider("nonexistent")

    def test_error_lists_available(self):
        with pytest.raises(ValueError, match="edge"):
            get_tts_provider("nonexistent")

    def test_list_providers(self):
        names = list_providers()
        assert "edge" in names
        assert names == sorted(names)

    def test_all_providers_implement_abc(self):
        for name in list_providers():
            provider = get_tts_provider(name)
            assert isinstance(provider, TTSProvider)
            assert hasattr(provider, "name")
            assert hasattr(provider, "synthesize")


# ── result contract ───────────────────────────────────────────────


class TestResultContract:
    def test_result_keys(self):
        result = _tts_result("path.wav", "edge", "voice", "hello", True)
        assert set(result.keys()) == _TTS_RESULT_KEYS

    def test_result_types(self):
        result = _tts_result("/tmp/out.wav", "edge", "v1", "text", True)
        assert isinstance(result["audio_path"], str)
        assert isinstance(result["success"], bool)
        assert isinstance(result["synthetic"], bool)
        assert result["synthetic"] is True

    def test_result_error_field(self):
        result = _tts_result("", "edge", "", "", False, error="failed")
        assert result["error"] == "failed"
        assert result["success"] is False

    def test_result_success_no_error(self):
        result = _tts_result("out.wav", "edge", "v1", "hi", True)
        assert result["error"] is None


# ── edge provider ─────────────────────────────────────────────────


class TestEdgeTTSProvider:
    def test_name(self):
        assert EdgeTTSProvider().name == "edge"

    def test_synthesize_returns_result_shape(self, tmp_path):
        """Mock edge_tts to verify result shape without network."""
        provider = EdgeTTSProvider()
        wav_path = tmp_path / "out.wav"

        mock_comm = MagicMock()
        mock_edge = MagicMock()
        mock_edge.Communicate.return_value = mock_comm

        with patch.dict("sys.modules", {"edge_tts": mock_edge}), \
             patch("app.tts_provider._convert_mp3_to_wav"):
            result = provider.synthesize("hello world", wav_path)

        assert set(result.keys()) == _TTS_RESULT_KEYS
        assert result["provider"] == "edge"
        assert result["text"] == "hello world"
        assert result["synthetic"] is True

    def test_missing_edge_tts_returns_failure(self, tmp_path):
        """If edge_tts is not installed, returns success=False."""
        provider = EdgeTTSProvider()

        with patch.dict("sys.modules", {"edge_tts": None}):
            # Force ImportError by patching the import inside synthesize.
            original = __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__

            def mock_import(name, *args, **kwargs):
                if name == "edge_tts":
                    raise ImportError("No module named 'edge_tts'")
                return original(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=mock_import):
                result = provider.synthesize("hello", tmp_path / "out.wav")

        assert result["success"] is False
        assert "not installed" in result["error"]

    def test_default_voice_english(self, tmp_path):
        provider = EdgeTTSProvider()
        wav_path = tmp_path / "out.wav"
        mock_edge = MagicMock()
        mock_edge.Communicate.return_value = MagicMock()

        with patch.dict("sys.modules", {"edge_tts": mock_edge}), \
             patch("app.tts_provider._convert_mp3_to_wav"):
            result = provider.synthesize("hello", wav_path, lang="en")

        assert result["voice"] == "en-US-GuyNeural"

    def test_default_voice_danish(self, tmp_path):
        provider = EdgeTTSProvider()
        wav_path = tmp_path / "out.wav"
        mock_edge = MagicMock()
        mock_edge.Communicate.return_value = MagicMock()

        with patch.dict("sys.modules", {"edge_tts": mock_edge}), \
             patch("app.tts_provider._convert_mp3_to_wav"):
            result = provider.synthesize("hej", wav_path, lang="da")

        assert result["voice"] == "da-DK-JeppeNeural"

    def test_custom_voice_overrides_default(self, tmp_path):
        provider = EdgeTTSProvider()
        wav_path = tmp_path / "out.wav"
        mock_edge = MagicMock()
        mock_edge.Communicate.return_value = MagicMock()

        with patch.dict("sys.modules", {"edge_tts": mock_edge}), \
             patch("app.tts_provider._convert_mp3_to_wav"):
            result = provider.synthesize("hello", wav_path, voice="custom-voice")

        assert result["voice"] == "custom-voice"


# ── audio helpers ─────────────────────────────────────────────────


class TestAudioHelpers:
    def test_write_and_read_pcm16_wav(self, tmp_path):
        """Round-trip: write PCM16 → read back as float32."""
        # Generate 100 samples of a simple signal.
        samples = struct.pack("<100h", *range(-50, 50))
        wav_path = tmp_path / "test.wav"

        write_pcm16_wav(samples, wav_path, sample_rate=16000)

        assert wav_path.exists()

        audio, rate = read_wav_float32(wav_path)
        assert rate == 16000
        assert len(audio) == 100
        assert audio.dtype.name == "float32"
        # Check value range.
        assert audio.min() >= -1.0
        assert audio.max() <= 1.0

    def test_write_creates_parent_dirs(self, tmp_path):
        samples = struct.pack("<10h", *range(10))
        wav_path = tmp_path / "sub" / "dir" / "test.wav"

        write_pcm16_wav(samples, wav_path)
        assert wav_path.exists()

    def test_read_rejects_non_16bit(self, tmp_path):
        """8-bit WAV should raise ValueError."""
        import wave

        wav_path = tmp_path / "8bit.wav"
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(1)  # 8-bit
            wf.setframerate(16000)
            wf.writeframes(b"\x80" * 100)

        with pytest.raises(ValueError, match="16-bit"):
            read_wav_float32(wav_path)


# ── synthetic marker ──────────────────────────────────────────────


class TestSyntheticMarker:
    def test_all_results_marked_synthetic(self):
        """Every TTS result must carry synthetic=True."""
        result = _tts_result("path.wav", "edge", "v1", "text", True)
        assert result["synthetic"] is True

        result = _tts_result("", "edge", "", "", False, error="fail")
        assert result["synthetic"] is True
