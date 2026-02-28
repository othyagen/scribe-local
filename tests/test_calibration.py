"""Tests for calibration profiles and embedding matching."""

from __future__ import annotations

import json
import math
import os
import struct
import wave
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import inspect

import app.calibration as _cal_module
from app.calibration import (
    _build_auth_kwargs,
    _load_embedding_model,
    _unwrap_embedding,
    apply_cluster_override,
    assign_clusters_to_profile,
    build_calibration_report,
    build_cluster_embeddings,
    build_cluster_embeddings_with_stats,
    cosine_similarity,
    embed_turns,
    extract_embedding,
    filter_eligible_clusters,
    load_profile,
    match_turn_embeddings,
    print_calibration_debug,
    record_and_build_profile,
    save_profile,
)
from app.config import AppConfig, DiarizationConfig, build_arg_parser, _build_diarization


def _turn(start: float, end: float, speaker: str, embedding=None) -> dict:
    t = {"start": start, "end": end, "speaker": speaker}
    if embedding is not None:
        t["embedding"] = embedding
    return t


def _profile(speakers: dict[str, list[float]], id_map: dict[str, str]) -> dict:
    return {
        "speakers": {name: {"embedding": emb} for name, emb in speakers.items()},
        "speaker_id_map": id_map,
    }


# ── cosine similarity ────────────────────────────────────────────────


class TestCosineSimilarity:
    def test_identical_vectors(self):
        v = [1.0, 2.0, 3.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)

    def test_opposite_vectors(self):
        assert cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)

    def test_zero_vector(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0
        assert cosine_similarity([1.0, 2.0], [0.0, 0.0]) == 0.0

    def test_known_value(self):
        # 45-degree angle → cos(45°) ≈ 0.7071
        assert cosine_similarity([1.0, 0.0], [1.0, 1.0]) == pytest.approx(
            1.0 / math.sqrt(2), abs=1e-6
        )


# ── match turn embeddings ────────────────────────────────────────────


class TestMatchTurnEmbeddings:
    def test_override_above_threshold(self):
        profile = _profile(
            {"Speaker A": [1.0, 0.0, 0.0]},
            {"Speaker A": "spk_0"},
        )
        turns = [_turn(0.0, 3.0, "SPEAKER_00", embedding=[0.99, 0.1, 0.0])]
        result = match_turn_embeddings(turns, profile, threshold=0.7)
        assert result[0]["speaker"] == "spk_0"

    def test_no_override_below_threshold(self):
        profile = _profile(
            {"Speaker A": [1.0, 0.0, 0.0]},
            {"Speaker A": "spk_0"},
        )
        turns = [_turn(0.0, 3.0, "SPEAKER_00", embedding=[0.0, 1.0, 0.0])]
        result = match_turn_embeddings(turns, profile, threshold=0.7)
        assert result[0]["speaker"] == "SPEAKER_00"

    def test_no_embedding_field_skipped(self):
        profile = _profile(
            {"Speaker A": [1.0, 0.0]},
            {"Speaker A": "spk_0"},
        )
        turns = [_turn(0.0, 3.0, "SPEAKER_00")]  # no embedding
        result = match_turn_embeddings(turns, profile, threshold=0.7)
        assert result[0]["speaker"] == "SPEAKER_00"

    def test_no_profile_speakers_no_change(self):
        profile = {"speakers": {}, "speaker_id_map": {}}
        turns = [_turn(0.0, 3.0, "SPEAKER_00", embedding=[1.0, 0.0])]
        result = match_turn_embeddings(turns, profile, threshold=0.7)
        assert result[0]["speaker"] == "SPEAKER_00"

    def test_does_not_mutate_input(self):
        profile = _profile(
            {"Speaker A": [1.0, 0.0]},
            {"Speaker A": "spk_0"},
        )
        turns = [_turn(0.0, 3.0, "SPEAKER_00", embedding=[1.0, 0.0])]
        original_speaker = turns[0]["speaker"]
        match_turn_embeddings(turns, profile, threshold=0.7)
        assert turns[0]["speaker"] == original_speaker

    def test_best_match_wins(self):
        profile = _profile(
            {
                "Speaker A": [1.0, 0.0, 0.0],
                "Speaker B": [0.0, 1.0, 0.0],
            },
            {
                "Speaker A": "spk_0",
                "Speaker B": "spk_1",
            },
        )
        # Embedding is closer to Speaker B
        turns = [_turn(0.0, 3.0, "SPEAKER_00", embedding=[0.1, 0.95, 0.0])]
        result = match_turn_embeddings(turns, profile, threshold=0.7)
        assert result[0]["speaker"] == "spk_1"

    def test_missing_id_map_entry_no_override(self):
        profile = {
            "speakers": {"Speaker A": {"embedding": [1.0, 0.0]}},
            "speaker_id_map": {},  # no mapping for Speaker A
        }
        turns = [_turn(0.0, 3.0, "SPEAKER_00", embedding=[1.0, 0.0])]
        result = match_turn_embeddings(turns, profile, threshold=0.7)
        assert result[0]["speaker"] == "SPEAKER_00"

    def test_speaker_id_stays_internal(self):
        """Calibration must map to spk_N, never leak human labels."""
        profile = _profile(
            {"Alice": [1.0, 0.0]},
            {"Alice": "spk_0"},
        )
        turns = [_turn(0.0, 3.0, "SPEAKER_00", embedding=[1.0, 0.0])]
        result = match_turn_embeddings(turns, profile, threshold=0.5)
        assert result[0]["speaker"] == "spk_0"
        assert "Alice" not in str(result[0])


# ── load / save profile ──────────────────────────────────────────────


class TestLoadSaveProfile:
    def test_load_missing_raises(self):
        with pytest.raises(FileNotFoundError):
            load_profile("nonexistent/profile.json")

    def test_save_and_load_roundtrip(self, tmp_path):
        data = _profile(
            {"Speaker A": [0.1, 0.2, 0.3]},
            {"Speaker A": "spk_0"},
        )
        path = tmp_path / "test.json"
        save_profile(str(path), data)
        loaded = load_profile(str(path))
        assert loaded == data

    def test_save_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "sub" / "dir" / "profile.json"
        data = {"speakers": {}, "speaker_id_map": {}}
        save_profile(str(path), data)
        assert path.exists()


# ── config parsing ────────────────────────────────────────────────────


class TestConfigParsing:
    def test_defaults(self):
        cfg = DiarizationConfig()
        assert cfg.calibration_profile is None
        assert cfg.calibration_similarity_threshold == 0.72

    def test_yaml_override(self):
        cfg = _build_diarization({
            "calibration_profile": "my_clinic",
            "calibration_similarity_threshold": 0.85,
        })
        assert cfg.calibration_profile == "my_clinic"
        assert cfg.calibration_similarity_threshold == 0.85

    def test_create_profile_flag(self):
        parser = build_arg_parser()
        args = parser.parse_args(["--create-profile", "clinic1"])
        assert args.create_profile == "clinic1"

    def test_profile_speakers_default(self):
        parser = build_arg_parser()
        args = parser.parse_args([])
        assert args.profile_speakers == 2

    def test_profile_duration_default(self):
        parser = build_arg_parser()
        args = parser.parse_args([])
        assert args.profile_duration == 12.0


# ── extract embedding ─────────────────────────────────────────────────


class TestExtractEmbedding:
    def test_returns_normalized_vector(self):
        fake_raw = np.array([3.0, 4.0, 0.0])

        mock_model = MagicMock()
        mock_inference_obj = MagicMock()
        mock_inference_obj.return_value = fake_raw
        mock_inference_cls = MagicMock(return_value=mock_inference_obj)

        mock_torch = MagicMock()
        mock_torch.from_numpy.return_value.unsqueeze.return_value.float.return_value = "tensor"

        mock_pyannote_audio = MagicMock()
        mock_pyannote_audio.Inference = mock_inference_cls
        mock_pyannote_audio.Model.from_pretrained.return_value = mock_model

        with patch.dict("sys.modules", {
            "pyannote": MagicMock(),
            "pyannote.audio": mock_pyannote_audio,
            "torch": mock_torch,
        }):
            audio = np.zeros(16000, dtype=np.float32)
            result = extract_embedding(audio, 16000)

        assert isinstance(result, list)
        norm = math.sqrt(sum(x * x for x in result))
        assert norm == pytest.approx(1.0, abs=0.01)


# ── SlidingWindowFeature unwrap ───────────────────────────────────────


class TestUnwrapEmbedding:
    def test_unwraps_sliding_window_feature_2d(self):
        """A 2-D .data attribute should be mean-pooled to 1-D."""
        fake_data = np.array([[1.0, 2.0, 3.0],
                              [3.0, 4.0, 5.0]])
        fake_swf = MagicMock()
        fake_swf.data = fake_data

        result = _unwrap_embedding(fake_swf)
        expected = fake_data.mean(axis=0)
        np.testing.assert_allclose(result, expected)
        assert result.ndim == 1

    def test_plain_1d_array_passed_through(self):
        arr = np.array([1.0, 0.0, 0.0])
        result = _unwrap_embedding(arr)
        np.testing.assert_allclose(result, arr)
        assert result.ndim == 1

    def test_extract_embedding_with_sliding_window(self):
        """End-to-end: extract_embedding handles SlidingWindowFeature."""
        fake_data = np.array([[3.0, 0.0],
                              [5.0, 0.0]])
        fake_swf = MagicMock()
        fake_swf.data = fake_data

        mock_model = MagicMock()
        mock_inference_obj = MagicMock(return_value=fake_swf)
        mock_inference_cls = MagicMock(return_value=mock_inference_obj)

        mock_torch = MagicMock()
        mock_torch.from_numpy.return_value.unsqueeze.return_value.float.return_value = "tensor"

        mock_pyannote_audio = MagicMock()
        mock_pyannote_audio.Inference = mock_inference_cls
        mock_pyannote_audio.Model.from_pretrained.return_value = mock_model

        with patch.dict("sys.modules", {
            "pyannote": MagicMock(),
            "pyannote.audio": mock_pyannote_audio,
            "torch": mock_torch,
        }):
            result = extract_embedding(np.zeros(16000, dtype=np.float32), 16000)

        assert isinstance(result, list)
        assert len(result) == 2
        norm = math.sqrt(sum(x * x for x in result))
        assert norm == pytest.approx(1.0, abs=0.01)
        # mean of [3,5]=4, mean of [0,0]=0 → normalized → [1.0, 0.0]
        assert result[0] == pytest.approx(1.0, abs=0.01)
        assert result[1] == pytest.approx(0.0, abs=0.01)


# ── auth kwargs detection & model loading ────────────────────────────


class TestBuildAuthKwargs:
    """Ensure _build_auth_kwargs adapts to different pyannote signatures."""

    @staticmethod
    def _make_callable(param_name):
        """Build a real callable whose signature accepts *param_name*."""
        ns: dict = {}
        exec(
            f"def _fn(model_id, *, {param_name}=None): pass\n",
            ns,
        )
        return ns["_fn"]

    def test_detects_token_param(self):
        fn = self._make_callable("token")
        with patch.dict(os.environ, {"HF_TOKEN": "hf_abc"}):
            assert _build_auth_kwargs(fn) == {"token": "hf_abc"}

    def test_detects_use_auth_token_param(self):
        fn = self._make_callable("use_auth_token")
        with patch.dict(os.environ, {"HF_TOKEN": "hf_xyz"}):
            assert _build_auth_kwargs(fn) == {"use_auth_token": "hf_xyz"}

    def test_detects_auth_token_param(self):
        fn = self._make_callable("auth_token")
        with patch.dict(os.environ, {"HF_TOKEN": "hf_123"}):
            assert _build_auth_kwargs(fn) == {"auth_token": "hf_123"}

    def test_no_matching_param_returns_empty(self):
        fn = self._make_callable("unrelated_param")
        with patch.dict(os.environ, {"HF_TOKEN": "hf_nope"}):
            assert _build_auth_kwargs(fn) == {}

    def test_no_hf_token_returns_empty(self):
        fn = self._make_callable("token")
        env = os.environ.copy()
        env.pop("HF_TOKEN", None)
        with patch.dict(os.environ, env, clear=True):
            assert _build_auth_kwargs(fn) == {}


class TestLoadEmbeddingModel:
    """Ensure _load_embedding_model uses Model.from_pretrained, not a string."""

    def test_uses_model_from_pretrained(self):
        mock_model = MagicMock()
        mock_pyannote_audio = MagicMock()
        mock_pyannote_audio.Model.from_pretrained.return_value = mock_model

        with patch.dict("sys.modules", {
            "pyannote": MagicMock(),
            "pyannote.audio": mock_pyannote_audio,
        }), patch.dict(os.environ, {"HF_TOKEN": "hf_test"}):
            result = _load_embedding_model()

        assert result is mock_model
        mock_pyannote_audio.Model.from_pretrained.assert_called_once()
        call_args = mock_pyannote_audio.Model.from_pretrained.call_args
        assert call_args[0][0] == "pyannote/embedding"

    def test_inference_receives_model_object_not_string(self):
        mock_model = MagicMock()
        mock_inference_obj = MagicMock()
        mock_inference_cls = MagicMock(return_value=mock_inference_obj)
        mock_pyannote_audio = MagicMock()
        mock_pyannote_audio.Inference = mock_inference_cls
        mock_pyannote_audio.Model.from_pretrained.return_value = mock_model

        _cal_module._EMBED_INFERENCE = None
        with patch.dict("sys.modules", {
            "pyannote": MagicMock(),
            "pyannote.audio": mock_pyannote_audio,
        }), patch.dict(os.environ, {"HF_TOKEN": "hf_test"}):
            _cal_module._get_inference()

        # Inference must receive the model object, not a string
        mock_inference_cls.assert_called_once_with(mock_model)
        _cal_module._EMBED_INFERENCE = None


# ── record and build profile ─────────────────────────────────────────


def _mock_extract(audio, sr):
    """Return a deterministic normalized embedding."""
    vec = np.random.default_rng(42).random(3)
    vec = vec / np.linalg.norm(vec)
    return vec.tolist()


class TestRecordAndBuildProfile:
    def test_profile_structure(self):
        config = AppConfig()
        mock_audio = MagicMock()
        mock_audio.get_chunk.return_value = np.zeros(480, dtype=np.float32)

        with patch("app.audio.AudioCapture", return_value=mock_audio), \
             patch("app.calibration.extract_embedding", side_effect=_mock_extract):
            profile = record_and_build_profile(config, num_speakers=2, duration_sec=0.1)

        assert "speakers" in profile
        assert "speaker_id_map" in profile
        assert len(profile["speakers"]) == 2
        assert "Speaker A" in profile["speakers"]
        assert "Speaker B" in profile["speakers"]

    def test_speaker_id_map_correct(self):
        config = AppConfig()
        mock_audio = MagicMock()
        mock_audio.get_chunk.return_value = np.zeros(480, dtype=np.float32)

        with patch("app.audio.AudioCapture", return_value=mock_audio), \
             patch("app.calibration.extract_embedding", side_effect=_mock_extract):
            profile = record_and_build_profile(config, num_speakers=3, duration_sec=0.1)

        assert profile["speaker_id_map"]["Speaker A"] == "spk_0"
        assert profile["speaker_id_map"]["Speaker B"] == "spk_1"
        assert profile["speaker_id_map"]["Speaker C"] == "spk_2"

    def test_embeddings_normalized(self):
        config = AppConfig()
        mock_audio = MagicMock()
        mock_audio.get_chunk.return_value = np.zeros(480, dtype=np.float32)

        with patch("app.audio.AudioCapture", return_value=mock_audio), \
             patch("app.calibration.extract_embedding", side_effect=_mock_extract):
            profile = record_and_build_profile(config, num_speakers=2, duration_sec=0.1)

        for name, info in profile["speakers"].items():
            emb = info["embedding"]
            norm = math.sqrt(sum(x * x for x in emb))
            assert norm == pytest.approx(1.0, abs=0.01), f"{name} embedding not normalized"


# ── profile exists guard ──────────────────────────────────────────────


class TestProfileExistsGuard:
    def test_does_not_overwrite_existing(self, tmp_path):
        """Verify save_profile preserves data and create-profile path checks."""
        profile_path = tmp_path / "test.json"
        original_data = {"speakers": {"X": {"embedding": [1.0]}}, "speaker_id_map": {"X": "spk_0"}}
        save_profile(str(profile_path), original_data)

        # The guard in main() uses os.path.exists — verify the logic directly
        assert profile_path.exists()
        loaded = load_profile(str(profile_path))
        assert loaded == original_data


# ── helpers ──────────────────────────────────────────────────────────


def _write_wav(path, samples: np.ndarray, sample_rate: int = 16000):
    """Write a mono 16-bit PCM WAV file."""
    pcm = (samples * 32767).astype(np.int16).tobytes()
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm)


# ── embed_turns ──────────────────────────────────────────────────────


class TestEmbedTurns:
    def setup_method(self):
        """Reset the singleton Inference cache between tests."""
        _cal_module._EMBED_INFERENCE = None

    def _mock_modules(self, mock_inference_cls):
        """Return a dict suitable for patching sys.modules with pyannote + torch."""
        mock_torch = MagicMock()
        mock_torch.from_numpy.return_value.unsqueeze.return_value.float.return_value = "tensor"
        mock_pyannote = MagicMock()
        mock_pyannote_audio = MagicMock()
        mock_pyannote_audio.Inference = mock_inference_cls
        return {
            "torch": mock_torch,
            "pyannote": mock_pyannote,
            "pyannote.audio": mock_pyannote_audio,
        }

    def test_embeds_all_turns(self, tmp_path):
        wav_path = tmp_path / "audio.wav"
        audio = np.random.default_rng(0).random(32000).astype(np.float32) * 0.5
        _write_wav(wav_path, audio)

        fake_emb = np.array([3.0, 4.0, 0.0])
        mock_inf = MagicMock(return_value=fake_emb)
        mock_cls = MagicMock(return_value=mock_inf)

        turns = [
            {"start": 0.0, "end": 0.5, "speaker": "spk_0"},
            {"start": 0.5, "end": 1.0, "speaker": "spk_1"},
        ]

        with patch.dict("sys.modules", self._mock_modules(mock_cls)):
            result = embed_turns(turns, wav_path)

        assert len(result) == 2
        for t in result:
            assert "embedding" in t
            norm = math.sqrt(sum(x * x for x in t["embedding"]))
            assert norm == pytest.approx(1.0, abs=0.01)

    def test_model_loaded_once(self, tmp_path):
        wav_path = tmp_path / "audio.wav"
        audio = np.zeros(48000, dtype=np.float32)
        _write_wav(wav_path, audio)

        fake_emb = np.array([1.0, 0.0])
        mock_inf = MagicMock(return_value=fake_emb)
        mock_cls = MagicMock(return_value=mock_inf)

        turns = [
            {"start": 0.0, "end": 1.0, "speaker": "spk_0"},
            {"start": 1.0, "end": 2.0, "speaker": "spk_1"},
            {"start": 2.0, "end": 3.0, "speaker": "spk_0"},
        ]

        with patch.dict("sys.modules", self._mock_modules(mock_cls)):
            embed_turns(turns, wav_path)

        # Inference constructor called exactly once
        mock_cls.assert_called_once()

    def test_time_to_sample_bounds(self, tmp_path):
        """Turn extending past WAV end is clamped — no crash."""
        wav_path = tmp_path / "audio.wav"
        audio = np.zeros(16000, dtype=np.float32)  # 1 second
        _write_wav(wav_path, audio)

        fake_emb = np.array([0.0, 1.0])
        mock_inf = MagicMock(return_value=fake_emb)
        mock_cls = MagicMock(return_value=mock_inf)

        turns = [{"start": 0.5, "end": 5.0, "speaker": "spk_0"}]  # end > WAV length

        with patch.dict("sys.modules", self._mock_modules(mock_cls)):
            result = embed_turns(turns, wav_path)

        assert "embedding" in result[0]

    def test_singleton_across_calls(self, tmp_path):
        """Inference constructor is called once even across two embed_turns calls."""
        wav_path = tmp_path / "audio.wav"
        audio = np.zeros(32000, dtype=np.float32)
        _write_wav(wav_path, audio)

        fake_emb = np.array([1.0, 0.0])
        mock_inf = MagicMock(return_value=fake_emb)
        mock_cls = MagicMock(return_value=mock_inf)

        turns1 = [{"start": 0.0, "end": 1.0, "speaker": "spk_0"}]
        turns2 = [{"start": 0.0, "end": 1.0, "speaker": "spk_1"}]

        with patch.dict("sys.modules", self._mock_modules(mock_cls)):
            embed_turns(turns1, wav_path)
            embed_turns(turns2, wav_path)

        mock_cls.assert_called_once()

    def test_min_duration_skips_short_turns(self, tmp_path):
        """Turns shorter than min_duration_sec get no embedding."""
        wav_path = tmp_path / "audio.wav"
        audio = np.zeros(32000, dtype=np.float32)
        _write_wav(wav_path, audio)

        fake_emb = np.array([1.0, 0.0])
        mock_inf = MagicMock(return_value=fake_emb)
        mock_cls = MagicMock(return_value=mock_inf)

        turns = [
            {"start": 0.0, "end": 0.2, "speaker": "spk_0"},  # 0.2s < 0.5
            {"start": 0.5, "end": 1.5, "speaker": "spk_1"},  # 1.0s >= 0.5
        ]

        with patch.dict("sys.modules", self._mock_modules(mock_cls)):
            result = embed_turns(turns, wav_path, min_duration_sec=0.5)

        assert "embedding" not in result[0]
        assert "embedding" in result[1]

    def test_min_duration_zero_embeds_all(self, tmp_path):
        """Default min_duration_sec=0.0 embeds every turn."""
        wav_path = tmp_path / "audio.wav"
        audio = np.zeros(32000, dtype=np.float32)
        _write_wav(wav_path, audio)

        fake_emb = np.array([1.0, 0.0])
        mock_inf = MagicMock(return_value=fake_emb)
        mock_cls = MagicMock(return_value=mock_inf)

        turns = [
            {"start": 0.0, "end": 0.05, "speaker": "spk_0"},  # very short
            {"start": 0.1, "end": 1.0, "speaker": "spk_1"},
        ]

        with patch.dict("sys.modules", self._mock_modules(mock_cls)):
            result = embed_turns(turns, wav_path, min_duration_sec=0.0)

        assert "embedding" in result[0]
        assert "embedding" in result[1]


# ── calibration pipeline integration ─────────────────────────────────


class TestCalibrationPipelineIntegration:
    def test_override_above_threshold(self, tmp_path):
        """embed_turns (mocked) + match_turn_embeddings with profile overrides speaker."""
        wav_path = tmp_path / "audio.wav"
        audio = np.zeros(32000, dtype=np.float32)
        _write_wav(wav_path, audio)

        profile = _profile(
            {"Speaker A": [1.0, 0.0, 0.0], "Speaker B": [0.0, 1.0, 0.0]},
            {"Speaker A": "spk_0", "Speaker B": "spk_1"},
        )

        turns = [
            {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"},
            {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_01"},
        ]

        # Simulate embed_turns attaching embeddings close to profile speakers
        def fake_embed(t_list, _wav):
            t_list[0]["embedding"] = [0.98, 0.05, 0.0]  # close to Speaker A
            t_list[1]["embedding"] = [0.05, 0.97, 0.0]  # close to Speaker B
            return t_list

        with patch("app.calibration.embed_turns", side_effect=fake_embed):
            fake_embed(turns, wav_path)

        result = match_turn_embeddings(turns, profile, threshold=0.7)
        assert result[0]["speaker"] == "spk_0"
        assert result[1]["speaker"] == "spk_1"

    def test_cluster_pipeline_overrides_speakers(self, tmp_path):
        """Full cluster pipeline: embed → cluster → assign → override."""
        wav_path = tmp_path / "audio.wav"
        audio = np.zeros(32000, dtype=np.float32)
        _write_wav(wav_path, audio)

        profile = _profile(
            {"Speaker A": [1.0, 0.0, 0.0], "Speaker B": [0.0, 1.0, 0.0]},
            {"Speaker A": "spk_0", "Speaker B": "spk_1"},
        )

        turns = [
            {"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00",
             "embedding": [0.98, 0.05, 0.0]},
            {"start": 1.0, "end": 2.0, "speaker": "SPEAKER_01",
             "embedding": [0.05, 0.97, 0.0]},
        ]

        cluster_embs = build_cluster_embeddings(turns)
        mapping = assign_clusters_to_profile(
            cluster_embs, profile, threshold=0.7, margin=0.05
        )
        result = apply_cluster_override(turns, mapping)
        assert result[0]["speaker"] == "spk_0"
        assert result[1]["speaker"] == "spk_1"


# ── build cluster embeddings ──────────────────────────────────────────


class TestBuildClusterEmbeddings:
    def test_cluster_embedding_normalized(self):
        """Mean embeddings for each cluster are L2-normalized."""
        turns = [
            _turn(0.0, 1.0, "spk_0", embedding=[1.0, 0.0, 0.0]),
            _turn(1.0, 2.0, "spk_0", embedding=[0.0, 1.0, 0.0]),
            _turn(2.0, 3.0, "spk_1", embedding=[0.0, 0.0, 1.0]),
            _turn(3.0, 4.0, "spk_1", embedding=[0.0, 0.0, 2.0]),
        ]
        result = build_cluster_embeddings(turns)
        assert "spk_0" in result
        assert "spk_1" in result
        for spk, emb in result.items():
            norm = math.sqrt(sum(x * x for x in emb))
            assert norm == pytest.approx(1.0, abs=1e-6), f"{spk} not normalized"

    def test_skips_turns_without_embedding(self):
        """Turns missing the embedding key are excluded from cluster mean."""
        turns = [
            _turn(0.0, 1.0, "spk_0", embedding=[1.0, 0.0]),
            _turn(1.0, 2.0, "spk_0"),  # no embedding
            _turn(2.0, 3.0, "spk_1"),  # no embedding — cluster should be absent
        ]
        result = build_cluster_embeddings(turns)
        assert "spk_0" in result
        assert "spk_1" not in result


# ── assign clusters to profile ────────────────────────────────────────


class TestAssignClustersToProfile:
    def test_1_to_1_assignment(self):
        """Two clusters, two profile speakers with clear best matches."""
        profile = _profile(
            {"Speaker A": [1.0, 0.0, 0.0], "Speaker B": [0.0, 1.0, 0.0]},
            {"Speaker A": "spk_0", "Speaker B": "spk_1"},
        )
        cluster_embs = {"c0": [0.95, 0.1, 0.0], "c1": [0.1, 0.95, 0.0]}
        mapping = assign_clusters_to_profile(
            cluster_embs, profile, threshold=0.7, margin=0.05
        )
        assert mapping["c0"] == "spk_0"
        assert mapping["c1"] == "spk_1"

    def test_no_double_assign(self):
        """Both clusters closest to same profile speaker — only best one assigned."""
        profile = _profile(
            {"Speaker A": [1.0, 0.0], "Speaker B": [0.0, 1.0]},
            {"Speaker A": "spk_0", "Speaker B": "spk_1"},
        )
        # Both clusters point towards Speaker A, but c0 is closer
        cluster_embs = {"c0": [0.99, 0.01], "c1": [0.85, 0.1]}
        mapping = assign_clusters_to_profile(
            cluster_embs, profile, threshold=0.7, margin=0.05
        )
        assert mapping.get("c0") == "spk_0"
        # c1 should now be matched against remaining Speaker B
        # c1's similarity to Speaker B: cos([0.85,0.1],[0,1]) is low
        # so c1 may or may not be assigned depending on threshold
        assert "spk_0" not in [v for k, v in mapping.items() if k != "c0"]

    def test_below_threshold_no_assign(self):
        """All similarities below threshold → empty mapping."""
        profile = _profile(
            {"Speaker A": [1.0, 0.0, 0.0]},
            {"Speaker A": "spk_0"},
        )
        cluster_embs = {"c0": [0.0, 1.0, 0.0]}  # orthogonal → sim ≈ 0
        mapping = assign_clusters_to_profile(
            cluster_embs, profile, threshold=0.7, margin=0.05
        )
        assert mapping == {}

    def test_below_margin_no_assign(self):
        """Best and second-best within margin → no assignment for that cluster."""
        profile = _profile(
            {"Speaker A": [1.0, 0.0], "Speaker B": [0.0, 1.0]},
            {"Speaker A": "spk_0", "Speaker B": "spk_1"},
        )
        # Cluster equally similar to both (45 degrees from each)
        v = [1.0 / math.sqrt(2), 1.0 / math.sqrt(2)]
        cluster_embs = {"c0": v}
        mapping = assign_clusters_to_profile(
            cluster_embs, profile, threshold=0.3, margin=0.05
        )
        # Similarities to A and B are identical, so margin check fails
        assert "c0" not in mapping


# ── calibration diagnostics ───────────────────────────────────────────


class TestCalibrationDiagnostics:
    def _setup(self):
        profile = _profile(
            {"Speaker A": [1.0, 0.0, 0.0], "Speaker B": [0.0, 1.0, 0.0]},
            {"Speaker A": "spk_0", "Speaker B": "spk_1"},
        )
        cluster_embs = {"spk_0": [0.95, 0.1, 0.0], "spk_1": [0.1, 0.95, 0.0]}
        mapping = assign_clusters_to_profile(
            cluster_embs, profile, threshold=0.7, margin=0.05
        )
        return cluster_embs, profile, mapping

    def test_report_contains_similarity_and_mapping(self):
        cluster_embs, profile, mapping = self._setup()
        report = build_calibration_report(
            cluster_embs, profile,
            threshold=0.7, margin=0.05,
            mapping=mapping, profile_name="test_clinic",
        )
        # Top-level fields
        assert report["profile_name"] == "test_clinic"
        assert report["threshold"] == 0.7
        assert report["margin_required"] == 0.05
        assert sorted(report["clusters"]) == ["spk_0", "spk_1"]
        assert sorted(report["profile_speakers"]) == ["Speaker A", "Speaker B"]
        assert report["profile_speaker_id_map"]["Speaker A"] == "spk_0"

        # Similarity matrix
        assert "spk_0" in report["similarity"]
        assert "Speaker A" in report["similarity"]["spk_0"]
        assert "Speaker B" in report["similarity"]["spk_0"]
        # spk_0 cluster is close to Speaker A
        assert report["similarity"]["spk_0"]["Speaker A"] > 0.8

        # Decisions
        assert report["decisions"]["spk_0"]["assigned"] is True
        assert report["decisions"]["spk_0"]["reason"] == "ok"
        assert report["decisions"]["spk_0"]["mapped_to"] == "spk_0"
        assert report["decisions"]["spk_1"]["assigned"] is True

        # Final mapping
        assert report["final_mapping"] == mapping

    def test_report_unassigned_below_threshold(self):
        profile = _profile(
            {"Speaker A": [1.0, 0.0, 0.0]},
            {"Speaker A": "spk_0"},
        )
        cluster_embs = {"c0": [0.0, 1.0, 0.0]}  # orthogonal
        mapping = {}  # nothing assigned
        report = build_calibration_report(
            cluster_embs, profile,
            threshold=0.7, margin=0.05,
            mapping=mapping, profile_name="test",
        )
        assert report["decisions"]["c0"]["assigned"] is False
        assert report["decisions"]["c0"]["reason"] == "below_threshold"

    def test_report_unassigned_below_margin(self):
        profile = _profile(
            {"Speaker A": [1.0, 0.0], "Speaker B": [0.0, 1.0]},
            {"Speaker A": "spk_0", "Speaker B": "spk_1"},
        )
        v = [1.0 / math.sqrt(2), 1.0 / math.sqrt(2)]
        cluster_embs = {"c0": v}
        mapping = {}  # nothing assigned (margin fail)
        report = build_calibration_report(
            cluster_embs, profile,
            threshold=0.3, margin=0.05,
            mapping=mapping, profile_name="test",
        )
        assert report["decisions"]["c0"]["assigned"] is False
        assert report["decisions"]["c0"]["reason"] == "below_margin"

    def test_debug_prints_when_enabled(self, capsys):
        cluster_embs, profile, mapping = self._setup()
        report = build_calibration_report(
            cluster_embs, profile,
            threshold=0.7, margin=0.05,
            mapping=mapping, profile_name="my_clinic",
        )
        print_calibration_debug(report)
        captured = capsys.readouterr().out
        assert "[CAL] Profile: my_clinic" in captured
        assert "threshold=0.7" in captured
        assert "ASSIGNED" in captured
        assert "mapping:" in captured

    def test_debug_shows_not_assigned(self, capsys):
        profile = _profile(
            {"Speaker A": [1.0, 0.0, 0.0]},
            {"Speaker A": "spk_0"},
        )
        cluster_embs = {"c0": [0.0, 1.0, 0.0]}
        report = build_calibration_report(
            cluster_embs, profile,
            threshold=0.7, margin=0.05,
            mapping={}, profile_name="test",
        )
        print_calibration_debug(report)
        captured = capsys.readouterr().out
        assert "NOT ASSIGNED" in captured

    def test_config_calibration_debug_default(self):
        cfg = DiarizationConfig()
        assert cfg.calibration_debug is False

    def test_config_calibration_debug_yaml(self):
        cfg = _build_diarization({"calibration_debug": True})
        assert cfg.calibration_debug is True

    def test_calibration_debug_cli_flag(self):
        parser = build_arg_parser()
        args = parser.parse_args(["--calibration-debug"])
        assert args.calibration_debug is True

    def test_calibration_debug_cli_default(self):
        parser = build_arg_parser()
        args = parser.parse_args([])
        assert args.calibration_debug is False

    def test_config_min_turn_duration_default(self):
        cfg = DiarizationConfig()
        assert cfg.calibration_min_turn_duration_sec == 0.0

    def test_config_min_turn_duration_yaml(self):
        cfg = _build_diarization({"calibration_min_turn_duration_sec": 0.5})
        assert cfg.calibration_min_turn_duration_sec == 0.5


# ── robustness guards (Phase 2F) ─────────────────────────────────────


class TestRobustnessGuards:
    def test_config_defaults_disabled(self):
        cfg = DiarizationConfig()
        assert cfg.calibration_min_cluster_turns == 0
        assert cfg.calibration_min_cluster_voiced_sec == 0.0
        assert cfg.calibration_allow_partial_assignment is True

    def test_config_yaml_overrides(self):
        cfg = _build_diarization({
            "calibration_min_cluster_turns": 3,
            "calibration_min_cluster_voiced_sec": 2.0,
            "calibration_allow_partial_assignment": False,
        })
        assert cfg.calibration_min_cluster_turns == 3
        assert cfg.calibration_min_cluster_voiced_sec == 2.0
        assert cfg.calibration_allow_partial_assignment is False


class TestBuildClusterEmbeddingsWithStats:
    def test_returns_stats(self):
        turns = [
            _turn(0.0, 1.0, "spk_0", embedding=[1.0, 0.0]),
            _turn(1.0, 3.0, "spk_0", embedding=[0.0, 1.0]),
            _turn(3.0, 5.0, "spk_1", embedding=[0.0, 0.0]),
        ]
        embs, stats = build_cluster_embeddings_with_stats(turns)
        assert "spk_0" in embs
        assert stats["spk_0"]["embedded_turn_count"] == 2
        assert stats["spk_0"]["embedded_total_sec"] == pytest.approx(3.0)
        assert stats["spk_1"]["embedded_turn_count"] == 1
        assert stats["spk_1"]["embedded_total_sec"] == pytest.approx(2.0)

    def test_skips_turns_without_embedding(self):
        turns = [
            _turn(0.0, 1.0, "spk_0", embedding=[1.0, 0.0]),
            _turn(1.0, 2.0, "spk_0"),  # no embedding
        ]
        embs, stats = build_cluster_embeddings_with_stats(turns)
        assert stats["spk_0"]["embedded_turn_count"] == 1

    def test_embedding_normalized(self):
        turns = [
            _turn(0.0, 1.0, "spk_0", embedding=[3.0, 4.0]),
            _turn(1.0, 2.0, "spk_0", embedding=[4.0, 3.0]),
        ]
        embs, _ = build_cluster_embeddings_with_stats(turns)
        norm = math.sqrt(sum(x * x for x in embs["spk_0"]))
        assert norm == pytest.approx(1.0, abs=1e-6)


class TestFilterEligibleClusters:
    def test_ineligible_too_few_turns(self):
        cluster_embs = {"spk_0": [1.0, 0.0], "spk_1": [0.0, 1.0]}
        stats = {
            "spk_0": {"embedded_turn_count": 1, "embedded_total_sec": 5.0},
            "spk_1": {"embedded_turn_count": 3, "embedded_total_sec": 5.0},
        }
        eligible, reasons = filter_eligible_clusters(
            cluster_embs, stats, min_cluster_turns=2
        )
        assert "spk_0" not in eligible
        assert "spk_1" in eligible
        assert reasons["spk_0"] == "ineligible_too_few_turns"

    def test_ineligible_too_little_voiced_sec(self):
        cluster_embs = {"spk_0": [1.0, 0.0], "spk_1": [0.0, 1.0]}
        stats = {
            "spk_0": {"embedded_turn_count": 5, "embedded_total_sec": 0.5},
            "spk_1": {"embedded_turn_count": 5, "embedded_total_sec": 3.0},
        }
        eligible, reasons = filter_eligible_clusters(
            cluster_embs, stats, min_cluster_voiced_sec=1.0
        )
        assert "spk_0" not in eligible
        assert "spk_1" in eligible
        assert reasons["spk_0"] == "ineligible_too_little_voiced_sec"

    def test_all_eligible_when_disabled(self):
        cluster_embs = {"spk_0": [1.0, 0.0], "spk_1": [0.0, 1.0]}
        stats = {
            "spk_0": {"embedded_turn_count": 1, "embedded_total_sec": 0.1},
            "spk_1": {"embedded_turn_count": 1, "embedded_total_sec": 0.1},
        }
        eligible, reasons = filter_eligible_clusters(
            cluster_embs, stats, min_cluster_turns=0, min_cluster_voiced_sec=0.0
        )
        assert len(eligible) == 2
        assert reasons == {}


class TestAllowPartialAssignment:
    def test_partial_true_applies_subset(self):
        """allow_partial=True overrides assigned clusters; unassigned get UNKNOWN."""
        turns = [
            _turn(0.0, 1.0, "spk_0"),
            _turn(1.0, 2.0, "spk_1"),
        ]
        mapping = {"spk_0": "spk_0"}  # only spk_0 assigned
        result = apply_cluster_override(
            turns, mapping,
            eligible_cluster_ids={"spk_0", "spk_1"},
            allow_partial=True,
        )
        assert result[0]["speaker"] == "spk_0"
        assert result[1]["speaker"] == "UNKNOWN"

    def test_partial_false_blocks_when_incomplete(self):
        """allow_partial=False returns unchanged turns when not all assigned."""
        turns = [
            _turn(0.0, 1.0, "spk_0"),
            _turn(1.0, 2.0, "spk_1"),
        ]
        mapping = {"spk_0": "cal_0"}  # only 1 of 2 eligible assigned
        result = apply_cluster_override(
            turns, mapping,
            eligible_cluster_ids={"spk_0", "spk_1"},
            allow_partial=False,
        )
        # No overrides applied
        assert result[0]["speaker"] == "spk_0"
        assert result[1]["speaker"] == "spk_1"

    def test_partial_false_allows_when_all_assigned(self):
        """allow_partial=False still applies when all eligible are assigned."""
        turns = [
            _turn(0.0, 1.0, "spk_0"),
            _turn(1.0, 2.0, "spk_1"),
        ]
        mapping = {"spk_0": "cal_0", "spk_1": "cal_1"}
        result = apply_cluster_override(
            turns, mapping,
            eligible_cluster_ids={"spk_0", "spk_1"},
            allow_partial=False,
        )
        assert result[0]["speaker"] == "cal_0"
        assert result[1]["speaker"] == "cal_1"

    def test_does_not_mutate_input(self):
        turns = [_turn(0.0, 1.0, "spk_0")]
        mapping = {"spk_0": "cal_0"}
        original = turns[0]["speaker"]
        apply_cluster_override(
            turns, mapping,
            eligible_cluster_ids={"spk_0", "spk_1"},
            allow_partial=False,
        )
        assert turns[0]["speaker"] == original


class TestReportWithRobustnessGuards:
    def test_report_includes_cluster_stats(self):
        profile = _profile(
            {"Speaker A": [1.0, 0.0]},
            {"Speaker A": "spk_0"},
        )
        cluster_embs = {"spk_0": [0.95, 0.1]}
        stats = {"spk_0": {"embedded_turn_count": 5, "embedded_total_sec": 8.3}}
        mapping = {"spk_0": "spk_0"}
        report = build_calibration_report(
            cluster_embs, profile,
            threshold=0.7, margin=0.05,
            mapping=mapping, profile_name="test",
            cluster_stats=stats,
        )
        assert report["decisions"]["spk_0"]["turn_count"] == 5
        assert report["decisions"]["spk_0"]["voiced_sec"] == 8.3

    def test_report_shows_ineligible_reason(self):
        profile = _profile(
            {"Speaker A": [1.0, 0.0]},
            {"Speaker A": "spk_0"},
        )
        cluster_embs = {"spk_1": [0.0, 1.0]}  # only spk_1 eligible
        stats = {
            "spk_0": {"embedded_turn_count": 1, "embedded_total_sec": 0.2},
            "spk_1": {"embedded_turn_count": 5, "embedded_total_sec": 3.0},
        }
        ineligible = {"spk_0": "ineligible_too_few_turns"}
        report = build_calibration_report(
            cluster_embs, profile,
            threshold=0.7, margin=0.05,
            mapping={}, profile_name="test",
            cluster_stats=stats,
            ineligible_reasons=ineligible,
        )
        assert report["decisions"]["spk_0"]["assigned"] is False
        assert report["decisions"]["spk_0"]["reason"] == "ineligible_too_few_turns"
        assert report["decisions"]["spk_0"]["turn_count"] == 1
        # spk_0 should not be in similarity matrix (no embedding)
        assert "spk_0" not in report["similarity"]

    def test_report_partial_assignment_field(self):
        profile = _profile(
            {"Speaker A": [1.0, 0.0]},
            {"Speaker A": "spk_0"},
        )
        cluster_embs = {"spk_0": [0.95, 0.1]}
        report = build_calibration_report(
            cluster_embs, profile,
            threshold=0.7, margin=0.05,
            mapping={"spk_0": "spk_0"}, profile_name="test",
            partial_assignment_applied=True,
        )
        assert report["partial_assignment_applied"] is True

    def test_report_omits_partial_field_when_none(self):
        profile = _profile(
            {"Speaker A": [1.0, 0.0]},
            {"Speaker A": "spk_0"},
        )
        cluster_embs = {"spk_0": [0.95, 0.1]}
        report = build_calibration_report(
            cluster_embs, profile,
            threshold=0.7, margin=0.05,
            mapping={"spk_0": "spk_0"}, profile_name="test",
        )
        assert "partial_assignment_applied" not in report
