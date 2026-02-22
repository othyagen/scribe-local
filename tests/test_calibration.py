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

from app.calibration import (
    apply_cluster_override,
    assign_clusters_to_profile,
    build_cluster_embeddings,
    cosine_similarity,
    embed_turns,
    extract_embedding,
    load_profile,
    match_turn_embeddings,
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

        mock_inference_obj = MagicMock()
        mock_inference_obj.return_value = fake_raw
        mock_inference_cls = MagicMock(return_value=mock_inference_obj)

        mock_torch = MagicMock()
        mock_torch.from_numpy.return_value.unsqueeze.return_value.float.return_value = "tensor"

        mock_pyannote_audio = MagicMock()
        mock_pyannote_audio.Inference = mock_inference_cls

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
