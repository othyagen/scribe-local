"""Tests for calibration profiles and embedding matching."""

from __future__ import annotations

import json
import math

import pytest

from app.calibration import (
    cosine_similarity,
    load_profile,
    match_turn_embeddings,
    save_profile,
)
from app.config import DiarizationConfig, _build_diarization


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
