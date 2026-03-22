"""Tests for app.evaluation_slicing."""

import pytest

from app.evaluation_slicing import slice_evaluation


def _make_entry(case: dict, hit_rate: float) -> dict:
    return {
        "case": case,
        "score": {"summary": {"hypothesis_hit_rate": hit_rate}},
    }


def _case(organ_systems=None, presenting_complaints=None,
          difficulty=None, origin=None, tags=None):
    c = {"case_id": "test", "segments": []}
    if organ_systems is not None or presenting_complaints is not None:
        c["classification"] = {}
        if organ_systems is not None:
            c["classification"]["organ_systems"] = organ_systems
        if presenting_complaints is not None:
            c["classification"]["presenting_complaints"] = presenting_complaints
    if difficulty is not None or tags is not None or origin is not None:
        c["meta"] = {}
        if difficulty is not None:
            c["meta"]["difficulty"] = difficulty
        if tags is not None:
            c["meta"]["tags"] = tags
    if origin is not None:
        c["provenance"] = {"origin": origin}
    return c


class TestSliceByOrganSystem:
    def test_single_group(self):
        results = [
            _make_entry(_case(organ_systems=["respiratory"]), 0.8),
            _make_entry(_case(organ_systems=["respiratory"]), 0.6),
        ]
        out = slice_evaluation(results, "organ_system")
        assert out == {"respiratory": pytest.approx(0.7)}

    def test_multiple_groups(self):
        results = [
            _make_entry(_case(organ_systems=["respiratory"]), 1.0),
            _make_entry(_case(organ_systems=["cardiovascular"]), 0.5),
        ]
        out = slice_evaluation(results, "organ_system")
        assert out == {"respiratory": 1.0, "cardiovascular": 0.5}

    def test_multi_valued_contributes_to_each(self):
        results = [
            _make_entry(_case(organ_systems=["respiratory", "cardiovascular"]), 0.8),
        ]
        out = slice_evaluation(results, "organ_system")
        assert out == {"respiratory": 0.8, "cardiovascular": 0.8}


class TestSliceByDifficulty:
    def test_groups(self):
        results = [
            _make_entry(_case(difficulty="easy"), 1.0),
            _make_entry(_case(difficulty="easy"), 0.5),
            _make_entry(_case(difficulty="hard"), 0.2),
        ]
        out = slice_evaluation(results, "difficulty")
        assert out == {"easy": pytest.approx(0.75), "hard": pytest.approx(0.2)}


class TestSliceByPresentingComplaint:
    def test_list_field(self):
        results = [
            _make_entry(_case(presenting_complaints=["cough", "fever"]), 0.6),
            _make_entry(_case(presenting_complaints=["fever"]), 1.0),
        ]
        out = slice_evaluation(results, "presenting_complaint")
        assert out["cough"] == pytest.approx(0.6)
        assert out["fever"] == pytest.approx(0.8)


class TestSliceByTag:
    def test_list_field(self):
        results = [
            _make_entry(_case(tags=["infection"]), 0.9),
            _make_entry(_case(tags=["infection", "acute"]), 0.5),
        ]
        out = slice_evaluation(results, "tag")
        assert out["infection"] == pytest.approx(0.7)
        assert out["acute"] == pytest.approx(0.5)


class TestSliceByOrigin:
    def test_scalar(self):
        results = [
            _make_entry(_case(origin="synthetic"), 0.8),
            _make_entry(_case(origin="synthea"), 0.4),
        ]
        out = slice_evaluation(results, "origin")
        assert out == {"synthetic": 0.8, "synthea": 0.4}


class TestMissingField:
    def test_missing_organ_system_unknown(self):
        results = [_make_entry(_case(), 0.5)]
        out = slice_evaluation(results, "organ_system")
        assert out == {"unknown": 0.5}

    def test_missing_difficulty_unknown(self):
        results = [_make_entry(_case(), 0.5)]
        out = slice_evaluation(results, "difficulty")
        assert "unspecified" in out or "unknown" in out


class TestEdgeCases:
    def test_empty_results(self):
        assert slice_evaluation([], "organ_system") == {}

    def test_unknown_key_raises(self):
        with pytest.raises(ValueError, match="unknown slicing key"):
            slice_evaluation([], "bogus")

    def test_deterministic(self):
        results = [
            _make_entry(_case(organ_systems=["a", "b"]), 0.5),
            _make_entry(_case(organ_systems=["b", "c"]), 1.0),
        ]
        out1 = slice_evaluation(results, "organ_system")
        out2 = slice_evaluation(results, "organ_system")
        assert out1 == out2
