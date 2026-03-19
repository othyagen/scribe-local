"""Tests for the adversarial case generation layer."""

from __future__ import annotations

import copy

import pytest

from app.case_adversarial import (
    apply_adversarial,
    generate_adversarial_cases,
    list_adversarial_strategies,
)
from app.case_system import validate_case


# ── helpers ──────────────────────────────────────────────────────────


def _minimal_case(**overrides) -> dict:
    base = {
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
    }
    base.update(overrides)
    return base


def _rich_case() -> dict:
    return {
        "case_id": "rich_01",
        "title": "Rich case",
        "segments": [
            {
                "seg_id": "seg_0001",
                "t0": 0.0,
                "t1": 3.0,
                "speaker_id": "spk_0",
                "normalized_text": "65-year-old male with cough and fever for 3 days.",
            },
            {
                "seg_id": "seg_0002",
                "t0": 3.0,
                "t1": 6.0,
                "speaker_id": "spk_0",
                "normalized_text": "Feels more short of breath today.",
            },
            {
                "seg_id": "seg_0003",
                "t0": 6.0,
                "t1": 9.0,
                "speaker_id": "spk_0",
                "normalized_text": "Denies chest pain. No nausea.",
            },
        ],
        "config": {"mode": "assist", "update_strategy": "manual"},
        "ground_truth": {
            "expected_hypotheses": ["Pneumonia"],
            "red_flags": [],
            "key_findings": ["cough", "fever", "shortness of breath"],
        },
        "meta": {
            "tags": ["respiratory"],
            "difficulty": "easy",
            "source": "synthetic",
        },
    }


# ── strategy listing ─────────────────────────────────────────────────


class TestListStrategies:
    def test_returns_list(self):
        assert isinstance(list_adversarial_strategies(), list)

    def test_has_expected_strategies(self):
        names = set(list_adversarial_strategies())
        assert "noise_injection" in names
        assert "contradiction_injection" in names
        assert "negation_flip" in names
        assert "temporal_confusion" in names
        assert "symptom_dilution" in names
        assert "incomplete_case" in names

    def test_sorted(self):
        names = list_adversarial_strategies()
        assert names == sorted(names)

    def test_count(self):
        assert len(list_adversarial_strategies()) == 6


# ── apply_adversarial general ────────────────────────────────────────


class TestApplyAdversarial:
    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown adversarial"):
            apply_adversarial(_minimal_case(), "nonexistent")

    def test_case_id_suffix(self):
        result = apply_adversarial(_minimal_case(), "noise_injection")
        assert result["case_id"] == "test_01__adv_noise_injection"

    def test_meta_base_case_id(self):
        result = apply_adversarial(_minimal_case(), "noise_injection")
        assert result["meta"]["base_case_id"] == "test_01"

    def test_meta_applied_variation(self):
        result = apply_adversarial(_minimal_case(), "noise_injection")
        assert result["meta"]["applied_variation"] == "adv_noise_injection"

    def test_adversarial_metadata(self):
        result = apply_adversarial(_minimal_case(), "noise_injection")
        assert result["adversarial"]["strategy"] == "noise_injection"

    def test_does_not_mutate_input(self):
        case = _rich_case()
        original = copy.deepcopy(case)
        for name in list_adversarial_strategies():
            apply_adversarial(case, name)
        assert case == original

    def test_all_strategies_return_valid_cases(self):
        case = _rich_case()
        for name in list_adversarial_strategies():
            result = apply_adversarial(case, name)
            validation = validate_case(result)
            assert validation["valid"] is True, (
                f"Strategy {name!r} produced invalid case: {validation['errors']}"
            )

    def test_all_strategies_preserve_required_fields(self):
        case = _rich_case()
        for name in list_adversarial_strategies():
            result = apply_adversarial(case, name)
            assert "case_id" in result
            assert "segments" in result
            assert isinstance(result["segments"], list)
            assert len(result["segments"]) >= 1

    def test_deterministic(self):
        case = _rich_case()
        for name in list_adversarial_strategies():
            r1 = apply_adversarial(case, name)
            r2 = apply_adversarial(case, name)
            assert r1 == r2, f"Strategy {name!r} is not deterministic"


# ── individual strategies ────────────────────────────────────────────


class TestNoiseInjection:
    def test_adds_segments(self):
        case = _rich_case()
        original_count = len(case["segments"])
        result = apply_adversarial(case, "noise_injection")
        assert len(result["segments"]) > original_count

    def test_preserves_original_segments(self):
        case = _rich_case()
        result = apply_adversarial(case, "noise_injection")
        original_texts = [s["normalized_text"] for s in case["segments"]]
        result_texts = [s["normalized_text"] for s in result["segments"]]
        for t in original_texts:
            assert t in result_texts


class TestContradictionInjection:
    def test_adds_contradiction(self):
        case = _rich_case()
        original_count = len(case["segments"])
        result = apply_adversarial(case, "contradiction_injection")
        assert len(result["segments"]) > original_count

    def test_generic_fallback(self):
        case = _minimal_case(segments=[{
            "seg_id": "seg_0001", "t0": 0.0, "t1": 3.0,
            "speaker_id": "spk_0",
            "normalized_text": "Patient feels unwell.",
        }])
        result = apply_adversarial(case, "contradiction_injection")
        assert len(result["segments"]) == 2


class TestNegationFlip:
    def test_adds_negation(self):
        result = apply_adversarial(_minimal_case(), "negation_flip")
        texts = [s["normalized_text"].lower() for s in result["segments"]]
        assert any("denies" in t for t in texts)

    def test_warning_when_no_symptoms(self):
        case = _minimal_case(segments=[{
            "seg_id": "seg_0001", "t0": 0.0, "t1": 3.0,
            "speaker_id": "spk_0",
            "normalized_text": "Hello doctor.",
        }])
        result = apply_adversarial(case, "negation_flip")
        assert result["meta"].get("variation_warning")

    def test_negates_up_to_two(self):
        case = _rich_case()
        original_ids = {s["seg_id"] for s in case["segments"]}
        result = apply_adversarial(case, "negation_flip")
        added_negations = [
            s for s in result["segments"]
            if s["seg_id"] not in original_ids
            and s["normalized_text"].lower().startswith("denies")
        ]
        assert 1 <= len(added_negations) <= 2


class TestTemporalConfusion:
    def test_adds_conflicting_times(self):
        case = _minimal_case()
        original_count = len(case["segments"])
        result = apply_adversarial(case, "temporal_confusion")
        assert len(result["segments"]) == original_count + 2

    def test_contains_time_references(self):
        result = apply_adversarial(_minimal_case(), "temporal_confusion")
        full_text = " ".join(
            s["normalized_text"] for s in result["segments"]
        ).lower()
        assert "morning" in full_text or "today" in full_text
        assert "weeks" in full_text


class TestSymptomDilution:
    def test_adds_many_segments(self):
        case = _minimal_case()
        original_count = len(case["segments"])
        result = apply_adversarial(case, "symptom_dilution")
        assert len(result["segments"]) >= original_count + 4

    def test_low_specificity_symptoms(self):
        result = apply_adversarial(_minimal_case(), "symptom_dilution")
        full_text = " ".join(
            s["normalized_text"] for s in result["segments"]
        ).lower()
        assert "tired" in full_text or "fatigue" in full_text
        assert "dizziness" in full_text or "lightheadedness" in full_text


class TestIncompleteCase:
    def test_removes_segment(self):
        case = _rich_case()
        original_count = len(case["segments"])
        result = apply_adversarial(case, "incomplete_case")
        assert len(result["segments"]) == original_count - 1

    def test_warning_with_single_segment(self):
        case = _minimal_case()
        result = apply_adversarial(case, "incomplete_case")
        assert result["meta"].get("variation_warning")
        assert len(result["segments"]) == 1

    def test_preserves_at_least_one(self):
        result = apply_adversarial(_rich_case(), "incomplete_case")
        assert len(result["segments"]) >= 1


# ── generate_adversarial_cases ───────────────────────────────────────


class TestGenerateAdversarialCases:
    def test_returns_list(self):
        assert isinstance(generate_adversarial_cases(_minimal_case()), list)

    def test_one_per_strategy(self):
        result = generate_adversarial_cases(_minimal_case())
        assert len(result) == len(list_adversarial_strategies())

    def test_all_valid(self):
        for var in generate_adversarial_cases(_rich_case()):
            validation = validate_case(var)
            assert validation["valid"] is True, (
                f"{var['case_id']}: {validation['errors']}"
            )

    def test_unique_case_ids(self):
        results = generate_adversarial_cases(_minimal_case())
        ids = [r["case_id"] for r in results]
        assert len(ids) == len(set(ids))

    def test_does_not_mutate_input(self):
        case = _rich_case()
        original = copy.deepcopy(case)
        generate_adversarial_cases(case)
        assert case == original

    def test_deterministic(self):
        case = _rich_case()
        r1 = generate_adversarial_cases(case)
        r2 = generate_adversarial_cases(case)
        assert r1 == r2

    def test_all_have_adversarial_metadata(self):
        for var in generate_adversarial_cases(_minimal_case()):
            assert "adversarial" in var
            assert "strategy" in var["adversarial"]


# ── tolerance ────────────────────────────────────────────────────────


class TestTolerance:
    def test_minimal_case_all_strategies(self):
        case = {"case_id": "bare", "segments": [{
            "seg_id": "seg_0001", "t0": 0.0, "t1": 3.0,
            "speaker_id": "spk_0",
            "normalized_text": "Patient feels unwell.",
        }]}
        for name in list_adversarial_strategies():
            result = apply_adversarial(case, name)
            assert result["case_id"].startswith("bare__adv_")
            assert len(result["segments"]) >= 1

    def test_no_meta_field(self):
        case = _minimal_case()
        case.pop("meta", None)
        for name in list_adversarial_strategies():
            result = apply_adversarial(case, name)
            assert "meta" in result
            assert result["meta"]["base_case_id"] == "test_01"

    def test_no_ground_truth(self):
        case = _minimal_case()
        for name in list_adversarial_strategies():
            result = apply_adversarial(case, name)
            assert "case_id" in result
