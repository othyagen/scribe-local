"""Tests for the case variation generator."""

from __future__ import annotations

import copy

import pytest

from app.case_variations import (
    apply_variation,
    generate_case_variations,
    list_supported_variations,
    summarize_case_variation,
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
        "description": "A richer test case.",
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
        "config": {
            "mode": "assist",
            "update_strategy": "manual",
            "show_questions": True,
        },
        "ground_truth": {
            "expected_hypotheses": ["Pneumonia"],
            "red_flags": [],
            "key_findings": ["cough", "fever", "shortness of breath"],
        },
        "answer_script": [
            {"question_type": "duration", "value": "3 days", "related": "cough"},
        ],
        "meta": {
            "tags": ["respiratory", "infection"],
            "difficulty": "easy",
            "source": "synthetic",
            "notes": "Test case.",
        },
    }


# ── supported variations ─────────────────────────────────────────────


class TestListSupportedVariations:
    def test_returns_list(self):
        result = list_supported_variations()
        assert isinstance(result, list)

    def test_non_empty(self):
        assert len(list_supported_variations()) >= 6

    def test_all_strings(self):
        for name in list_supported_variations():
            assert isinstance(name, str)

    def test_sorted(self):
        names = list_supported_variations()
        assert names == sorted(names)

    def test_expected_names(self):
        names = set(list_supported_variations())
        assert "remove_fever" in names
        assert "add_duration_longer" in names
        assert "add_elderly_context" in names
        assert "add_negation_of_core_symptom" in names
        assert "add_missing_information" in names
        assert "add_conflicting_information" in names


# ── apply_variation general ──────────────────────────────────────────


class TestApplyVariation:
    def test_unknown_variation_raises(self):
        with pytest.raises(ValueError, match="Unknown variation"):
            apply_variation(_minimal_case(), "nonexistent")

    def test_case_id_format(self):
        result = apply_variation(_minimal_case(), "remove_fever")
        assert result["case_id"] == "test_01__remove_fever"

    def test_meta_base_case_id(self):
        result = apply_variation(_minimal_case(), "remove_fever")
        assert result["meta"]["base_case_id"] == "test_01"

    def test_meta_applied_variation(self):
        result = apply_variation(_minimal_case(), "remove_fever")
        assert result["meta"]["applied_variation"] == "remove_fever"

    def test_does_not_mutate_input(self):
        case = _rich_case()
        original = copy.deepcopy(case)
        for name in list_supported_variations():
            apply_variation(case, name)
        assert case == original

    def test_all_variations_return_valid_cases(self):
        case = _rich_case()
        for name in list_supported_variations():
            result = apply_variation(case, name)
            validation = validate_case(result)
            assert validation["valid"] is True, (
                f"Variation {name!r} produced invalid case: {validation['errors']}"
            )

    def test_all_variations_preserve_required_fields(self):
        case = _rich_case()
        for name in list_supported_variations():
            result = apply_variation(case, name)
            assert "case_id" in result
            assert "segments" in result
            assert isinstance(result["segments"], list)
            assert len(result["segments"]) >= 1

    def test_deterministic(self):
        case = _rich_case()
        for name in list_supported_variations():
            r1 = apply_variation(case, name)
            r2 = apply_variation(case, name)
            assert r1 == r2, f"Variation {name!r} is not deterministic"


# ── individual variations ────────────────────────────────────────────


class TestRemoveFever:
    def test_removes_fever_from_text(self):
        result = apply_variation(_minimal_case(), "remove_fever")
        for seg in result["segments"]:
            assert "fever" not in seg["normalized_text"].lower()

    def test_removes_fever_from_ground_truth(self):
        case = _rich_case()
        result = apply_variation(case, "remove_fever")
        findings = result.get("ground_truth", {}).get("key_findings", [])
        assert "fever" not in [f.lower() for f in findings]

    def test_warning_when_no_fever(self):
        case = _minimal_case(segments=[{
            "seg_id": "seg_0001", "t0": 0.0, "t1": 3.0,
            "speaker_id": "spk_0",
            "normalized_text": "Patient has headache.",
        }])
        result = apply_variation(case, "remove_fever")
        assert result["meta"].get("variation_warning")

    def test_preserves_other_text(self):
        result = apply_variation(_minimal_case(), "remove_fever")
        text = result["segments"][0]["normalized_text"]
        assert "headache" in text


class TestAddDurationLonger:
    def test_replaces_short_duration(self):
        result = apply_variation(_minimal_case(), "add_duration_longer")
        full_text = " ".join(
            s["normalized_text"] for s in result["segments"]
        )
        assert "2 weeks" in full_text

    def test_appends_segment_when_no_duration(self):
        case = _minimal_case(segments=[{
            "seg_id": "seg_0001", "t0": 0.0, "t1": 3.0,
            "speaker_id": "spk_0",
            "normalized_text": "Patient has headache.",
        }])
        result = apply_variation(case, "add_duration_longer")
        assert len(result["segments"]) == 2

    def test_updates_answer_script_duration(self):
        case = _rich_case()
        result = apply_variation(case, "add_duration_longer")
        for entry in result.get("answer_script", []):
            if entry.get("question_type") == "duration":
                assert entry["value"] == "2 weeks"


class TestAddElderlyContext:
    def test_replaces_existing_age(self):
        case = _rich_case()
        result = apply_variation(case, "add_elderly_context")
        text = result["segments"][0]["normalized_text"]
        assert "82-year-old" in text

    def test_prepends_segment_when_no_age(self):
        case = _minimal_case()
        result = apply_variation(case, "add_elderly_context")
        assert len(result["segments"]) == 2
        assert "82-year-old" in result["segments"][0]["normalized_text"]

    def test_adds_elderly_tag(self):
        result = apply_variation(_rich_case(), "add_elderly_context")
        tags = result.get("meta", {}).get("tags", [])
        assert "elderly" in tags

    def test_no_duplicate_tag(self):
        case = _minimal_case(meta={"tags": ["elderly"]})
        result = apply_variation(case, "add_elderly_context")
        assert result["meta"]["tags"].count("elderly") == 1


class TestAddNegation:
    def test_appends_negation_segment(self):
        result = apply_variation(_minimal_case(), "add_negation_of_core_symptom")
        texts = [s["normalized_text"] for s in result["segments"]]
        assert any("denies" in t.lower() for t in texts)

    def test_warning_when_no_symptom(self):
        case = _minimal_case(segments=[{
            "seg_id": "seg_0001", "t0": 0.0, "t1": 3.0,
            "speaker_id": "spk_0",
            "normalized_text": "Hello doctor.",
        }])
        result = apply_variation(case, "add_negation_of_core_symptom")
        assert result["meta"].get("variation_warning")


class TestAddMissingInformation:
    def test_removes_segment(self):
        case = _rich_case()
        original_count = len(case["segments"])
        result = apply_variation(case, "add_missing_information")
        assert len(result["segments"]) < original_count

    def test_warning_with_single_segment(self):
        case = _minimal_case()
        result = apply_variation(case, "add_missing_information")
        assert result["meta"].get("variation_warning")

    def test_preserves_at_least_one_segment(self):
        result = apply_variation(_rich_case(), "add_missing_information")
        assert len(result["segments"]) >= 1


class TestAddConflictingInformation:
    def test_appends_segment(self):
        case = _rich_case()
        original_count = len(case["segments"])
        result = apply_variation(case, "add_conflicting_information")
        assert len(result["segments"]) == original_count + 1

    def test_contradiction_for_fever(self):
        result = apply_variation(_minimal_case(), "add_conflicting_information")
        last = result["segments"][-1]["normalized_text"]
        assert "fever" in last.lower() or "fine" in last.lower()

    def test_fallback_when_no_keywords(self):
        case = _minimal_case(segments=[{
            "seg_id": "seg_0001", "t0": 0.0, "t1": 3.0,
            "speaker_id": "spk_0",
            "normalized_text": "Hello doctor.",
        }])
        result = apply_variation(case, "add_conflicting_information")
        assert len(result["segments"]) == 2


# ── generate_case_variations ─────────────────────────────────────────


class TestGenerateCaseVariations:
    def test_returns_list(self):
        result = generate_case_variations(_minimal_case())
        assert isinstance(result, list)

    def test_one_per_variation(self):
        result = generate_case_variations(_minimal_case())
        assert len(result) == len(list_supported_variations())

    def test_all_valid(self):
        for var in generate_case_variations(_rich_case()):
            validation = validate_case(var)
            assert validation["valid"] is True, (
                f"{var['case_id']}: {validation['errors']}"
            )

    def test_unique_case_ids(self):
        results = generate_case_variations(_minimal_case())
        ids = [r["case_id"] for r in results]
        assert len(ids) == len(set(ids))

    def test_does_not_mutate_input(self):
        case = _rich_case()
        original = copy.deepcopy(case)
        generate_case_variations(case)
        assert case == original

    def test_deterministic(self):
        case = _rich_case()
        r1 = generate_case_variations(case)
        r2 = generate_case_variations(case)
        assert r1 == r2


# ── summarize_case_variation ─────────────────────────────────────────


class TestSummarizeCaseVariation:
    def test_original_case(self):
        summary = summarize_case_variation(_rich_case())
        assert summary["case_id"] == "rich_01"
        assert summary["base_case_id"] == ""
        assert summary["applied_variation"] == ""
        assert summary["segment_count"] == 3

    def test_variation_case(self):
        var = apply_variation(_rich_case(), "remove_fever")
        summary = summarize_case_variation(var)
        assert summary["case_id"] == "rich_01__remove_fever"
        assert summary["base_case_id"] == "rich_01"
        assert summary["applied_variation"] == "remove_fever"

    def test_tags_preserved(self):
        var = apply_variation(_rich_case(), "add_elderly_context")
        summary = summarize_case_variation(var)
        assert "elderly" in summary["tags"]

    def test_minimal_case(self):
        summary = summarize_case_variation({"case_id": "x", "segments": []})
        assert summary["segment_count"] == 0
        assert summary["tags"] == []


# ── tolerance of minimal / missing fields ────────────────────────────


class TestTolerance:
    def test_minimal_case_all_variations(self):
        case = {"case_id": "bare", "segments": [{
            "seg_id": "seg_0001", "t0": 0.0, "t1": 3.0,
            "speaker_id": "spk_0",
            "normalized_text": "Patient feels unwell.",
        }]}
        for name in list_supported_variations():
            result = apply_variation(case, name)
            assert result["case_id"].startswith("bare__")
            assert len(result["segments"]) >= 1

    def test_no_meta_field(self):
        case = _minimal_case()
        case.pop("meta", None)
        for name in list_supported_variations():
            result = apply_variation(case, name)
            assert "meta" in result
            assert result["meta"]["base_case_id"] == "test_01"

    def test_no_ground_truth(self):
        case = _minimal_case()
        result = apply_variation(case, "remove_fever")
        # Should not crash even without ground_truth.
        assert result["case_id"] == "test_01__remove_fever"

    def test_no_answer_script(self):
        case = _minimal_case()
        result = apply_variation(case, "add_duration_longer")
        assert result["case_id"] == "test_01__add_duration_longer"
