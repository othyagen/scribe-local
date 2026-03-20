"""Tests for deterministic clinical text extractors."""

from __future__ import annotations

import pytest

from app.extractors import (
    extract_symptoms,
    extract_negations,
    extract_durations,
    extract_medications,
    get_extractor,
    EXTRACTORS,
)


# ── symptom extraction ────────────────────────────────────────────────


class TestSymptomExtraction:
    def test_single_symptom(self):
        result = extract_symptoms("patient reports headache")
        assert result == ["headache"]

    def test_multiple_symptoms(self):
        result = extract_symptoms("nausea and vomiting with dizziness")
        assert "nausea" in result
        assert "vomiting" in result
        assert "dizziness" in result

    def test_no_symptoms(self):
        result = extract_symptoms("patient is doing well today")
        assert result == []

    def test_case_insensitive(self):
        result = extract_symptoms("Patient has HEADACHE and Nausea.")
        assert "headache" in result
        assert "nausea" in result

    def test_multiword_symptom(self):
        result = extract_symptoms("reports shortness of breath and chest pain")
        assert "dyspnea" in result  # canonicalised from "shortness of breath"
        assert "chest pain" in result

    def test_unique_results(self):
        result = extract_symptoms("headache and more headache, still headache")
        assert result.count("headache") == 1

    def test_deterministic(self):
        text = "fever, nausea, cough, and pain"
        r1 = extract_symptoms(text)
        r2 = extract_symptoms(text)
        assert r1 == r2

    def test_dysuria_direct(self):
        result = extract_symptoms("patient reports dysuria")
        assert result == ["dysuria"]

    def test_urinary_frequency_direct(self):
        result = extract_symptoms("reports urinary frequency")
        assert result == ["urinary frequency"]

    def test_painful_urination_canonicalised_to_dysuria(self):
        result = extract_symptoms("patient reports painful urination")
        assert "dysuria" in result
        assert "painful urination" not in result

    def test_synonym_deduplicates_with_canonical(self):
        """If text contains both 'painful urination' and 'dysuria', only one entry."""
        result = extract_symptoms("painful urination and dysuria noted")
        assert result.count("dysuria") == 1

    def test_painful_urination_does_not_extract_bare_pain(self):
        """'painful urination' should match as multi-word, not extract bare 'pain'."""
        result = extract_symptoms("patient reports painful urination")
        assert "pain" not in result

    def test_shortness_of_breath_canonicalised_to_dyspnea(self):
        result = extract_symptoms("patient has shortness of breath")
        assert "dyspnea" in result
        assert "shortness of breath" not in result

    def test_short_of_breath_canonicalised_to_dyspnea(self):
        result = extract_symptoms("feels more short of breath today")
        assert "dyspnea" in result

    def test_difficulty_breathing_canonicalised_to_dyspnea(self):
        result = extract_symptoms("patient has difficulty breathing")
        assert "dyspnea" in result

    def test_breathlessness_canonicalised_to_dyspnea(self):
        result = extract_symptoms("reports breathlessness")
        assert "dyspnea" in result

    def test_dyspnea_direct(self):
        result = extract_symptoms("patient presents with dyspnea")
        assert result == ["dyspnea"]

    def test_dyspnea_synonym_deduplicates(self):
        """Multiple synonym forms should produce single 'dyspnea'."""
        result = extract_symptoms("shortness of breath and dyspnea noted")
        assert result.count("dyspnea") == 1


# ── negation extraction ──────────────────────────────────────────────


class TestNegationExtraction:
    def test_denies_pattern(self):
        result = extract_negations("denies chest pain.")
        assert len(result) == 1
        assert result[0] == "Denies chest pain"

    def test_no_pattern(self):
        result = extract_negations("no fever.")
        assert len(result) == 1
        assert result[0] == "No fever"

    def test_without_pattern(self):
        result = extract_negations("without nausea.")
        assert len(result) == 1
        assert result[0] == "Without nausea"

    def test_no_negation(self):
        result = extract_negations("reports headache and fever")
        assert result == []

    def test_multiple_negations(self):
        result = extract_negations("no fever, denies nausea.")
        assert len(result) == 2
        # Should be clean bullet items
        bullets_lower = [b.lower() for b in result]
        assert any("fever" in b for b in bullets_lower)
        assert any("nausea" in b for b in bullets_lower)

    def test_negative_for_pattern(self):
        result = extract_negations("negative for strep.")
        assert len(result) == 1
        assert result[0] == "Negative for strep"

    def test_capitalized_trigger(self):
        """Trigger is always capitalized in output regardless of input case."""
        result = extract_negations("NO fever.")
        assert result[0].startswith("No ")  # Not "NO "

    def test_trailing_punctuation_stripped(self):
        """Output should not include trailing punctuation from match."""
        result = extract_negations("denies vomiting.")
        assert not result[0].endswith(".")

    def test_unique_results(self):
        result = extract_negations("no fever, no fever.")
        assert len(result) == 1

    def test_deterministic(self):
        text = "no fever, denies chest pain, without nausea."
        r1 = extract_negations(text)
        r2 = extract_negations(text)
        assert r1 == r2

    def test_readable_bullet_format(self):
        """Each negation is a clean, readable bullet item."""
        results = extract_negations(
            "no fever, denies chest pain, without nausea, "
            "negative for strep, denied headache."
        )
        for item in results:
            # Each item should start with a capitalized word
            assert item[0].isupper()
            # Should not end with punctuation
            assert not item.endswith((".", ",", ";", "!", "?"))
            # Should contain at least two words (trigger + phrase)
            assert " " in item


# ── duration extraction ──────────────────────────────────────────────


class TestDurationExtraction:
    def test_duration_days(self):
        result = extract_durations("symptoms for 3 days")
        assert result == ["3 days"]

    def test_multiple_durations(self):
        result = extract_durations("started 2 weeks ago, lasting 30 minutes")
        assert "2 weeks" in result
        assert "30 minutes" in result

    def test_no_duration(self):
        result = extract_durations("patient feels better")
        assert result == []

    def test_singular_unit(self):
        result = extract_durations("for 1 day and 1 hour")
        assert "1 day" in result
        assert "1 hour" in result

    def test_deterministic(self):
        text = "3 days, 2 weeks, 1 month"
        r1 = extract_durations(text)
        r2 = extract_durations(text)
        assert r1 == r2

    def test_spelled_out_days(self):
        result = extract_durations("chest pain for three days")
        assert result == ["three days"]

    def test_spelled_out_weeks(self):
        result = extract_durations("pain for two weeks now")
        assert result == ["two weeks"]

    def test_spelled_out_months(self):
        result = extract_durations("cough lasting five months")
        assert result == ["five months"]

    def test_spelled_out_case_insensitive(self):
        result = extract_durations("symptoms for Three Days")
        assert result == ["three days"]

    def test_mixed_numeric_and_spelled(self):
        result = extract_durations("three days of pain and 2 weeks of cough")
        assert "three days" in result
        assert "2 weeks" in result

    def test_spelled_out_unique(self):
        result = extract_durations("three days ago, still three days later")
        assert result.count("three days") == 1

    def test_spelled_deterministic(self):
        text = "three days, five weeks, two months"
        r1 = extract_durations(text)
        r2 = extract_durations(text)
        assert r1 == r2


# ── medication extraction ────────────────────────────────────────────


class TestMedicationExtraction:
    def test_med_with_dosage(self):
        result = extract_medications("ibuprofen 400 mg twice daily")
        assert len(result) >= 1
        assert any("ibuprofen" in r and "400" in r for r in result)

    def test_med_keyword_only(self):
        result = extract_medications("taking aspirin daily")
        assert "aspirin" in result

    def test_no_meds(self):
        result = extract_medications("feeling better today")
        assert result == []

    def test_multiple_meds(self):
        result = extract_medications("metformin 500 mg and lisinopril 10 mg")
        found = " ".join(result).lower()
        assert "metformin" in found
        assert "lisinopril" in found

    def test_dosage_takes_priority(self):
        """When a med is found with dosage, keyword-only fallback is skipped."""
        result = extract_medications("ibuprofen 400 mg for pain")
        ibu_entries = [r for r in result if "ibuprofen" in r.lower()]
        assert len(ibu_entries) == 1  # Not duplicated

    def test_deterministic(self):
        text = "aspirin 81 mg and metformin 500 mg"
        r1 = extract_medications(text)
        r2 = extract_medications(text)
        assert r1 == r2


# ── extractor registry ───────────────────────────────────────────────


class TestExtractorRegistry:
    def test_get_known_extractor(self):
        func = get_extractor("symptoms")
        assert callable(func)

    def test_all_registered(self):
        for name in ["symptoms", "negations", "durations", "medications"]:
            assert name in EXTRACTORS

    def test_unknown_extractor_raises(self):
        with pytest.raises(KeyError):
            get_extractor("unknown_extractor")

    def test_registry_functions_callable(self):
        for name, func in EXTRACTORS.items():
            result = func("test text")
            assert isinstance(result, list)
