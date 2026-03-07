"""Tests for deterministic diagnostic hints with SNOMED codes."""

from __future__ import annotations

import pytest

from app.diagnostic_hints import generate_diagnostic_hints, _extract_negated_symptoms
from app.export_clinical_note import build_clinical_note


# ── rule matching ─────────────────────────────────────────────────────


class TestRuleMatching:
    def test_pneumonia_rule(self):
        symptoms = ["fever", "cough", "shortness of breath"]
        hints = generate_diagnostic_hints(symptoms)
        conditions = [h["condition"] for h in hints]
        assert "Pneumonia" in conditions
        pneumonia = next(h for h in hints if h["condition"] == "Pneumonia")
        assert pneumonia["snomed_code"] == "233604007"
        assert set(pneumonia["evidence"]) == {"fever", "cough", "shortness of breath"}

    def test_meningitis_rule(self):
        symptoms = ["headache", "neck stiffness", "fever"]
        hints = generate_diagnostic_hints(symptoms)
        conditions = [h["condition"] for h in hints]
        assert "Meningitis" in conditions
        meningitis = next(h for h in hints if h["condition"] == "Meningitis")
        assert meningitis["snomed_code"] == "7180009"

    def test_partial_match_does_not_trigger(self):
        """Only fever+cough — missing 'shortness of breath' for pneumonia."""
        symptoms = ["fever", "cough"]
        hints = generate_diagnostic_hints(symptoms)
        conditions = [h["condition"] for h in hints]
        assert "Pneumonia" not in conditions

    def test_no_match(self):
        symptoms = ["rash"]
        hints = generate_diagnostic_hints(symptoms)
        assert hints == []

    def test_empty_symptoms(self):
        hints = generate_diagnostic_hints([])
        assert hints == []

    def test_multiple_rules_triggered(self):
        """Symptoms matching multiple rules should return all of them."""
        symptoms = ["fever", "cough", "shortness of breath", "fatigue",
                     "headache", "nausea", "vomiting"]
        hints = generate_diagnostic_hints(symptoms)
        conditions = {h["condition"] for h in hints}
        assert "Pneumonia" in conditions
        assert "Upper respiratory tract infection" in conditions
        assert "Migraine" in conditions

    def test_case_insensitive(self):
        symptoms = ["Fever", "Cough", "Shortness of Breath"]
        hints = generate_diagnostic_hints(symptoms)
        conditions = [h["condition"] for h in hints]
        assert "Pneumonia" in conditions

    def test_evidence_sorted_alphabetically(self):
        symptoms = ["shortness of breath", "fever", "cough"]
        hints = generate_diagnostic_hints(symptoms)
        pneumonia = next(h for h in hints if h["condition"] == "Pneumonia")
        assert pneumonia["evidence"] == sorted(pneumonia["evidence"])

    def test_sorted_by_evidence_count_descending(self):
        """Rules with more evidence come first."""
        symptoms = ["fever", "sore throat", "cough", "fatigue"]
        hints = generate_diagnostic_hints(symptoms)
        assert len(hints) >= 2
        for i in range(len(hints) - 1):
            assert len(hints[i]["evidence"]) >= len(hints[i + 1]["evidence"])


# ── negation suppression ─────────────────────────────────────────────


class TestNegationSuppression:
    def test_negated_symptom_blocks_rule(self):
        """If a required symptom is negated, rule should not trigger."""
        symptoms = ["fever", "cough", "shortness of breath"]
        negations = ["No fever"]
        hints = generate_diagnostic_hints(symptoms, negations)
        conditions = [h["condition"] for h in hints]
        assert "Pneumonia" not in conditions

    def test_denies_negation(self):
        symptoms = ["headache", "neck stiffness", "fever"]
        negations = ["Denies fever"]
        hints = generate_diagnostic_hints(symptoms, negations)
        conditions = [h["condition"] for h in hints]
        assert "Meningitis" not in conditions

    def test_without_negation(self):
        symptoms = ["fever", "sore throat"]
        negations = ["Without fever"]
        hints = generate_diagnostic_hints(symptoms, negations)
        conditions = [h["condition"] for h in hints]
        assert "Pharyngitis" not in conditions

    def test_negation_only_removes_negated_symptom(self):
        """Non-negated symptoms still contribute to other rules."""
        symptoms = ["fever", "cough", "shortness of breath", "sore throat"]
        negations = ["No shortness of breath"]
        hints = generate_diagnostic_hints(symptoms, negations)
        conditions = [h["condition"] for h in hints]
        # Pneumonia blocked (needs shortness of breath), but pharyngitis ok
        assert "Pneumonia" not in conditions
        assert "Pharyngitis" in conditions

    def test_none_negations(self):
        """None negations should not cause errors."""
        symptoms = ["fever", "sore throat"]
        hints = generate_diagnostic_hints(symptoms, None)
        conditions = [h["condition"] for h in hints]
        assert "Pharyngitis" in conditions

    def test_empty_negations(self):
        symptoms = ["fever", "sore throat"]
        hints = generate_diagnostic_hints(symptoms, [])
        conditions = [h["condition"] for h in hints]
        assert "Pharyngitis" in conditions


class TestExtractNegatedSymptoms:
    def test_no_prefix(self):
        result = _extract_negated_symptoms(["No fever"])
        assert "fever" in result

    def test_denies_prefix(self):
        result = _extract_negated_symptoms(["Denies chest pain"])
        assert "chest pain" in result

    def test_without_prefix(self):
        result = _extract_negated_symptoms(["Without nausea"])
        assert "nausea" in result

    def test_trailing_punctuation_stripped(self):
        result = _extract_negated_symptoms(["No fever."])
        assert "fever" in result


# ── clinical note rendering ──────────────────────────────────────────

_HINTS_TEMPLATE = {
    "name": "Test Note",
    "format": "markdown",
    "sections": [],
    "show_diagnostic_hints": True,
}


class TestDiagnosticHintsRendering:
    def test_hints_rendered_in_note(self):
        hints = [
            {
                "condition": "Pneumonia",
                "snomed_code": "233604007",
                "evidence": ["cough", "fever", "shortness of breath"],
            },
        ]
        result = build_clinical_note([], _HINTS_TEMPLATE, diagnostic_hints=hints)
        assert "## Diagnostic Hints" in result
        assert "Pneumonia (SNOMED: 233604007)" in result
        assert "Evidence: cough, fever, shortness of breath" in result

    def test_hints_not_rendered_without_flag(self):
        template = {**_HINTS_TEMPLATE, "show_diagnostic_hints": False}
        hints = [{"condition": "X", "snomed_code": "0", "evidence": ["a"]}]
        result = build_clinical_note([], template, diagnostic_hints=hints)
        assert "## Diagnostic Hints" not in result

    def test_hints_empty_no_section(self):
        result = build_clinical_note([], _HINTS_TEMPLATE, diagnostic_hints=[])
        assert "## Diagnostic Hints" not in result

    def test_hints_none_no_section(self):
        result = build_clinical_note([], _HINTS_TEMPLATE, diagnostic_hints=None)
        assert "## Diagnostic Hints" not in result

    def test_multiple_hints_rendered(self):
        hints = [
            {"condition": "A", "snomed_code": "1", "evidence": ["x", "y"]},
            {"condition": "B", "snomed_code": "2", "evidence": ["z"]},
        ]
        result = build_clinical_note([], _HINTS_TEMPLATE, diagnostic_hints=hints)
        assert "A (SNOMED: 1)" in result
        assert "B (SNOMED: 2)" in result
        assert "Evidence: x, y" in result
        assert "Evidence: z" in result

    def test_text_format_rendering(self):
        template = {**_HINTS_TEMPLATE, "format": "text"}
        hints = [{"condition": "A", "snomed_code": "1", "evidence": ["x"]}]
        result = build_clinical_note([], template, diagnostic_hints=hints)
        assert "Diagnostic Hints" in result
        assert "A (SNOMED: 1)" in result
        assert "# " not in result
