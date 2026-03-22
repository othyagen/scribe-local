"""Tests for deterministic diagnostic hints with SNOMED codes."""

from __future__ import annotations

import pytest

from app.diagnostic_hints import generate_diagnostic_hints, _extract_negated_symptoms
from app.export_clinical_note import build_clinical_note


# ── rule matching ─────────────────────────────────────────────────────


class TestRuleMatching:
    def test_pneumonia_rule_all_three(self):
        symptoms = ["fever", "cough", "dyspnea"]
        hints = generate_diagnostic_hints(symptoms)
        conditions = [h["condition"] for h in hints]
        assert "Pneumonia" in conditions
        pneumonia = next(h for h in hints if h["condition"] == "Pneumonia")
        assert pneumonia["snomed_code"] == "233604007"
        assert set(pneumonia["evidence"]) == {"fever", "cough", "dyspnea"}

    def test_pneumonia_rule_two_of_three(self):
        """Pneumonia fires with any 2 of fever, cough, dyspnea."""
        for pair in [["fever", "cough"], ["cough", "dyspnea"], ["fever", "dyspnea"]]:
            hints = generate_diagnostic_hints(pair)
            conditions = [h["condition"] for h in hints]
            assert "Pneumonia" in conditions, f"Pneumonia should fire for {pair}"

    def test_pneumonia_single_symptom_does_not_trigger(self):
        """A single symptom should not trigger pneumonia."""
        for sym in ["fever", "cough", "dyspnea"]:
            hints = generate_diagnostic_hints([sym])
            conditions = [h["condition"] for h in hints]
            assert "Pneumonia" not in conditions

    def test_meningitis_rule(self):
        symptoms = ["headache", "neck stiffness", "fever"]
        hints = generate_diagnostic_hints(symptoms)
        conditions = [h["condition"] for h in hints]
        assert "Meningitis" in conditions
        meningitis = next(h for h in hints if h["condition"] == "Meningitis")
        assert meningitis["snomed_code"] == "7180009"

    def test_partial_match_does_not_trigger(self):
        """Only fever — missing both cough and dyspnea for pneumonia."""
        symptoms = ["fever"]
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
        symptoms = ["fever", "cough", "dyspnea", "fatigue",
                     "headache", "nausea", "vomiting"]
        hints = generate_diagnostic_hints(symptoms)
        conditions = {h["condition"] for h in hints}
        assert "Pneumonia" in conditions
        assert "Upper respiratory tract infection" in conditions
        assert "Migraine" in conditions

    def test_uti_rule(self):
        symptoms = ["dysuria", "urinary frequency"]
        hints = generate_diagnostic_hints(symptoms)
        conditions = [h["condition"] for h in hints]
        assert "Urinary tract infection" in conditions
        uti = next(h for h in hints if h["condition"] == "Urinary tract infection")
        assert uti["snomed_code"] == "68566005"
        assert set(uti["evidence"]) == {"dysuria", "urinary frequency"}

    def test_uti_partial_match_does_not_trigger(self):
        symptoms = ["dysuria"]
        hints = generate_diagnostic_hints(symptoms)
        conditions = [h["condition"] for h in hints]
        assert "Urinary tract infection" not in conditions

    def test_uti_negation_blocks(self):
        symptoms = ["dysuria", "urinary frequency"]
        negations = ["Denies dysuria"]
        hints = generate_diagnostic_hints(symptoms, negations)
        conditions = [h["condition"] for h in hints]
        assert "Urinary tract infection" not in conditions

    def test_case_insensitive(self):
        symptoms = ["Fever", "Cough", "Dyspnea"]
        hints = generate_diagnostic_hints(symptoms)
        conditions = [h["condition"] for h in hints]
        assert "Pneumonia" in conditions

    def test_evidence_sorted_alphabetically(self):
        symptoms = ["dyspnea", "fever", "cough"]
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


# ── pulmonary embolism rule ───────────────────────────────────────────


class TestPulmonaryEmbolismRule:
    def test_pe_fires_with_triad(self):
        symptoms = ["chest pain", "dyspnea", "swelling"]
        hints = generate_diagnostic_hints(symptoms)
        conditions = [h["condition"] for h in hints]
        assert "Pulmonary embolism" in conditions
        pe = next(h for h in hints if h["condition"] == "Pulmonary embolism")
        assert pe["snomed_code"] == "59282003"
        assert set(pe["evidence"]) == {"chest pain", "dyspnea", "swelling"}

    def test_pe_does_not_fire_without_swelling(self):
        symptoms = ["chest pain", "dyspnea"]
        hints = generate_diagnostic_hints(symptoms)
        conditions = [h["condition"] for h in hints]
        assert "Pulmonary embolism" not in conditions
        assert "Acute coronary syndrome" in conditions

    def test_pe_does_not_fire_without_dyspnea(self):
        symptoms = ["chest pain", "swelling"]
        hints = generate_diagnostic_hints(symptoms)
        conditions = [h["condition"] for h in hints]
        assert "Pulmonary embolism" not in conditions

    def test_pe_does_not_fire_without_chest_pain(self):
        symptoms = ["dyspnea", "swelling"]
        hints = generate_diagnostic_hints(symptoms)
        conditions = [h["condition"] for h in hints]
        assert "Pulmonary embolism" not in conditions

    def test_pe_ranks_above_acs_by_evidence_count(self):
        """PE (3 evidence) should sort before ACS (2 evidence)."""
        symptoms = ["chest pain", "dyspnea", "swelling"]
        hints = generate_diagnostic_hints(symptoms)
        conditions = [h["condition"] for h in hints]
        assert "Pulmonary embolism" in conditions
        assert "Acute coronary syndrome" in conditions
        pe_idx = conditions.index("Pulmonary embolism")
        acs_idx = conditions.index("Acute coronary syndrome")
        assert pe_idx < acs_idx

    def test_pe_negated_swelling_blocks(self):
        symptoms = ["chest pain", "dyspnea", "swelling"]
        negations = ["No swelling"]
        hints = generate_diagnostic_hints(symptoms, negations)
        conditions = [h["condition"] for h in hints]
        assert "Pulmonary embolism" not in conditions
        assert "Acute coronary syndrome" in conditions

    def test_pe_case_insensitive(self):
        symptoms = ["Chest Pain", "Dyspnea", "Swelling"]
        hints = generate_diagnostic_hints(symptoms)
        conditions = [h["condition"] for h in hints]
        assert "Pulmonary embolism" in conditions


# ── pericarditis rule ─────────────────────────────────────────────────


def _pericarditis_qualifiers():
    """Qualifier list matching the pericarditis pattern."""
    return [
        {
            "symptom": "chest pain",
            "qualifiers": {
                "character": "sharp",
                "aggravating_factors": ["deep breathing"],
                "relieving_factors": ["leaning forward"],
            },
        },
    ]


class TestPericarditisRule:
    def test_fires_with_full_qualifier_match(self):
        symptoms = ["chest pain"]
        hints = generate_diagnostic_hints(
            symptoms, qualifiers=_pericarditis_qualifiers(),
        )
        conditions = [h["condition"] for h in hints]
        assert "Pericarditis" in conditions
        peri = next(h for h in hints if h["condition"] == "Pericarditis")
        assert peri["snomed_code"] == "3238004"
        assert peri["evidence"] == ["chest pain"]

    def test_does_not_fire_without_qualifiers(self):
        symptoms = ["chest pain"]
        hints = generate_diagnostic_hints(symptoms)
        conditions = [h["condition"] for h in hints]
        assert "Pericarditis" not in conditions

    def test_does_not_fire_wrong_character(self):
        quals = [
            {
                "symptom": "chest pain",
                "qualifiers": {
                    "character": "pressure-like",
                    "aggravating_factors": ["deep breathing"],
                    "relieving_factors": ["leaning forward"],
                },
            },
        ]
        hints = generate_diagnostic_hints(["chest pain"], qualifiers=quals)
        conditions = [h["condition"] for h in hints]
        assert "Pericarditis" not in conditions

    def test_does_not_fire_missing_aggravating(self):
        quals = [
            {
                "symptom": "chest pain",
                "qualifiers": {
                    "character": "sharp",
                    "aggravating_factors": [],
                    "relieving_factors": ["leaning forward"],
                },
            },
        ]
        hints = generate_diagnostic_hints(["chest pain"], qualifiers=quals)
        conditions = [h["condition"] for h in hints]
        assert "Pericarditis" not in conditions

    def test_does_not_fire_missing_relieving(self):
        quals = [
            {
                "symptom": "chest pain",
                "qualifiers": {
                    "character": "sharp",
                    "aggravating_factors": ["deep breathing"],
                    "relieving_factors": [],
                },
            },
        ]
        hints = generate_diagnostic_hints(["chest pain"], qualifiers=quals)
        conditions = [h["condition"] for h in hints]
        assert "Pericarditis" not in conditions

    def test_does_not_fire_without_chest_pain_symptom(self):
        hints = generate_diagnostic_hints(
            ["pain"], qualifiers=_pericarditis_qualifiers(),
        )
        conditions = [h["condition"] for h in hints]
        assert "Pericarditis" not in conditions

    def test_case_insensitive_qualifiers(self):
        quals = [
            {
                "symptom": "Chest Pain",
                "qualifiers": {
                    "character": "Sharp",
                    "aggravating_factors": ["Deep Breathing"],
                    "relieving_factors": ["Leaning Forward"],
                },
            },
        ]
        hints = generate_diagnostic_hints(["chest pain"], qualifiers=quals)
        conditions = [h["condition"] for h in hints]
        assert "Pericarditis" in conditions

    def test_existing_rules_unaffected_by_qualifiers_param(self):
        """Passing qualifiers does not break rules without qualifier constraints."""
        symptoms = ["chest pain", "dyspnea"]
        hints = generate_diagnostic_hints(
            symptoms, qualifiers=_pericarditis_qualifiers(),
        )
        conditions = [h["condition"] for h in hints]
        assert "Acute coronary syndrome" in conditions


# ── negation suppression ─────────────────────────────────────────────


class TestNegationSuppression:
    def test_negated_symptom_blocks_rule(self):
        """If enough required symptoms are negated, rule should not trigger."""
        symptoms = ["fever", "cough", "dyspnea"]
        negations = ["No fever", "No dyspnea"]
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
        symptoms = ["fever", "cough", "dyspnea", "sore throat"]
        negations = ["No dyspnea", "No cough"]
        hints = generate_diagnostic_hints(symptoms, negations)
        conditions = [h["condition"] for h in hints]
        # Pneumonia blocked (only fever remains), but pharyngitis ok
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
