"""Regression tests for UTI symptom extraction and scoring.

Validates that the UTI seed case and its variants score correctly
after adding urinary symptom vocabulary, synonym canonicalisation,
and the UTI diagnostic hint rule.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.extractors import extract_symptoms
from app.diagnostic_hints import generate_diagnostic_hints
from app.case_system import load_case, validate_case, run_case
from app.case_scoring import score_result_against_ground_truth
from app.case_variations import apply_variation

_UTI_CASE_PATH = Path(__file__).resolve().parent.parent / "resources" / "cases" / "uti.yaml"


# ── extraction ──────────────────────────────────────────────────────


class TestUTIExtraction:
    def test_painful_urination_extracts_as_dysuria(self):
        text = "Patient reports painful urination and urinary frequency for 2 days."
        symptoms = extract_symptoms(text)
        assert "dysuria" in symptoms
        assert "urinary frequency" in symptoms

    def test_dysuria_literal_extracted(self):
        text = "42-year-old female presenting with dysuria and urinary frequency."
        symptoms = extract_symptoms(text)
        assert "dysuria" in symptoms
        assert "urinary frequency" in symptoms

    def test_abdominal_pain_still_extracted(self):
        text = "Also has lower abdominal pain."
        symptoms = extract_symptoms(text)
        assert "abdominal pain" in symptoms

    def test_full_uti_transcript(self):
        text = (
            "Patient reports painful urination and urinary frequency for 2 days. "
            "Also has lower abdominal pain. "
            "Denies fever. Denies nausea."
        )
        symptoms = extract_symptoms(text)
        assert "dysuria" in symptoms
        assert "urinary frequency" in symptoms
        assert "abdominal pain" in symptoms


# ── diagnostic hint ─────────────────────────────────────────────────


class TestUTIDiagnosticHint:
    def test_uti_hint_fires(self):
        hints = generate_diagnostic_hints(["dysuria", "urinary frequency"])
        conditions = [h["condition"] for h in hints]
        assert "Urinary tract infection" in conditions

    def test_uti_hint_suppressed_by_negation(self):
        hints = generate_diagnostic_hints(
            ["dysuria", "urinary frequency"],
            ["Denies urinary frequency"],
        )
        conditions = [h["condition"] for h in hints]
        assert "Urinary tract infection" not in conditions


# ── seed case replay ────────────────────────────────────────────────


class TestUTISeedCase:
    @pytest.fixture()
    def uti_case(self):
        if not _UTI_CASE_PATH.exists():
            pytest.skip("UTI case file missing")
        return load_case(_UTI_CASE_PATH)

    def test_valid(self, uti_case):
        assert validate_case(uti_case)["valid"]

    def test_key_findings_all_present(self, uti_case):
        result = run_case(uti_case)
        score = score_result_against_ground_truth(result)
        kf = score["key_findings"]
        assert kf["hit_rate"] == 1.0, f"missing: {kf['missing']}"

    def test_hypothesis_present(self, uti_case):
        result = run_case(uti_case)
        score = score_result_against_ground_truth(result)
        hyp = score["hypotheses"]
        assert "Urinary tract infection" in hyp["present"]

    def test_overall_score_above_threshold(self, uti_case):
        result = run_case(uti_case)
        score = score_result_against_ground_truth(result)
        summary = score["summary"]
        composite = (
            summary["hypothesis_hit_rate"]
            + summary["red_flag_hit_rate"]
            + summary["key_finding_hit_rate"]
        ) / 3
        assert composite >= 0.5


# ── variant replay ──────────────────────────────────────────────────


class TestUTIVariant:
    @pytest.fixture()
    def uti_case(self):
        if not _UTI_CASE_PATH.exists():
            pytest.skip("UTI case file missing")
        return load_case(_UTI_CASE_PATH)

    def test_add_duration_longer_still_scores(self, uti_case):
        variant = apply_variation(uti_case, "add_duration_longer")
        assert validate_case(variant)["valid"]
        result = run_case(variant)
        score = score_result_against_ground_truth(result)
        kf = score["key_findings"]
        assert kf["hit_rate"] >= 0.66, f"missing: {kf['missing']}"

    def test_add_negation_variant_scores(self, uti_case):
        variant = apply_variation(uti_case, "add_negation_of_core_symptom")
        assert validate_case(variant)["valid"]
        result = run_case(variant)
        score = score_result_against_ground_truth(result)
        assert score["has_ground_truth"]
