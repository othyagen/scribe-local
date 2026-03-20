"""Tests for the centralized clinical terminology registry."""

from __future__ import annotations

import pytest

from app.clinical_terminology import (
    CLINICAL_TERMS,
    get_canonical_label,
    get_term,
    is_red_flag,
    get_all_labels,
)


# ── get_canonical_label ──────────────────────────────────────────────


class TestGetCanonicalLabel:
    def test_known_synonym_shortness_of_breath(self):
        assert get_canonical_label("shortness of breath") == "dyspnea"

    def test_known_synonym_sob(self):
        assert get_canonical_label("SOB") == "dyspnea"

    def test_known_synonym_chest_discomfort(self):
        assert get_canonical_label("chest discomfort") == "chest pain"

    def test_known_synonym_chest_tightness(self):
        assert get_canonical_label("chest tightness") == "chest pain"

    def test_known_synonym_painful_urination(self):
        assert get_canonical_label("painful urination") == "dysuria"

    def test_known_synonym_burning_urination(self):
        assert get_canonical_label("burning urination") == "dysuria"

    def test_known_synonym_fainting(self):
        assert get_canonical_label("fainting") == "syncope"

    def test_known_synonym_coughing_up_blood(self):
        assert get_canonical_label("coughing up blood") == "hemoptysis"

    def test_canonical_label_returned_as_is(self):
        assert get_canonical_label("dyspnea") == "dyspnea"
        assert get_canonical_label("chest pain") == "chest pain"
        assert get_canonical_label("fever") == "fever"

    def test_case_insensitive(self):
        assert get_canonical_label("Shortness of Breath") == "dyspnea"
        assert get_canonical_label("CHEST PAIN") == "chest pain"
        assert get_canonical_label("Dyspnea") == "dyspnea"

    def test_whitespace_trimmed(self):
        assert get_canonical_label("  dyspnea  ") == "dyspnea"
        assert get_canonical_label("  shortness of breath  ") == "dyspnea"

    def test_unknown_label_preserved(self):
        assert get_canonical_label("headache") == "headache"

    def test_unknown_label_preserves_casing(self):
        assert get_canonical_label("Headache") == "Headache"
        assert get_canonical_label("Pneumonia") == "Pneumonia"

    def test_empty_string(self):
        assert get_canonical_label("") == ""

    def test_difficulty_breathing(self):
        assert get_canonical_label("difficulty breathing") == "dyspnea"

    def test_breathlessness(self):
        assert get_canonical_label("breathlessness") == "dyspnea"

    def test_loss_of_consciousness(self):
        assert get_canonical_label("loss of consciousness") == "syncope"


# ── get_term ─────────────────────────────────────────────────────────


class TestGetTerm:
    def test_known_term(self):
        term = get_term("dyspnea")
        assert term is not None
        assert term["display"] == "Dyspnea"
        assert term["type"] == "symptom"
        assert term["red_flag"] is True
        assert term["snomed_code"] == "267036007"

    def test_known_term_case_insensitive(self):
        assert get_term("Dyspnea") is not None
        assert get_term("CHEST PAIN") is not None

    def test_known_term_whitespace(self):
        assert get_term("  fever  ") is not None

    def test_unknown_term(self):
        assert get_term("headache") is None

    def test_all_terms_have_required_keys(self):
        required = {"display", "type", "synonyms", "red_flag", "snomed_code"}
        for label, term in CLINICAL_TERMS.items():
            assert required <= set(term.keys()), f"{label} missing keys"


# ── is_red_flag ──────────────────────────────────────────────────────


class TestIsRedFlag:
    def test_dyspnea_is_red_flag(self):
        assert is_red_flag("dyspnea") is True

    def test_chest_pain_is_red_flag(self):
        assert is_red_flag("chest pain") is True

    def test_hemoptysis_is_red_flag(self):
        assert is_red_flag("hemoptysis") is True

    def test_syncope_is_red_flag(self):
        assert is_red_flag("syncope") is True

    def test_fever_is_not_red_flag(self):
        assert is_red_flag("fever") is False

    def test_cough_is_not_red_flag(self):
        assert is_red_flag("cough") is False

    def test_dysuria_is_not_red_flag(self):
        assert is_red_flag("dysuria") is False

    def test_synonym_resolves_to_red_flag(self):
        assert is_red_flag("shortness of breath") is True
        assert is_red_flag("chest discomfort") is True
        assert is_red_flag("fainting") is True
        assert is_red_flag("coughing up blood") is True

    def test_synonym_resolves_to_non_red_flag(self):
        assert is_red_flag("painful urination") is False

    def test_unknown_label_not_red_flag(self):
        assert is_red_flag("headache") is False
        assert is_red_flag("rash") is False

    def test_case_insensitive(self):
        assert is_red_flag("DYSPNEA") is True
        assert is_red_flag("Chest Pain") is True


# ── get_all_labels ───────────────────────────────────────────────────


class TestGetAllLabels:
    def test_returns_list(self):
        labels = get_all_labels()
        assert isinstance(labels, list)

    def test_contains_all_terms(self):
        labels = get_all_labels()
        for key in CLINICAL_TERMS:
            assert key in labels

    def test_length_matches(self):
        assert len(get_all_labels()) == len(CLINICAL_TERMS)

    def test_order_matches_dict(self):
        assert get_all_labels() == list(CLINICAL_TERMS.keys())


# ── registry invariants ──────────────────────────────────────────────


class TestRegistryInvariants:
    def test_canonical_keys_are_lowercase(self):
        for key in CLINICAL_TERMS:
            assert key == key.lower(), f"key not lowercase: {key!r}"

    def test_synonyms_are_lowercase(self):
        for label, term in CLINICAL_TERMS.items():
            for syn in term["synonyms"]:
                assert syn == syn.lower(), f"{label} synonym not lowercase: {syn!r}"

    def test_no_synonym_is_also_canonical(self):
        for label, term in CLINICAL_TERMS.items():
            for syn in term["synonyms"]:
                assert syn not in CLINICAL_TERMS, (
                    f"synonym {syn!r} of {label!r} is also a canonical key"
                )

    def test_no_duplicate_synonyms_across_terms(self):
        seen: dict[str, str] = {}
        for label, term in CLINICAL_TERMS.items():
            for syn in term["synonyms"]:
                assert syn not in seen, (
                    f"synonym {syn!r} appears in both {seen[syn]!r} and {label!r}"
                )
                seen[syn] = label

    def test_red_flag_is_bool(self):
        for label, term in CLINICAL_TERMS.items():
            assert isinstance(term["red_flag"], bool), f"{label} red_flag not bool"

    def test_type_is_symptom(self):
        for label, term in CLINICAL_TERMS.items():
            assert term["type"] == "symptom", f"{label} type not symptom"
