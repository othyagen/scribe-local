"""Tests for semantic qualifier extraction module."""

from __future__ import annotations

import pytest

from app.qualifier_extraction import extract_qualifiers
from app.clinical_state import build_clinical_state


def _seg(text: str, seg_id: str = "seg_0001", speaker_id: str = "spk_0",
         t0: float = 0.0, t1: float = 1.0) -> dict:
    return {
        "seg_id": seg_id,
        "t0": t0,
        "t1": t1,
        "speaker_id": speaker_id,
        "normalized_text": text,
    }


_ALL_SYMPTOMS = [
    "headache", "nausea", "pain", "dizziness", "fever",
    "cough", "chest pain", "shortness of breath",
]


# ── severity detection ──────────────────────────────────────────────


class TestSeverity:
    def test_severe_headache(self):
        result = extract_qualifiers(
            [_seg("severe headache for 2 days.")],
            extracted_findings=["headache"],
        )
        assert len(result) == 1
        assert result[0]["symptom"] == "headache"
        assert result[0]["qualifiers"]["severity"] == "severe"

    def test_mild_pain(self):
        result = extract_qualifiers(
            [_seg("mild pain in the abdomen.")],
            extracted_findings=["pain"],
        )
        assert len(result) == 1
        assert result[0]["qualifiers"]["severity"] == "mild"

    def test_moderate_nausea(self):
        result = extract_qualifiers(
            [_seg("moderate nausea after eating.")],
            extracted_findings=["nausea"],
        )
        assert len(result) == 1
        assert result[0]["qualifiers"]["severity"] == "moderate"

    def test_intense_pain(self):
        result = extract_qualifiers(
            [_seg("intense pain in the chest.")],
            extracted_findings=["pain"],
        )
        assert len(result) == 1
        assert result[0]["qualifiers"]["severity"] == "intense"

    def test_synonym_terrible(self):
        result = extract_qualifiers(
            [_seg("terrible headache all day.")],
            extracted_findings=["headache"],
        )
        assert len(result) == 1
        assert result[0]["qualifiers"]["severity"] == "severe"

    def test_synonym_excruciating(self):
        result = extract_qualifiers(
            [_seg("excruciating pain in the back.")],
            extracted_findings=["pain"],
        )
        assert result[0]["qualifiers"]["severity"] == "severe"

    def test_synonym_slight(self):
        result = extract_qualifiers(
            [_seg("slight headache this morning.")],
            extracted_findings=["headache"],
        )
        assert result[0]["qualifiers"]["severity"] == "mild"


# ── onset detection ─────────────────────────────────────────────────


class TestOnset:
    def test_sudden_onset(self):
        result = extract_qualifiers(
            [_seg("sudden headache at work.")],
            extracted_findings=["headache"],
        )
        assert result[0]["qualifiers"]["onset"] == "sudden"

    def test_gradual_onset(self):
        result = extract_qualifiers(
            [_seg("gradual pain in the knee.")],
            extracted_findings=["pain"],
        )
        assert result[0]["qualifiers"]["onset"] == "gradual"

    def test_acute_onset(self):
        result = extract_qualifiers(
            [_seg("acute chest pain.")],
            extracted_findings=["chest pain"],
        )
        assert result[0]["qualifiers"]["onset"] == "acute"

    def test_synonym_abrupt(self):
        result = extract_qualifiers(
            [_seg("abrupt headache while exercising.")],
            extracted_findings=["headache"],
        )
        assert result[0]["qualifiers"]["onset"] == "sudden"

    def test_synonym_insidious(self):
        result = extract_qualifiers(
            [_seg("insidious nausea over several weeks.")],
            extracted_findings=["nausea"],
        )
        assert result[0]["qualifiers"]["onset"] == "gradual"


# ── character detection ─────────────────────────────────────────────


class TestCharacter:
    def test_cramping(self):
        result = extract_qualifiers(
            [_seg("cramping pain in the abdomen.")],
            extracted_findings=["pain"],
        )
        assert result[0]["qualifiers"]["character"] == "cramping"

    def test_burning(self):
        result = extract_qualifiers(
            [_seg("burning pain in the chest.")],
            extracted_findings=["pain"],
        )
        assert result[0]["qualifiers"]["character"] == "burning"

    def test_stabbing(self):
        result = extract_qualifiers(
            [_seg("stabbing headache.")],
            extracted_findings=["headache"],
        )
        assert result[0]["qualifiers"]["character"] == "stabbing"

    def test_dull(self):
        result = extract_qualifiers(
            [_seg("dull pain in the back.")],
            extracted_findings=["pain"],
        )
        assert result[0]["qualifiers"]["character"] == "dull"

    def test_sharp(self):
        result = extract_qualifiers(
            [_seg("sharp chest pain.")],
            extracted_findings=["chest pain"],
        )
        assert result[0]["qualifiers"]["character"] == "sharp"

    def test_throbbing(self):
        result = extract_qualifiers(
            [_seg("throbbing headache.")],
            extracted_findings=["headache"],
        )
        assert result[0]["qualifiers"]["character"] == "throbbing"

    def test_pressure_like(self):
        result = extract_qualifiers(
            [_seg("pressure-like chest pain.")],
            extracted_findings=["chest pain"],
        )
        assert result[0]["qualifiers"]["character"] == "pressure-like"

    def test_pressure_synonym(self):
        result = extract_qualifiers(
            [_seg("pressure in the chest pain area.")],
            extracted_findings=["chest pain"],
        )
        assert result[0]["qualifiers"]["character"] == "pressure-like"

    def test_productive(self):
        result = extract_qualifiers(
            [_seg("productive cough with mucus.")],
            extracted_findings=["cough"],
        )
        assert result[0]["qualifiers"]["character"] == "productive"

    def test_squeezing(self):
        result = extract_qualifiers(
            [_seg("squeezing chest pain.")],
            extracted_findings=["chest pain"],
        )
        assert result[0]["qualifiers"]["character"] == "squeezing"

    def test_colicky(self):
        result = extract_qualifiers(
            [_seg("colicky pain in the abdomen.")],
            extracted_findings=["pain"],
        )
        assert result[0]["qualifiers"]["character"] == "colicky"

    def test_pounding_maps_to_throbbing(self):
        result = extract_qualifiers(
            [_seg("pounding headache.")],
            extracted_findings=["headache"],
        )
        assert result[0]["qualifiers"]["character"] == "throbbing"

    def test_negated_character(self):
        result = extract_qualifiers(
            [_seg("not sharp headache.")],
            extracted_findings=["headache"],
        )
        if result:
            assert "character" not in result[0]["qualifiers"]


# ── pattern detection ───────────────────────────────────────────────


class TestPattern:
    def test_intermittent(self):
        result = extract_qualifiers(
            [_seg("intermittent headache.")],
            extracted_findings=["headache"],
        )
        assert result[0]["qualifiers"]["pattern"] == "intermittent"

    def test_constant(self):
        result = extract_qualifiers(
            [_seg("constant pain in the lower back.")],
            extracted_findings=["pain"],
        )
        assert result[0]["qualifiers"]["pattern"] == "constant"

    def test_episodic(self):
        result = extract_qualifiers(
            [_seg("episodic dizziness.")],
            extracted_findings=["dizziness"],
        )
        assert result[0]["qualifiers"]["pattern"] == "episodic"

    def test_comes_and_goes(self):
        result = extract_qualifiers(
            [_seg("headache comes and goes.")],
            extracted_findings=["headache"],
        )
        assert result[0]["qualifiers"]["pattern"] == "intermittent"

    def test_persistent(self):
        result = extract_qualifiers(
            [_seg("persistent cough for a week.")],
            extracted_findings=["cough"],
        )
        assert result[0]["qualifiers"]["pattern"] == "constant"

    def test_recurrent(self):
        result = extract_qualifiers(
            [_seg("recurrent nausea.")],
            extracted_findings=["nausea"],
        )
        assert result[0]["qualifiers"]["pattern"] == "episodic"


# ── progression detection ──────────────────────────────────────────


class TestProgression:
    def test_worsening(self):
        result = extract_qualifiers(
            [_seg("worsening headache over the past week.")],
            extracted_findings=["headache"],
        )
        assert result[0]["qualifiers"]["progression"] == "worsening"

    def test_improving(self):
        result = extract_qualifiers(
            [_seg("pain improving with medication.")],
            extracted_findings=["pain"],
        )
        assert result[0]["qualifiers"]["progression"] == "improving"

    def test_stable(self):
        result = extract_qualifiers(
            [_seg("stable headache no change.")],
            extracted_findings=["headache"],
        )
        assert result[0]["qualifiers"]["progression"] == "stable"

    def test_getting_worse(self):
        result = extract_qualifiers(
            [_seg("headache getting worse.")],
            extracted_findings=["headache"],
        )
        assert result[0]["qualifiers"]["progression"] == "worsening"

    def test_getting_better(self):
        result = extract_qualifiers(
            [_seg("nausea getting better today.")],
            extracted_findings=["nausea"],
        )
        assert result[0]["qualifiers"]["progression"] == "improving"

    def test_resolving(self):
        result = extract_qualifiers(
            [_seg("cough resolving gradually.")],
            extracted_findings=["cough"],
        )
        assert result[0]["qualifiers"]["progression"] == "improving"


# ── laterality detection ───────────────────────────────────────────


class TestLaterality:
    def test_left(self):
        result = extract_qualifiers(
            [_seg("left headache.")],
            extracted_findings=["headache"],
        )
        assert result[0]["qualifiers"]["laterality"] == "left"

    def test_right(self):
        result = extract_qualifiers(
            [_seg("right pain in the knee.")],
            extracted_findings=["pain"],
        )
        assert result[0]["qualifiers"]["laterality"] == "right"

    def test_bilateral(self):
        result = extract_qualifiers(
            [_seg("bilateral pain in the legs.")],
            extracted_findings=["pain"],
        )
        assert result[0]["qualifiers"]["laterality"] == "bilateral"

    def test_unilateral(self):
        result = extract_qualifiers(
            [_seg("unilateral headache.")],
            extracted_findings=["headache"],
        )
        assert result[0]["qualifiers"]["laterality"] == "unilateral"

    def test_left_sided(self):
        result = extract_qualifiers(
            [_seg("left-sided chest pain.")],
            extracted_findings=["chest pain"],
        )
        assert result[0]["qualifiers"]["laterality"] == "left"


# ── radiation detection ────────────────────────────────────────────


class TestRadiation:
    def test_radiating_to(self):
        result = extract_qualifiers(
            [_seg("chest pain radiating to the left arm.")],
            extracted_findings=["chest pain"],
        )
        assert result[0]["qualifiers"]["radiation"] == "to left arm"

    def test_spreading_to(self):
        result = extract_qualifiers(
            [_seg("headache spreading to the neck.")],
            extracted_findings=["headache"],
        )
        assert result[0]["qualifiers"]["radiation"] == "to neck"

    def test_radiates_to(self):
        result = extract_qualifiers(
            [_seg("pain radiates to the back.")],
            extracted_findings=["pain"],
        )
        assert result[0]["qualifiers"]["radiation"] == "to back"

    def test_extending_to(self):
        result = extract_qualifiers(
            [_seg("pain extending to the shoulder.")],
            extracted_findings=["pain"],
        )
        assert result[0]["qualifiers"]["radiation"] == "to shoulder"


# ── aggravating factors ───────────────────────────────────────────


class TestAggravatingFactors:
    def test_worse_with(self):
        result = extract_qualifiers(
            [_seg("headache worse with movement.")],
            extracted_findings=["headache"],
        )
        assert "movement" in result[0]["qualifiers"]["aggravating_factors"]

    def test_triggered_by(self):
        result = extract_qualifiers(
            [_seg("headache triggered by bright light.")],
            extracted_findings=["headache"],
        )
        assert any(
            "bright light" in f for f in result[0]["qualifiers"]["aggravating_factors"]
        )

    def test_aggravated_by(self):
        result = extract_qualifiers(
            [_seg("pain aggravated by coughing.")],
            extracted_findings=["pain"],
        )
        assert any(
            "coughing" in f for f in result[0]["qualifiers"]["aggravating_factors"]
        )

    def test_exacerbated_by(self):
        result = extract_qualifiers(
            [_seg("pain exacerbated by exercise.")],
            extracted_findings=["pain"],
        )
        assert any(
            "exercise" in f for f in result[0]["qualifiers"]["aggravating_factors"]
        )


# ── relieving factors ─────────────────────────────────────────────


class TestRelievingFactors:
    def test_relieved_by(self):
        result = extract_qualifiers(
            [_seg("headache relieved by rest.")],
            extracted_findings=["headache"],
        )
        assert "rest" in result[0]["qualifiers"]["relieving_factors"]

    def test_better_with(self):
        result = extract_qualifiers(
            [_seg("pain better with ibuprofen.")],
            extracted_findings=["pain"],
        )
        assert any(
            "ibuprofen" in f for f in result[0]["qualifiers"]["relieving_factors"]
        )

    def test_improved_by(self):
        result = extract_qualifiers(
            [_seg("nausea improved by ginger.")],
            extracted_findings=["nausea"],
        )
        assert any(
            "ginger" in f for f in result[0]["qualifiers"]["relieving_factors"]
        )

    def test_eased_by(self):
        result = extract_qualifiers(
            [_seg("headache eased by sleep.")],
            extracted_findings=["headache"],
        )
        assert any(
            "sleep" in f for f in result[0]["qualifiers"]["relieving_factors"]
        )


# ── multi-symptom segments ────────────────────────────────────────


class TestMultiSymptom:
    def test_two_symptoms_different_qualifiers(self):
        result = extract_qualifiers(
            [_seg("severe headache and mild nausea.")],
            extracted_findings=["headache", "nausea"],
        )
        symptoms = {r["symptom"]: r["qualifiers"] for r in result}
        assert symptoms["headache"]["severity"] == "severe"
        assert symptoms["nausea"]["severity"] == "mild"

    def test_two_symptoms_same_segment(self):
        result = extract_qualifiers(
            [_seg("intermittent headache and constant pain.")],
            extracted_findings=["headache", "pain"],
        )
        symptoms = {r["symptom"]: r["qualifiers"] for r in result}
        assert "headache" in symptoms
        assert "pain" in symptoms

    def test_symptoms_across_segments(self):
        result = extract_qualifiers(
            [
                _seg("severe headache.", seg_id="seg_0001", t0=0.0, t1=1.0),
                _seg("mild nausea.", seg_id="seg_0002", t0=1.0, t1=2.0),
            ],
            extracted_findings=["headache", "nausea"],
        )
        symptoms = {r["symptom"]: r["qualifiers"] for r in result}
        assert symptoms["headache"]["severity"] == "severe"
        assert symptoms["nausea"]["severity"] == "mild"


# ── no symptom present ────────────────────────────────────────────


class TestNoSymptom:
    def test_no_segments(self):
        result = extract_qualifiers([], extracted_findings=["headache"])
        assert result == []

    def test_no_findings(self):
        result = extract_qualifiers(
            [_seg("severe and constant.")],
            extracted_findings=[],
        )
        assert result == []

    def test_no_matching_symptom_in_text(self):
        result = extract_qualifiers(
            [_seg("the weather is severe today.")],
            extracted_findings=["headache"],
        )
        assert result == []

    def test_no_qualifiers_detected(self):
        result = extract_qualifiers(
            [_seg("headache.")],
            extracted_findings=["headache"],
        )
        assert result == []

    def test_none_findings_uses_auto_detect(self):
        result = extract_qualifiers(
            [_seg("severe headache.")],
            extracted_findings=None,
        )
        assert len(result) == 1
        assert result[0]["symptom"] == "headache"
        assert result[0]["qualifiers"]["severity"] == "severe"


# ── ambiguous phrases ─────────────────────────────────────────────


class TestAmbiguity:
    def test_qualifier_too_far_from_symptom(self):
        result = extract_qualifiers(
            [_seg("severe. the patient also mentioned many other things and talked at great length about various unrelated topics before finally bringing up a headache.")],
            extracted_findings=["headache"],
        )
        # "severe" is too far from "headache" — should not link
        if result:
            assert "severity" not in result[0]["qualifiers"]

    def test_first_segment_wins_dedup(self):
        result = extract_qualifiers(
            [
                _seg("mild headache.", seg_id="seg_0001", t0=0.0, t1=1.0),
                _seg("severe headache.", seg_id="seg_0002", t0=1.0, t1=2.0),
            ],
            extracted_findings=["headache"],
        )
        assert len(result) == 1
        assert result[0]["qualifiers"]["severity"] == "mild"


# ── negated phrases ───────────────────────────────────────────────


class TestNegation:
    def test_no_severe(self):
        result = extract_qualifiers(
            [_seg("no severe headache.")],
            extracted_findings=["headache"],
        )
        if result:
            assert "severity" not in result[0]["qualifiers"]

    def test_not_worsening(self):
        result = extract_qualifiers(
            [_seg("headache not worsening.")],
            extracted_findings=["headache"],
        )
        if result:
            assert "progression" not in result[0]["qualifiers"]

    def test_denies_sudden(self):
        result = extract_qualifiers(
            [_seg("denies sudden headache.")],
            extracted_findings=["headache"],
        )
        if result:
            assert "onset" not in result[0]["qualifiers"]

    def test_without_intermittent(self):
        result = extract_qualifiers(
            [_seg("pain without intermittent episodes.")],
            extracted_findings=["pain"],
        )
        if result:
            assert "pattern" not in result[0]["qualifiers"]


# ── combined qualifiers ───────────────────────────────────────────


class TestCombined:
    def test_multiple_qualifiers_on_one_symptom(self):
        result = extract_qualifiers(
            [_seg("severe sudden headache radiating to the neck.")],
            extracted_findings=["headache"],
        )
        assert len(result) == 1
        q = result[0]["qualifiers"]
        assert q["severity"] == "severe"
        assert q["onset"] == "sudden"
        assert q["radiation"] == "to neck"

    def test_full_qualifier_set(self):
        result = extract_qualifiers(
            [_seg("severe sudden throbbing intermittent worsening left headache "
                  "radiating to the neck worse with movement relieved by rest.")],
            extracted_findings=["headache"],
        )
        assert len(result) == 1
        q = result[0]["qualifiers"]
        assert q["severity"] == "severe"
        assert q["onset"] == "sudden"
        assert q["character"] == "throbbing"
        assert q["pattern"] == "intermittent"
        assert q["progression"] == "worsening"
        assert q["laterality"] == "left"
        assert q["radiation"] == "to neck"
        assert "movement" in q["aggravating_factors"]
        assert "rest" in q["relieving_factors"]

    def test_laterality_and_severity(self):
        result = extract_qualifiers(
            [_seg("severe left-sided chest pain.")],
            extracted_findings=["chest pain"],
        )
        q = result[0]["qualifiers"]
        assert q["severity"] == "severe"
        assert q["laterality"] == "left"


# ── clinical state integration ────────────────────────────────────


class TestClinicalStateIntegration:
    def test_qualifiers_in_clinical_state(self):
        state = build_clinical_state([
            _seg("severe headache for 3 days."),
        ])
        assert "qualifiers" in state
        assert isinstance(state["qualifiers"], list)
        assert len(state["qualifiers"]) >= 1
        assert state["qualifiers"][0]["symptom"] == "headache"
        assert state["qualifiers"][0]["qualifiers"]["severity"] == "severe"

    def test_empty_qualifiers(self):
        state = build_clinical_state([_seg("hello.")])
        assert "qualifiers" in state
        assert state["qualifiers"] == []

    def test_qualifiers_does_not_break_other_keys(self):
        state = build_clinical_state([_seg("severe headache for 3 days.")])
        assert "symptoms" in state
        assert "headache" in state["symptoms"]
        assert "qualifiers" in state
        assert "timeline" in state


# ── edge cases ────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_text_segment(self):
        result = extract_qualifiers(
            [_seg("")],
            extracted_findings=["headache"],
        )
        assert result == []

    def test_case_insensitive(self):
        result = extract_qualifiers(
            [_seg("SEVERE HEADACHE.")],
            extracted_findings=["headache"],
        )
        assert len(result) == 1
        assert result[0]["qualifiers"]["severity"] == "severe"

    def test_multiple_aggravating_factors(self):
        result = extract_qualifiers(
            [_seg("headache worse with movement and exacerbated by stress.")],
            extracted_findings=["headache"],
        )
        agg = result[0]["qualifiers"]["aggravating_factors"]
        assert len(agg) >= 2

    def test_on_and_off_pattern(self):
        result = extract_qualifiers(
            [_seg("headache on and off for a week.")],
            extracted_findings=["headache"],
        )
        assert result[0]["qualifiers"]["pattern"] == "intermittent"

    def test_unbearable_severity(self):
        result = extract_qualifiers(
            [_seg("unbearable headache.")],
            extracted_findings=["headache"],
        )
        assert result[0]["qualifiers"]["severity"] == "severe"

    def test_worst_severity(self):
        result = extract_qualifiers(
            [_seg("worst headache of my life.")],
            extracted_findings=["headache"],
        )
        assert result[0]["qualifiers"]["severity"] == "severe"

    def test_unchanged_progression(self):
        result = extract_qualifiers(
            [_seg("headache unchanged for weeks.")],
            extracted_findings=["headache"],
        )
        assert result[0]["qualifiers"]["progression"] == "stable"

    def test_both_sides_laterality(self):
        result = extract_qualifiers(
            [_seg("pain on both sides.")],
            extracted_findings=["pain"],
        )
        assert result[0]["qualifiers"]["laterality"] == "bilateral"

    def test_only_qualified_symptoms_returned(self):
        result = extract_qualifiers(
            [_seg("headache and nausea.")],
            extracted_findings=["headache", "nausea"],
        )
        # No qualifiers on either — should return empty
        assert result == []


# ── cross-segment linking ────────────────────────────────────────


class TestCrossSegmentLinking:
    def test_qualifier_in_later_segment(self):
        """Qualifier in seg 2 links to symptom in seg 1."""
        result = extract_qualifiers(
            [
                _seg("I have chest pain.", seg_id="seg_0001", t0=0.0, t1=1.0),
                _seg("it's dull.", seg_id="seg_0002", t0=1.0, t1=2.0),
            ],
            extracted_findings=["chest pain"],
        )
        assert len(result) == 1
        assert result[0]["symptom"] == "chest pain"
        assert result[0]["qualifiers"]["character"] == "dull"

    def test_qualifier_two_segments_later(self):
        """Qualifier 2 segments after symptom still links (within window)."""
        result = extract_qualifiers(
            [
                _seg("I have chest pain.", seg_id="seg_0001", t0=0.0, t1=1.0),
                _seg("it started yesterday.", seg_id="seg_0002", t0=1.0, t1=2.0),
                _seg("it's a dull ache.", seg_id="seg_0003", t0=2.0, t1=3.0),
            ],
            extracted_findings=["chest pain"],
        )
        assert len(result) == 1
        assert result[0]["qualifiers"]["character"] == "dull"

    def test_qualifier_beyond_window_no_link(self):
        """Qualifier too far from symptom doesn't link."""
        result = extract_qualifiers(
            [
                _seg("I have chest pain.", seg_id="seg_0001", t0=0.0, t1=1.0),
                _seg("unrelated.", seg_id="seg_0002", t0=1.0, t1=2.0),
                _seg("unrelated.", seg_id="seg_0003", t0=2.0, t1=3.0),
                _seg("unrelated.", seg_id="seg_0004", t0=3.0, t1=4.0),
                _seg("it's dull.", seg_id="seg_0005", t0=4.0, t1=5.0),
            ],
            extracted_findings=["chest pain"],
        )
        assert result == []

    def test_multiword_symptom_carried_forward(self):
        """Multi-word symptom from prior segment carried to later segment."""
        result = extract_qualifiers(
            [
                _seg("I have chest pain.", seg_id="seg_0001", t0=0.0, t1=1.0),
                _seg("it's dull and constant.", seg_id="seg_0002", t0=1.0, t1=2.0),
            ],
            extracted_findings=["chest pain"],
        )
        # "chest pain" carried forward from seg 1 to enrich seg 2
        assert len(result) == 1
        assert result[0]["symptom"] == "chest pain"
        assert result[0]["qualifiers"]["character"] == "dull"
        assert result[0]["qualifiers"]["pattern"] == "constant"

    def test_new_symptom_resets_context(self):
        """A new symptom in a later segment becomes the context."""
        result = extract_qualifiers(
            [
                _seg("I have chest pain.", seg_id="seg_0001", t0=0.0, t1=1.0),
                _seg("I also have a cough.", seg_id="seg_0002", t0=1.0, t1=2.0),
                _seg("it's productive.", seg_id="seg_0003", t0=2.0, t1=3.0),
            ],
            extracted_findings=["chest pain", "cough"],
        )
        symptoms = {r["symptom"]: r["qualifiers"] for r in result}
        assert symptoms["cough"]["character"] == "productive"


# ── question skipping ────────────────────────────────────────────


class TestQuestionSkipping:
    def test_doctor_question_skipped(self):
        """Doctor question ending in ? should not produce qualifiers."""
        result = extract_qualifiers(
            [_seg("Is it sharp or dull?")],
            extracted_findings=["pain"],
        )
        assert result == []

    def test_doctor_question_dry_or_productive(self):
        """'Is it dry or productive?' should not match cough character."""
        result = extract_qualifiers(
            [
                _seg("I have a cough.", seg_id="seg_0001", t0=0.0, t1=1.0),
                _seg("Is it dry or productive?", seg_id="seg_0002", t0=1.0, t1=2.0),
                _seg("it's productive.", seg_id="seg_0003", t0=2.0, t1=3.0),
            ],
            extracted_findings=["cough"],
        )
        assert len(result) == 1
        assert result[0]["qualifiers"]["character"] == "productive"

    def test_non_question_not_skipped(self):
        """Statement segments are still processed."""
        result = extract_qualifiers(
            [_seg("severe headache all day.")],
            extracted_findings=["headache"],
        )
        assert len(result) == 1
        assert result[0]["qualifiers"]["severity"] == "severe"


# ── conversational factor patterns ───────────────────────────────


class TestConversationalFactors:
    def test_gets_worse_when(self):
        result = extract_qualifiers(
            [_seg("chest pain gets worse when I walk up stairs.")],
            extracted_findings=["chest pain"],
        )
        agg = result[0]["qualifiers"]["aggravating_factors"]
        assert any("walk" in f.lower() for f in agg)

    def test_worse_when(self):
        result = extract_qualifiers(
            [_seg("headache worse when I bend over.")],
            extracted_findings=["headache"],
        )
        agg = result[0]["qualifiers"]["aggravating_factors"]
        assert any("bend" in f.lower() for f in agg)

    def test_gets_better_when(self):
        result = extract_qualifiers(
            [_seg("chest pain gets better when I rest.")],
            extracted_findings=["chest pain"],
        )
        rel = result[0]["qualifiers"]["relieving_factors"]
        assert any("rest" in f.lower() for f in rel)

    def test_better_when(self):
        result = extract_qualifiers(
            [_seg("pain better when I sit down.")],
            extracted_findings=["pain"],
        )
        rel = result[0]["qualifiers"]["relieving_factors"]
        assert any("sit" in f.lower() for f in rel)

    def test_worse_if(self):
        result = extract_qualifiers(
            [_seg("cough worse if I lie down.")],
            extracted_findings=["cough"],
        )
        agg = result[0]["qualifiers"]["aggravating_factors"]
        assert any("lie" in f.lower() for f in agg)

    def test_better_if(self):
        result = extract_qualifiers(
            [_seg("pain better if I take ibuprofen.")],
            extracted_findings=["pain"],
        )
        rel = result[0]["qualifiers"]["relieving_factors"]
        assert any("ibuprofen" in f.lower() for f in rel)

    def test_cross_segment_aggravating(self):
        """Conversational factor in later segment links to prior symptom."""
        result = extract_qualifiers(
            [
                _seg("I have chest pain.", seg_id="seg_0001", t0=0.0, t1=1.0),
                _seg("it gets worse when I walk up stairs.", seg_id="seg_0002", t0=1.0, t1=2.0),
            ],
            extracted_findings=["chest pain"],
        )
        assert len(result) == 1
        agg = result[0]["qualifiers"]["aggravating_factors"]
        assert any("walk" in f.lower() for f in agg)

    def test_cross_segment_relieving(self):
        """Conversational relieving factor in later segment."""
        result = extract_qualifiers(
            [
                _seg("I have chest pain.", seg_id="seg_0001", t0=0.0, t1=1.0),
                _seg("it gets better when I rest.", seg_id="seg_0002", t0=1.0, t1=2.0),
            ],
            extracted_findings=["chest pain"],
        )
        assert len(result) == 1
        rel = result[0]["qualifiers"]["relieving_factors"]
        assert any("rest" in f.lower() for f in rel)
