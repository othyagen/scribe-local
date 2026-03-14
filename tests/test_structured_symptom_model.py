"""Tests for Layer 3: Structured symptom model.

Tests conservative linking — prefers null/empty over speculative filling.
"""

from __future__ import annotations

import pytest

from app.structured_symptom_model import build_structured_symptoms


# ── helpers ──────────────────────────────────────────────────────


def _minimal_state(**overrides) -> dict:
    """Minimal clinical_state with overrides."""
    base: dict = {
        "symptoms": [],
        "qualifiers": [],
        "timeline": [],
        "durations": [],
        "negations": [],
        "medications": [],
        "observations": [],
        "sites": [],
        "intensities": [],
        "ice": {"ideas": [], "concerns": [], "expectations": []},
        "derived": {
            "red_flags": [],
            "symptom_representations": [],
        },
    }
    base.update(overrides)
    return base


def _obs(finding_type: str, value: str, seg_id: str,
         t_start: float = 0.0, obs_id: str = "obs_0001",
         source_text: str | None = None) -> dict:
    return {
        "observation_id": obs_id,
        "finding_type": finding_type,
        "value": value,
        "seg_id": seg_id,
        "speaker_id": "spk_0",
        "t_start": t_start,
        "t_end": t_start + 2.0,
        "source_text": source_text if source_text is not None else f"text with {value}",
    }


# ── empty state ──────────────────────────────────────────────────


class TestEmptyState:
    def test_empty_symptoms(self):
        assert build_structured_symptoms(_minimal_state()) == []

    def test_no_observations(self):
        result = build_structured_symptoms(_minimal_state(symptoms=["headache"]))
        assert len(result) == 1
        assert result[0]["symptom"] == "headache"


# ── structure ────────────────────────────────────────────────────


class TestStructure:
    def test_has_all_domains(self):
        result = build_structured_symptoms(
            _minimal_state(symptoms=["headache"]),
        )
        ss = result[0]
        assert "spatial" in ss
        assert "qualitative" in ss
        assert "temporal" in ss
        assert "modifiers" in ss
        assert "context" in ss
        assert "safety" in ss
        assert "patient_perspective" in ss
        assert "observation_ids" in ss

    def test_spatial_keys(self):
        result = build_structured_symptoms(
            _minimal_state(symptoms=["headache"]),
        )
        spatial = result[0]["spatial"]
        assert set(spatial.keys()) == {"site", "laterality", "radiation"}

    def test_qualitative_keys(self):
        result = build_structured_symptoms(
            _minimal_state(symptoms=["headache"]),
        )
        qual = result[0]["qualitative"]
        assert set(qual.keys()) == {"character", "intensity", "intensity_raw", "severity"}

    def test_temporal_keys(self):
        result = build_structured_symptoms(
            _minimal_state(symptoms=["headache"]),
        )
        temp = result[0]["temporal"]
        assert set(temp.keys()) == {"onset", "onset_type", "duration", "pattern", "progression"}

    def test_preserves_symptom_order(self):
        result = build_structured_symptoms(
            _minimal_state(symptoms=["fever", "headache", "nausea"]),
        )
        assert [r["symptom"] for r in result] == ["fever", "headache", "nausea"]


# ── qualifiers populated ─────────────────────────────────────────


class TestQualifiers:
    def test_severity_from_qualifiers(self):
        result = build_structured_symptoms(_minimal_state(
            symptoms=["headache"],
            qualifiers=[{
                "symptom": "headache",
                "qualifiers": {"severity": "severe"},
            }],
        ))
        assert result[0]["qualitative"]["severity"] == "severe"

    def test_character_from_qualifiers(self):
        result = build_structured_symptoms(_minimal_state(
            symptoms=["chest pain"],
            qualifiers=[{
                "symptom": "chest pain",
                "qualifiers": {"character": "sharp"},
            }],
        ))
        assert result[0]["qualitative"]["character"] == "sharp"

    def test_onset_from_qualifiers(self):
        result = build_structured_symptoms(_minimal_state(
            symptoms=["headache"],
            qualifiers=[{
                "symptom": "headache",
                "qualifiers": {"onset": "sudden"},
            }],
        ))
        assert result[0]["temporal"]["onset"] == "sudden"

    def test_no_qualifier_cross_contamination(self):
        result = build_structured_symptoms(_minimal_state(
            symptoms=["headache", "nausea"],
            qualifiers=[{
                "symptom": "headache",
                "qualifiers": {"severity": "severe", "laterality": "left"},
            }],
        ))
        assert result[1]["qualitative"]["severity"] is None
        assert result[1]["spatial"]["laterality"] is None

    def test_duration_from_timeline(self):
        result = build_structured_symptoms(_minimal_state(
            symptoms=["headache"],
            timeline=[
                {"symptom": "headache", "time_expression": "3 days", "t_start": 0.0},
            ],
        ))
        assert result[0]["temporal"]["duration"] == "3 days"

    def test_modifiers_from_qualifiers(self):
        result = build_structured_symptoms(_minimal_state(
            symptoms=["headache"],
            qualifiers=[{
                "symptom": "headache",
                "qualifiers": {
                    "aggravating_factors": ["light", "noise"],
                    "relieving_factors": ["rest"],
                },
            }],
        ))
        assert result[0]["modifiers"]["aggravating_factors"] == ["light", "noise"]
        assert result[0]["modifiers"]["relieving_factors"] == ["rest"]


# ── conservative site linking ─────────────────────────────────────


class TestSiteLinking:
    def test_site_linked_same_segment(self):
        result = build_structured_symptoms(_minimal_state(
            symptoms=["headache"],
            observations=[
                _obs("symptom", "headache", "seg_0001", obs_id="obs_0001",
                     source_text="frontal headache reported"),
            ],
            sites=[{"site": "frontal", "seg_id": "seg_0001", "speaker_id": "spk_0", "t_start": 0.0}],
        ))
        assert result[0]["spatial"]["site"] == "frontal"

    def test_site_not_linked_different_segment(self):
        """Site in segment A, symptom in segment B -> site is null."""
        result = build_structured_symptoms(_minimal_state(
            symptoms=["headache"],
            observations=[
                _obs("symptom", "headache", "seg_0001", obs_id="obs_0001"),
            ],
            sites=[{"site": "frontal", "seg_id": "seg_0002", "speaker_id": "spk_0", "t_start": 2.0}],
        ))
        assert result[0]["spatial"]["site"] is None

    def test_site_only_for_same_segment_symptom(self):
        """Multiple symptoms in different segments, site only in one."""
        result = build_structured_symptoms(_minimal_state(
            symptoms=["headache", "nausea"],
            observations=[
                _obs("symptom", "headache", "seg_0001", obs_id="obs_0001",
                     source_text="frontal headache reported"),
                _obs("symptom", "nausea", "seg_0002", obs_id="obs_0002"),
            ],
            sites=[{"site": "frontal", "seg_id": "seg_0001", "speaker_id": "spk_0", "t_start": 0.0}],
        ))
        assert result[0]["spatial"]["site"] == "frontal"
        assert result[1]["spatial"]["site"] is None


# ── conservative intensity linking ────────────────────────────────


class TestIntensityLinking:
    def test_intensity_linked_same_segment(self):
        result = build_structured_symptoms(_minimal_state(
            symptoms=["headache"],
            observations=[
                _obs("symptom", "headache", "seg_0001", obs_id="obs_0001"),
            ],
            intensities=[{
                "value": 7, "raw_text": "7/10", "scale": "numeric",
                "seg_id": "seg_0001", "speaker_id": "spk_0", "t_start": 0.0,
            }],
        ))
        assert result[0]["qualitative"]["intensity"] == 7
        assert result[0]["qualitative"]["intensity_raw"] == "7/10"

    def test_intensity_not_linked_different_segment(self):
        result = build_structured_symptoms(_minimal_state(
            symptoms=["headache"],
            observations=[
                _obs("symptom", "headache", "seg_0001", obs_id="obs_0001"),
            ],
            intensities=[{
                "value": 7, "raw_text": "7/10", "scale": "numeric",
                "seg_id": "seg_0002", "speaker_id": "spk_0", "t_start": 2.0,
            }],
        ))
        assert result[0]["qualitative"]["intensity"] is None


# ── context: associated_present ──────────────────────────────────


class TestAssociatedPresent:
    def test_same_segment_symptoms_associated(self):
        """Two symptoms in same segment -> each lists the other."""
        result = build_structured_symptoms(_minimal_state(
            symptoms=["headache", "nausea"],
            observations=[
                _obs("symptom", "headache", "seg_0001", obs_id="obs_0001"),
                _obs("symptom", "nausea", "seg_0001", obs_id="obs_0002"),
            ],
        ))
        assert "nausea" in result[0]["context"]["associated_present"]
        assert "headache" in result[1]["context"]["associated_present"]

    def test_different_segment_not_associated(self):
        """Symptoms in different segments -> NOT associated."""
        result = build_structured_symptoms(_minimal_state(
            symptoms=["headache", "nausea"],
            observations=[
                _obs("symptom", "headache", "seg_0001", obs_id="obs_0001"),
                _obs("symptom", "nausea", "seg_0002", obs_id="obs_0002"),
            ],
        ))
        assert result[0]["context"]["associated_present"] == []
        assert result[1]["context"]["associated_present"] == []


# ── context: associated_absent ───────────────────────────────────


class TestAssociatedAbsent:
    def test_negation_same_segment(self):
        result = build_structured_symptoms(_minimal_state(
            symptoms=["headache"],
            observations=[
                _obs("symptom", "headache", "seg_0001", obs_id="obs_0001"),
                _obs("negation", "No fever", "seg_0001", obs_id="obs_0002"),
            ],
        ))
        assert "No fever" in result[0]["context"]["associated_absent"]

    def test_negation_different_segment_not_linked(self):
        """Negation in segment A, symptom in segment B -> not linked."""
        result = build_structured_symptoms(_minimal_state(
            symptoms=["headache"],
            observations=[
                _obs("symptom", "headache", "seg_0001", obs_id="obs_0001"),
                _obs("negation", "No fever", "seg_0002", obs_id="obs_0002"),
            ],
        ))
        assert result[0]["context"]["associated_absent"] == []


# ── context: prior_episodes ──────────────────────────────────────


class TestPriorEpisodes:
    def test_prior_episode_detected(self):
        result = build_structured_symptoms(_minimal_state(
            symptoms=["headache"],
            observations=[{
                "observation_id": "obs_0001",
                "finding_type": "symptom",
                "value": "headache",
                "seg_id": "seg_0001",
                "speaker_id": "spk_0",
                "t_start": 0.0,
                "t_end": 2.0,
                "source_text": "headache, this has happened before",
            }],
        ))
        assert len(result[0]["context"]["prior_episodes"]) >= 1

    def test_no_prior_episode_phrase(self):
        result = build_structured_symptoms(_minimal_state(
            symptoms=["headache"],
            observations=[
                _obs("symptom", "headache", "seg_0001", obs_id="obs_0001"),
            ],
        ))
        assert result[0]["context"]["prior_episodes"] == []


# ── safety ───────────────────────────────────────────────────────


class TestSafety:
    def test_red_flags_present_linked(self):
        result = build_structured_symptoms(_minimal_state(
            symptoms=["headache"],
            derived={
                "red_flags": [{
                    "flag": "sudden_severe_headache",
                    "label": "Sudden severe headache",
                    "severity": "high",
                    "evidence": ["headache", "severity: severe", "onset: sudden"],
                }],
                "symptom_representations": [],
            },
        ))
        assert "Sudden severe headache" in result[0]["safety"]["red_flags_present"]

    def test_red_flags_absent_always_empty(self):
        result = build_structured_symptoms(
            _minimal_state(symptoms=["headache"]),
        )
        assert result[0]["safety"]["red_flags_absent"] == []


# ── patient_perspective ──────────────────────────────────────────


class TestPatientPerspective:
    def test_ice_linked_same_segment(self):
        result = build_structured_symptoms(_minimal_state(
            symptoms=["headache"],
            observations=[
                _obs("symptom", "headache", "seg_0001", obs_id="obs_0001"),
            ],
            ice={
                "ideas": [{"text": "I think it might be a migraine", "seg_id": "seg_0001", "speaker_id": "spk_0", "t_start": 0.0}],
                "concerns": [],
                "expectations": [],
            },
        ))
        assert "I think it might be a migraine" in result[0]["patient_perspective"]["ideas"]

    def test_ice_not_linked_different_segment(self):
        """ICE in segment A, symptom in segment B -> not linked."""
        result = build_structured_symptoms(_minimal_state(
            symptoms=["headache"],
            observations=[
                _obs("symptom", "headache", "seg_0001", obs_id="obs_0001"),
            ],
            ice={
                "ideas": [{"text": "I think it might be a migraine", "seg_id": "seg_0002", "speaker_id": "spk_0", "t_start": 2.0}],
                "concerns": [],
                "expectations": [],
            },
        ))
        assert result[0]["patient_perspective"]["ideas"] == []


# ── onset_type deferred ──────────────────────────────────────────


class TestDeferredFields:
    def test_onset_type_always_none(self):
        result = build_structured_symptoms(_minimal_state(
            symptoms=["headache"],
            qualifiers=[{
                "symptom": "headache",
                "qualifiers": {"onset": "sudden"},
            }],
        ))
        assert result[0]["temporal"]["onset_type"] is None


# ── observation_ids ──────────────────────────────────────────────


class TestObservationIds:
    def test_observation_ids_populated(self):
        result = build_structured_symptoms(_minimal_state(
            symptoms=["headache"],
            observations=[
                _obs("symptom", "headache", "seg_0001", obs_id="obs_0001"),
                _obs("symptom", "headache", "seg_0002", obs_id="obs_0003"),
                _obs("symptom", "nausea", "seg_0001", obs_id="obs_0002"),
            ],
        ))
        assert result[0]["observation_ids"] == ["obs_0001", "obs_0003"]


# ── determinism ──────────────────────────────────────────────────


class TestDeterminism:
    def test_identical_input_identical_output(self):
        state = _minimal_state(
            symptoms=["headache", "nausea"],
            qualifiers=[{
                "symptom": "headache",
                "qualifiers": {"severity": "severe"},
            }],
        )
        r1 = build_structured_symptoms(state)
        r2 = build_structured_symptoms(state)
        assert r1 == r2
