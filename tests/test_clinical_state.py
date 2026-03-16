"""Tests for structured clinical state assembly."""

from __future__ import annotations

import pytest

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


_EXPECTED_KEYS = {
    "symptoms",
    "durations",
    "negations",
    "medications",
    "timeline",
    "review_flags",
    "diagnostic_hints",
    "speaker_roles",
    "history",
    "qualifiers",
    "derived",
    "observations",
    "ice",
    "intensities",
    "sites",
    "encounters",
    "problems",
    "clinical_graph",
}


# ── structure tests ──────────────────────────────────────────────────


class TestStructure:
    def test_contains_expected_keys(self):
        state = build_clinical_state([_seg("hello.")])
        assert set(state.keys()) == _EXPECTED_KEYS

    def test_all_values_are_lists_except_roles_history_derived_ice(self):
        state = build_clinical_state([_seg("hello.")])
        for key in _EXPECTED_KEYS - {"speaker_roles", "history", "derived", "ice", "clinical_graph"}:
            assert isinstance(state[key], list), f"{key} should be a list"
        assert isinstance(state["history"], dict)
        assert isinstance(state["derived"], dict)
        assert isinstance(state["ice"], dict)

    def test_qualifiers_is_list(self):
        state = build_clinical_state([_seg("hello.")])
        assert isinstance(state["qualifiers"], list)

    def test_empty_segments(self):
        state = build_clinical_state([])
        assert set(state.keys()) == _EXPECTED_KEYS
        assert state["symptoms"] == []
        assert state["timeline"] == []
        assert state["diagnostic_hints"] == []
        assert state["speaker_roles"] is None


# ── extractor outputs ────────────────────────────────────────────────


class TestExtractorOutputs:
    def test_symptoms_extracted(self):
        state = build_clinical_state([_seg("patient has headache and nausea.")])
        assert "headache" in state["symptoms"]
        assert "nausea" in state["symptoms"]

    def test_durations_extracted(self):
        state = build_clinical_state([_seg("headache for 3 days.")])
        assert "3 days" in state["durations"]

    def test_negations_extracted(self):
        state = build_clinical_state([_seg("denies fever, no chest pain.")])
        negation_lower = [n.lower() for n in state["negations"]]
        assert any("fever" in n for n in negation_lower)
        assert any("chest pain" in n for n in negation_lower)

    def test_medications_extracted(self):
        state = build_clinical_state([_seg("prescribed ibuprofen 400 mg.")])
        assert any("ibuprofen" in m for m in state["medications"])


# ── timeline ─────────────────────────────────────────────────────────


class TestTimeline:
    def test_timeline_included(self):
        state = build_clinical_state([_seg("headache for 3 days.")])
        assert len(state["timeline"]) >= 1
        entry = state["timeline"][0]
        assert entry["symptom"] == "headache"
        assert entry["time_expression"] == "3 days"

    def test_timeline_without_time(self):
        state = build_clinical_state([_seg("patient reports nausea.")])
        assert len(state["timeline"]) >= 1
        entry = state["timeline"][0]
        assert entry["symptom"] == "nausea"
        assert entry["time_expression"] is None


# ── review flags ─────────────────────────────────────────────────────


class TestReviewFlags:
    def test_review_flags_included(self):
        state = build_clinical_state([_seg("prescribed ibuprofen.")])
        # ibuprofen without dosage should trigger a flag
        flag_types = [f["type"] for f in state["review_flags"]]
        assert "medication_without_dosage" in flag_types

    def test_confidence_flags_forwarded(self):
        confidence = [{"seg_id": "seg_0001", "avg_logprob": -2.0}]
        state = build_clinical_state(
            [_seg("hello.")], confidence_entries=confidence,
        )
        flag_types = [f["type"] for f in state["review_flags"]]
        assert "low_confidence_segment" in flag_types

    def test_no_confidence_entries(self):
        state = build_clinical_state([_seg("hello.")])
        # Should not crash; review_flags is still a list
        assert isinstance(state["review_flags"], list)


# ── diagnostic hints ─────────────────────────────────────────────────


class TestDiagnosticHints:
    def test_hints_generated(self):
        state = build_clinical_state([
            _seg("patient has fever and sore throat."),
        ])
        conditions = [h["condition"] for h in state["diagnostic_hints"]]
        assert "Pharyngitis" in conditions

    def test_negation_suppresses_hint(self):
        state = build_clinical_state([
            _seg("patient has sore throat. no fever."),
        ])
        conditions = [h["condition"] for h in state["diagnostic_hints"]]
        assert "Pharyngitis" not in conditions

    def test_no_hints_when_no_matching_symptoms(self):
        state = build_clinical_state([_seg("patient feels fine.")])
        assert state["diagnostic_hints"] == []


# ── speaker roles ────────────────────────────────────────────────────


class TestSpeakerRoles:
    def test_roles_none_by_default(self):
        state = build_clinical_state([_seg("hello.")])
        assert state["speaker_roles"] is None

    def test_roles_passed_through(self):
        roles = {
            "spk_0": {"role": "clinician", "confidence": 0.9, "evidence": []},
        }
        state = build_clinical_state([_seg("hello.")], speaker_roles=roles)
        assert state["speaker_roles"] == roles

    def test_empty_roles_dict(self):
        state = build_clinical_state([_seg("hello.")], speaker_roles={})
        assert state["speaker_roles"] == {}


# ── integration ──────────────────────────────────────────────────────


class TestIntegration:
    def test_full_clinical_scenario(self):
        """A realistic clinical transcript produces a coherent state."""
        segments = [
            _seg("patient reports headache and nausea for 3 days.",
                 seg_id="seg_0001", t0=0.0, t1=3.0),
            _seg("denies fever, no chest pain.",
                 seg_id="seg_0002", t0=3.0, t1=5.0),
            _seg("prescribed ibuprofen 400 mg twice daily.",
                 seg_id="seg_0003", t0=5.0, t1=8.0),
            _seg("follow up in 2 weeks.",
                 seg_id="seg_0004", t0=8.0, t1=10.0),
        ]
        roles = {
            "spk_0": {"role": "patient", "confidence": 0.8, "evidence": []},
        }
        state = build_clinical_state(segments, speaker_roles=roles)

        assert "headache" in state["symptoms"]
        assert "nausea" in state["symptoms"]
        assert "3 days" in state["durations"]
        assert any("fever" in n.lower() for n in state["negations"])
        assert any("ibuprofen" in m for m in state["medications"])
        assert len(state["timeline"]) >= 1
        assert isinstance(state["review_flags"], list)
        assert isinstance(state["diagnostic_hints"], list)
        assert state["speaker_roles"] == roles


# ── 5-layer model ───────────────────────────────────────────────


class TestFiveLayerModel:
    def test_observations_is_list(self):
        state = build_clinical_state([_seg("patient has headache.")])
        assert isinstance(state["observations"], list)

    def test_ice_is_dict_with_keys(self):
        state = build_clinical_state([_seg("hello.")])
        assert isinstance(state["ice"], dict)
        assert set(state["ice"].keys()) == {"ideas", "concerns", "expectations"}

    def test_intensities_is_list(self):
        state = build_clinical_state([_seg("hello.")])
        assert isinstance(state["intensities"], list)

    def test_sites_is_list(self):
        state = build_clinical_state([_seg("hello.")])
        assert isinstance(state["sites"], list)

    def test_structured_symptoms_in_derived(self):
        state = build_clinical_state([_seg("patient has headache.")])
        assert "structured_symptoms" in state["derived"]
        assert isinstance(state["derived"]["structured_symptoms"], list)

    def test_problem_narrative_in_derived(self):
        state = build_clinical_state([_seg("patient has headache.")])
        assert "problem_narrative" in state["derived"]
        narrative = state["derived"]["problem_narrative"]
        assert isinstance(narrative, dict)
        assert "positive_features" in narrative
        assert "negative_features" in narrative
        assert "narrative" in narrative

    def test_clinical_graph_present(self):
        state = build_clinical_state([_seg("patient has headache.")])
        assert "clinical_graph" in state
        cg = state["clinical_graph"]
        assert isinstance(cg, dict)
        assert "nodes" in cg
        assert "edges" in cg
        assert isinstance(cg["nodes"], list)
        assert isinstance(cg["edges"], list)

    def test_clinical_graph_has_symptom_node(self):
        state = build_clinical_state([_seg("patient has headache.")])
        nodes = state["clinical_graph"]["nodes"]
        symptom_nodes = [n for n in nodes if n["node_type"] == "symptom"]
        assert len(symptom_nodes) >= 1
        assert any(n["value"] == "headache" for n in symptom_nodes)

    def test_clinical_graph_empty_for_no_symptoms(self):
        state = build_clinical_state([_seg("hello.")])
        cg = state["clinical_graph"]
        # Encounter node still present even without symptoms
        non_enc_prob = [n for n in cg["nodes"]
                        if n["node_type"] not in ("encounter", "problem")]
        assert non_enc_prob == []
        assert cg["edges"] == []

    def test_backward_compatibility(self):
        """All existing keys still present and unchanged."""
        state = build_clinical_state([_seg("patient has headache.")])
        existing_keys = {
            "symptoms", "durations", "negations", "medications",
            "timeline", "review_flags", "diagnostic_hints",
            "speaker_roles", "history", "qualifiers", "derived",
        }
        assert existing_keys.issubset(set(state.keys()))
        derived_existing = {
            "problem_representation", "problem_focus",
            "symptom_representations", "problem_summary",
            "ontology_concepts", "clinical_patterns",
        }
        assert derived_existing.issubset(set(state["derived"].keys()))

    def test_full_scenario_all_layers(self):
        """A realistic scenario produces coherent 5-layer output."""
        segments = [
            _seg("patient reports severe headache for 3 days.",
                 seg_id="seg_0001", t0=0.0, t1=3.0),
            _seg("also has nausea. denies fever.",
                 seg_id="seg_0002", t0=3.0, t1=5.0),
            _seg("prescribed ibuprofen 400 mg.",
                 seg_id="seg_0003", t0=5.0, t1=8.0),
        ]
        state = build_clinical_state(segments)

        # Layer 1: observations
        assert len(state["observations"]) >= 1

        # Layer 2: extractors
        assert isinstance(state["ice"], dict)
        assert isinstance(state["intensities"], list)
        assert isinstance(state["sites"], list)

        # Layer 3: structured symptoms
        ss = state["derived"]["structured_symptoms"]
        assert isinstance(ss, list)
        assert len(ss) >= 1

        # Layer 4: problem narrative
        pn = state["derived"]["problem_narrative"]
        assert isinstance(pn["positive_features"], list)
        assert isinstance(pn["negative_features"], list)
        assert isinstance(pn["narrative"], str)
