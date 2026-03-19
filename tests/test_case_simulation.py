"""Tests for the case simulation runner — formatting and logic only."""

from __future__ import annotations

import pytest

from scripts.run_case_simulation import (
    _build_case_segments,
    _DEFAULT_CONFIG,
    classify_question_type,
    build_structured_answer,
    format_summary_text,
    format_hypotheses_text,
    format_red_flags_text,
    format_questions_text,
    format_metrics_text,
    format_pending_status,
    build_display_bundle,
    render_display,
    _get_questions_from_view,
)
from app.clinical_session import initialize_session, get_app_view, submit_answers
from app.clinical_metrics import derive_clinical_metrics


# ── case data ────────────────────────────────────────────────────────


class TestCaseData:
    def test_segments_non_empty(self):
        segs = _build_case_segments()
        assert len(segs) >= 2

    def test_segments_have_required_keys(self):
        for seg in _build_case_segments():
            assert "seg_id" in seg
            assert "t0" in seg
            assert "t1" in seg
            assert "speaker_id" in seg
            assert "normalized_text" in seg

    def test_default_config(self):
        assert _DEFAULT_CONFIG["mode"] == "assist"
        assert _DEFAULT_CONFIG["show_questions"] is True


# ── question type classification ─────────────────────────────────────


class TestClassifyQuestionType:
    def test_duration_keywords(self):
        assert classify_question_type("How long have you had cough?") == "duration"
        assert classify_question_type("When did it start?") == "duration"

    def test_severity_keywords(self):
        assert classify_question_type("How severe is the pain?") == "severity"
        assert classify_question_type("How bad is the cough?") == "severity"

    def test_allergy_keywords(self):
        assert classify_question_type("Do you have any known drug allergies?") == "allergy"

    def test_dosage_keywords(self):
        assert classify_question_type("What is the dosage?") == "dosage"
        assert classify_question_type("What medication are you taking?") == "dosage"

    def test_unsupported(self):
        assert classify_question_type("Tell me about your family.") is None

    def test_case_insensitive(self):
        assert classify_question_type("HOW LONG have you had fever?") == "duration"


# ── structured answer building ───────────────────────────────────────


class TestBuildStructuredAnswer:
    def test_duration_answer(self):
        q = {"question": "How long have you had cough?", "reason": "No duration recorded for cough"}
        ans = build_structured_answer(q, "3 days")
        assert ans is not None
        assert ans["type"] == "duration"
        assert ans["value"] == "3 days"
        assert ans["related"] == "cough"

    def test_severity_answer(self):
        q = {"question": "How severe is headache?", "reason": "No severity recorded for headache"}
        ans = build_structured_answer(q, "moderate")
        assert ans is not None
        assert ans["type"] == "severity"
        assert ans["value"] == "moderate"

    def test_allergy_answer(self):
        q = {"question": "Any allergies?", "reason": "Medications prescribed without documented allergy status"}
        ans = build_structured_answer(q, "penicillin")
        assert ans is not None
        assert ans["type"] == "allergy"
        assert ans["value"] == "penicillin"

    def test_unsupported_returns_none(self):
        q = {"question": "Tell me more.", "reason": "General"}
        assert build_structured_answer(q, "some text") is None

    def test_strips_answer_text(self):
        q = {"question": "How long?", "reason": "No duration recorded for fever"}
        ans = build_structured_answer(q, "  5 days  ")
        assert ans["value"] == "5 days"

    def test_related_from_question_dict(self):
        q = {"question": "How long?", "reason": "test", "related": "headache"}
        ans = build_structured_answer(q, "2 weeks")
        assert ans["related"] == "headache"


# ── formatting functions ─────────────────────────────────────────────


class TestFormatting:
    def test_format_summary_empty(self):
        text = format_summary_text({})
        assert "SUMMARY" in text

    def test_format_summary_with_narrative(self):
        view = {
            "orchestrated": {
                "visible_outputs": {
                    "clinical_summary": {
                        "problem_narrative": {
                            "narrative": "Patient presents with cough.",
                            "positive_features": [],
                            "negative_features": [],
                        },
                    },
                },
            },
        }
        text = format_summary_text(view)
        assert "cough" in text

    def test_format_hypotheses_empty(self):
        text = format_hypotheses_text({})
        assert "HYPOTHESES" in text
        assert "none" in text.lower()

    def test_format_hypotheses_with_data(self):
        view = {
            "orchestrated": {
                "visible_outputs": {
                    "clinical_summary": {
                        "ranked_hypotheses": [
                            {"title": "Pneumonia", "score": 3, "rank": 1, "confidence": "moderate"},
                        ],
                    },
                },
            },
        }
        text = format_hypotheses_text(view)
        assert "Pneumonia" in text

    def test_format_red_flags_none(self):
        text = format_red_flags_text({})
        assert "None" in text

    def test_format_red_flags_with_data(self):
        view = {
            "orchestrated": {
                "visible_outputs": {
                    "clinical_summary": {
                        "red_flags": [{"label": "Sepsis risk", "evidence": ["fever"]}],
                    },
                },
            },
        }
        text = format_red_flags_text(view)
        assert "Sepsis" in text

    def test_format_questions_empty(self):
        text = format_questions_text([])
        assert "no questions" in text.lower()

    def test_format_questions_with_data(self):
        qs = [{"question": "How long?", "reason": "test", "priority": "medium"}]
        text = format_questions_text(qs)
        assert "1." in text
        assert "How long?" in text

    def test_format_metrics(self):
        metrics = derive_clinical_metrics({})
        text = format_metrics_text(metrics)
        assert "METRICS" in text
        assert "observations:" in text

    def test_format_pending_none(self):
        text = format_pending_status({"pending_observations": []})
        assert "No pending" in text

    def test_format_pending_with_items(self):
        text = format_pending_status({"pending_observations": [{"id": 1}, {"id": 2}]})
        assert "2 pending" in text
        assert "update" in text.lower()


# ── display bundle ───────────────────────────────────────────────────


class TestDisplayBundle:
    def test_bundle_has_expected_keys(self):
        bundle = build_display_bundle({}, [], derive_clinical_metrics({}), {"pending_observations": []})
        assert set(bundle.keys()) == {
            "summary", "hypotheses", "red_flags",
            "questions", "metrics", "pending_status",
        }

    def test_all_values_are_strings(self):
        bundle = build_display_bundle({}, [], derive_clinical_metrics({}), {"pending_observations": []})
        for key, val in bundle.items():
            assert isinstance(val, str), f"{key} should be a string"

    def test_render_display(self):
        bundle = build_display_bundle({}, [], derive_clinical_metrics({}), {"pending_observations": []})
        output = render_display(bundle)
        assert isinstance(output, str)
        assert "SUMMARY" in output
        assert "METRICS" in output


# ── integration ──────────────────────────────────────────────────────


class TestIntegration:
    def test_full_init_and_view(self):
        segments = _build_case_segments()
        session = initialize_session(segments, config=_DEFAULT_CONFIG)
        app_view = get_app_view(session)
        metrics = derive_clinical_metrics(session["clinical_state"])
        questions = _get_questions_from_view(app_view)

        bundle = build_display_bundle(app_view, questions, metrics, session)
        output = render_display(bundle)

        assert "SUMMARY" in output
        assert "HYPOTHESES" in output
        assert "METRICS" in output

    def test_answer_and_update_cycle(self):
        segments = _build_case_segments()
        session = initialize_session(segments, config=_DEFAULT_CONFIG)

        app_view = get_app_view(session)
        questions = _get_questions_from_view(app_view)

        # Find a duration question if available.
        duration_q = None
        for q in questions:
            if classify_question_type(q.get("question", "")) == "duration":
                duration_q = q
                break

        if duration_q:
            answer = build_structured_answer(duration_q, "3 days")
            assert answer is not None
            session = submit_answers(session, [answer])
            assert len(session.get("pending_observations", [])) >= 1

    def test_display_after_submit(self):
        segments = _build_case_segments()
        session = initialize_session(segments, config=_DEFAULT_CONFIG)

        # Submit a manual answer.
        session = submit_answers(session, [
            {"type": "allergy", "value": "penicillin"},
        ])

        app_view = get_app_view(session)
        metrics = derive_clinical_metrics(session["clinical_state"])
        questions = _get_questions_from_view(app_view)
        bundle = build_display_bundle(app_view, questions, metrics, session)
        output = render_display(bundle)

        assert "pending" in output.lower()
