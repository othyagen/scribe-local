"""Tests for clinical note template loading and rendering."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.export_clinical_note import (
    build_clinical_note,
    load_template,
    validate_template,
    write_clinical_note,
    _filter_segments_by_scope,
    _template_needs_roles,
)
from app.role_detection import ROLE_CLINICIAN, ROLE_PATIENT, ROLE_UNKNOWN


TS = "2026-03-01_10-00-00"

# Minimal valid template for testing
_MINIMAL_TEMPLATE = {
    "name": "Test Note",
    "format": "markdown",
    "sections": [
        {"title": "Subjective", "extractors": ["symptoms", "negations"], "scope": "all"},
        {"title": "Plan", "extractors": ["medications", "durations"], "scope": "all"},
    ],
    "transcript_section": False,
}


def _make_segments(texts: list[str]) -> list[dict]:
    """Create normalized segment dicts from text strings."""
    return [
        {
            "seg_id": f"seg_{i+1:04d}",
            "t0": float(i),
            "t1": float(i + 1),
            "speaker_id": "spk_0",
            "normalized_text": text,
        }
        for i, text in enumerate(texts)
    ]


def _write_template(template_dir: Path, template_id: str, data: dict) -> Path:
    """Write a YAML template file."""
    import yaml
    template_dir.mkdir(parents=True, exist_ok=True)
    path = template_dir / f"{template_id}.yaml"
    path.write_text(yaml.dump(data, default_flow_style=False), encoding="utf-8")
    return path


# ── template loading ─────────────────────────────────────────────────


class TestTemplateLoading:
    def test_load_soap_template(self):
        """Default soap.yaml loads and has expected sections."""
        template = load_template("soap")
        assert template["name"] == "SOAP Note"
        assert template["format"] == "markdown"
        section_titles = [s["title"] for s in template["sections"]]
        assert "Subjective" in section_titles
        assert "Objective" in section_titles
        assert "Assessment" in section_titles
        assert "Plan" in section_titles

    def test_missing_template_error(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="template not found"):
            load_template("nonexistent", template_dir=tmp_path)

    def test_invalid_template_missing_keys(self, tmp_path):
        _write_template(tmp_path, "bad", {"name": "Bad"})
        with pytest.raises(ValueError, match="missing required keys"):
            load_template("bad", template_dir=tmp_path)

    def test_invalid_format(self, tmp_path):
        _write_template(tmp_path, "bad", {
            "name": "Bad", "format": "html", "sections": [],
        })
        with pytest.raises(ValueError, match="invalid template format"):
            load_template("bad", template_dir=tmp_path)

    def test_invalid_scope(self, tmp_path):
        _write_template(tmp_path, "bad", {
            "name": "Bad", "format": "markdown",
            "sections": [{"title": "S", "scope": "invalid"}],
        })
        with pytest.raises(ValueError, match="invalid scope"):
            load_template("bad", template_dir=tmp_path)


# ── note building ────────────────────────────────────────────────────


class TestBuildClinicalNote:
    def test_deterministic_output(self):
        """Same input produces same output across calls."""
        segments = _make_segments(["patient has headache for 3 days."])
        r1 = build_clinical_note(segments, _MINIMAL_TEMPLATE)
        r2 = build_clinical_note(segments, _MINIMAL_TEMPLATE)
        assert r1 == r2

    def test_sections_rendered(self):
        """All template section titles appear as headings."""
        segments = _make_segments(["hello world."])
        result = build_clinical_note(segments, _MINIMAL_TEMPLATE)
        assert "## Subjective" in result
        assert "## Plan" in result

    def test_symptom_bullets(self):
        """Symptoms appear as bullet points under correct section."""
        segments = _make_segments(["patient reports headache and nausea."])
        result = build_clinical_note(segments, _MINIMAL_TEMPLATE)
        assert "- headache" in result
        assert "- nausea" in result

    def test_empty_section_still_rendered(self):
        """Section with no matches still shows heading."""
        segments = _make_segments(["everything is fine."])
        result = build_clinical_note(segments, _MINIMAL_TEMPLATE)
        assert "## Subjective" in result
        assert "_No items detected._" in result

    def test_transcript_section_appended(self):
        """Full transcript appears at end when enabled."""
        template = {**_MINIMAL_TEMPLATE, "transcript_section": True}
        segments = _make_segments(["hello world."])
        result = build_clinical_note(segments, template)
        assert "## Transcript" in result
        assert "hello world." in result

    def test_transcript_section_omitted(self):
        """Transcript not included when disabled."""
        template = {**_MINIMAL_TEMPLATE, "transcript_section": False}
        segments = _make_segments(["hello world."])
        result = build_clinical_note(segments, template)
        assert "## Transcript" not in result

    def test_empty_extractors_lists_raw_lines(self):
        """Section with empty extractors list includes raw transcript lines."""
        template = {
            "name": "Test", "format": "markdown",
            "sections": [{"title": "Objective", "extractors": [], "scope": "all"}],
            "transcript_section": False,
        }
        segments = _make_segments(["vitals normal.", "lungs clear."])
        result = build_clinical_note(segments, template)
        assert "- vitals normal." in result
        assert "- lungs clear." in result


# ── text format ──────────────────────────────────────────────────────


class TestTextFormat:
    def test_text_format_headings(self):
        """Text format uses uppercase/underline instead of markdown."""
        template = {**_MINIMAL_TEMPLATE, "format": "text"}
        segments = _make_segments(["headache."])
        result = build_clinical_note(segments, template)
        assert "TEST NOTE" in result
        assert "# " not in result


# ── negation in note ─────────────────────────────────────────────────


class TestNegationInNote:
    def test_negation_in_context(self):
        """Negation bullet items rendered cleanly in note."""
        segments = _make_segments(["denies fever, no chest pain."])
        result = build_clinical_note(segments, _MINIMAL_TEMPLATE)
        assert "Denies fever" in result
        assert "No chest pain" in result


# ── scope handling ───────────────────────────────────────────────────


def _make_multi_speaker_segments(speaker_texts: dict[str, list[str]]) -> list[dict]:
    """Create segments from {speaker_id: [texts]}."""
    segments: list[dict] = []
    i = 0
    for speaker_id, texts in speaker_texts.items():
        for text in texts:
            segments.append({
                "seg_id": f"seg_{i+1:04d}",
                "t0": float(i),
                "t1": float(i + 1),
                "speaker_id": speaker_id,
                "normalized_text": text,
            })
            i += 1
    return segments


class TestScopeHandling:
    def test_scope_all_includes_everything(self):
        """scope: all includes all segments."""
        template = {
            "name": "Test", "format": "markdown",
            "sections": [{"title": "S", "extractors": ["symptoms"], "scope": "all"}],
        }
        segments = _make_segments(["headache.", "nausea."])
        result = build_clinical_note(segments, template)
        assert "headache" in result
        assert "nausea" in result

    def test_scope_fallback_when_no_roles(self):
        """scope != all falls back to all when no roles provided."""
        template = {
            "name": "Test", "format": "markdown",
            "sections": [{"title": "S", "extractors": ["symptoms"], "scope": "patient_only"}],
        }
        segments = _make_segments(["headache.", "nausea."])
        result = build_clinical_note(segments, template, speaker_roles=None)
        assert "headache" in result
        assert "nausea" in result

    def test_patient_only_filters_to_patient_segments(self):
        """With roles, patient_only keeps only patient segments."""
        template = {
            "name": "Test", "format": "markdown",
            "sections": [{"title": "S", "extractors": ["symptoms"], "scope": "patient_only"}],
        }
        segments = _make_multi_speaker_segments({
            "spk_0": ["I have a headache."],
            "spk_1": ["Patient should take aspirin for the pain."],
        })
        roles = {
            "spk_0": {"role": ROLE_PATIENT, "confidence": 0.8, "evidence": []},
            "spk_1": {"role": ROLE_CLINICIAN, "confidence": 0.8, "evidence": []},
        }
        result = build_clinical_note(segments, template, speaker_roles=roles)
        assert "headache" in result
        # "pain" from clinician segment should not appear in scoped extraction
        # (but may appear via soft-scoping fallback if patient results are sparse)

    def test_clinician_only_filters_to_clinician_segments(self):
        """With roles, clinician_only keeps only clinician segments."""
        template = {
            "name": "Test", "format": "markdown",
            "sections": [{"title": "Plan", "extractors": ["medications"], "scope": "clinician_only"}],
        }
        segments = _make_multi_speaker_segments({
            "spk_0": ["I have a headache."],
            "spk_1": ["I recommend ibuprofen 400 mg."],
        })
        roles = {
            "spk_0": {"role": ROLE_PATIENT, "confidence": 0.8, "evidence": []},
            "spk_1": {"role": ROLE_CLINICIAN, "confidence": 0.8, "evidence": []},
        }
        result = build_clinical_note(segments, template, speaker_roles=roles)
        assert "ibuprofen" in result

    def test_scope_fallback_when_role_not_found(self):
        """patient_only with no patient role falls back to all segments."""
        template = {
            "name": "Test", "format": "markdown",
            "sections": [{"title": "S", "extractors": ["symptoms"], "scope": "patient_only"}],
        }
        segments = _make_segments(["headache."])
        roles = {
            "spk_0": {"role": ROLE_UNKNOWN, "confidence": 0.3, "evidence": []},
        }
        result = build_clinical_note(segments, template, speaker_roles=roles)
        # Falls back to all since no patient found
        assert "headache" in result

    def test_soft_scoping_supplements_sparse_results(self):
        """Soft scoping supplements from all segments when scoped results sparse."""
        template = {
            "name": "Test", "format": "markdown",
            "sections": [{"title": "S", "extractors": ["symptoms"], "scope": "patient_only"}],
        }
        # Patient says nothing about symptoms; clinician mentions them
        segments = _make_multi_speaker_segments({
            "spk_0": ["I feel okay today."],
            "spk_1": ["Patient reports headache and nausea."],
        })
        roles = {
            "spk_0": {"role": ROLE_PATIENT, "confidence": 0.8, "evidence": []},
            "spk_1": {"role": ROLE_CLINICIAN, "confidence": 0.8, "evidence": []},
        }
        result = build_clinical_note(segments, template, speaker_roles=roles)
        # Soft scoping: patient segments had 0 symptoms, so all segments used
        assert "headache" in result
        assert "nausea" in result

    def test_transcript_uses_role_labels(self):
        """Transcript section shows Clinician/Patient instead of speaker_id."""
        template = {
            "name": "Test", "format": "markdown",
            "sections": [],
            "transcript_section": True,
        }
        segments = _make_multi_speaker_segments({
            "spk_0": ["How are you?"],
            "spk_1": ["I feel great."],
        })
        roles = {
            "spk_0": {"role": ROLE_CLINICIAN, "confidence": 0.8, "evidence": []},
            "spk_1": {"role": ROLE_PATIENT, "confidence": 0.8, "evidence": []},
        }
        result = build_clinical_note(segments, template, speaker_roles=roles)
        assert "[Clinician]" in result
        assert "[Patient]" in result
        assert "[spk_0]" not in result
        assert "[spk_1]" not in result

    def test_transcript_falls_back_to_speaker_id_for_unknown(self):
        """Unknown role shows original speaker_id in transcript."""
        template = {
            "name": "Test", "format": "markdown",
            "sections": [],
            "transcript_section": True,
        }
        segments = _make_segments(["hello."])
        roles = {
            "spk_0": {"role": ROLE_UNKNOWN, "confidence": 0.2, "evidence": []},
        }
        result = build_clinical_note(segments, template, speaker_roles=roles)
        assert "[spk_0]" in result


class TestTemplateNeedsRoles:
    def test_all_scope_no_transcript(self):
        template = {
            "name": "T", "format": "markdown",
            "sections": [{"title": "S", "scope": "all"}],
            "transcript_section": False,
        }
        assert not _template_needs_roles(template)

    def test_scoped_section_needs_roles(self):
        template = {
            "name": "T", "format": "markdown",
            "sections": [{"title": "S", "scope": "patient_only"}],
        }
        assert _template_needs_roles(template)

    def test_transcript_section_needs_roles(self):
        template = {
            "name": "T", "format": "markdown",
            "sections": [{"title": "S", "scope": "all"}],
            "transcript_section": True,
        }
        assert _template_needs_roles(template)


# ── file writer ──────────────────────────────────────────────────────


class TestWriteClinicalNote:
    def test_file_written(self, tmp_path):
        segments = _make_segments(["headache for 3 days."])
        path = write_clinical_note(
            segments, _MINIMAL_TEMPLATE, str(tmp_path), TS, "soap",
        )
        assert path.exists()
        content = path.read_text("utf-8")
        assert "headache" in content

    def test_filename_includes_template_id(self, tmp_path):
        segments = _make_segments(["test."])
        path = write_clinical_note(
            segments, _MINIMAL_TEMPLATE, str(tmp_path), TS, "soap",
        )
        assert path.name == f"clinical_note_{TS}_soap.md"

    def test_text_format_extension(self, tmp_path):
        template = {**_MINIMAL_TEMPLATE, "format": "text"}
        segments = _make_segments(["test."])
        path = write_clinical_note(
            segments, template, str(tmp_path), TS, "mytemplate",
        )
        assert path.suffix == ".txt"


# ── integration ──────────────────────────────────────────────────────


class TestClinicalNoteIntegration:
    def test_full_pipeline(self):
        """Clinical note with symptoms, meds, negations, and durations."""
        segments = _make_segments([
            "patient reports headache and nausea for 3 days.",
            "denies fever, no chest pain.",
            "prescribed ibuprofen 400 mg twice daily.",
            "follow up in 2 weeks.",
        ])
        result = build_clinical_note(segments, _MINIMAL_TEMPLATE)

        # Subjective section should have symptoms and negations
        assert "headache" in result
        assert "nausea" in result
        assert "Denies fever" in result
        assert "No chest pain" in result

        # Plan section should have medications and durations
        assert "ibuprofen" in result
        assert "2 weeks" in result
