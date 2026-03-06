"""Tests for evidence linking — transcript-to-note traceability."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.export_clinical_note import (
    _fmt_evidence,
    _run_extractors_on_segments,
    build_clinical_note,
    write_clinical_note,
)


# ── fixtures ─────────────────────────────────────────────────────────

_SEGMENTS = [
    {
        "seg_id": "seg_0001",
        "t0": 0.0,
        "t1": 2.5,
        "speaker_id": "spk_0",
        "normalized_text": "I have had a headache for 3 days",
    },
    {
        "seg_id": "seg_0002",
        "t0": 3.0,
        "t1": 5.5,
        "speaker_id": "spk_1",
        "normalized_text": "any nausea or chest pain",
    },
    {
        "seg_id": "seg_0003",
        "t0": 6.0,
        "t1": 8.0,
        "speaker_id": "spk_0",
        "normalized_text": "no vomiting but I feel dizzy",
    },
]


def _make_template(
    extractors: list[str],
    show_evidence: bool = False,
    fmt: str = "markdown",
    scope: str = "all",
    transcript_section: bool = False,
) -> dict:
    return {
        "name": "Test Note",
        "format": fmt,
        "show_evidence": show_evidence,
        "sections": [
            {
                "title": "Findings",
                "extractors": extractors,
                "scope": scope,
            },
        ],
        "transcript_section": transcript_section,
    }


# ── _run_extractors_on_segments tests ────────────────────────────────


class TestRunExtractorsOnSegments:
    def test_returns_evidence_metadata(self):
        seen: set[str] = set()
        results = _run_extractors_on_segments(
            _SEGMENTS[:1], ["symptoms"], seen,
        )
        assert len(results) == 1
        assert results[0]["item"] == "headache"
        ev = results[0]["evidence"]
        assert ev["segment_id"] == "seg_0001"
        assert ev["speaker_id"] == "spk_0"
        assert ev["t_start"] == 0.0

    def test_multiple_segments(self):
        seen: set[str] = set()
        results = _run_extractors_on_segments(
            _SEGMENTS, ["symptoms"], seen,
        )
        items = [r["item"] for r in results]
        assert "headache" in items
        assert "nausea" in items
        assert "chest pain" in items

        # headache from seg_0001
        headache = next(r for r in results if r["item"] == "headache")
        assert headache["evidence"]["segment_id"] == "seg_0001"

        # nausea from seg_0002
        nausea = next(r for r in results if r["item"] == "nausea")
        assert nausea["evidence"]["segment_id"] == "seg_0002"

    def test_deduplication_first_occurrence_wins(self):
        """Same symptom in two segments — first occurrence keeps its evidence."""
        segs = [
            {"seg_id": "seg_0001", "t0": 0.0, "t1": 1.0,
             "speaker_id": "spk_0", "normalized_text": "headache"},
            {"seg_id": "seg_0002", "t0": 2.0, "t1": 3.0,
             "speaker_id": "spk_1", "normalized_text": "headache again"},
        ]
        seen: set[str] = set()
        results = _run_extractors_on_segments(segs, ["symptoms"], seen)
        assert len(results) == 1
        assert results[0]["evidence"]["segment_id"] == "seg_0001"

    def test_multiple_extractors(self):
        seen: set[str] = set()
        results = _run_extractors_on_segments(
            _SEGMENTS[:1], ["symptoms", "durations"], seen,
        )
        items = [r["item"] for r in results]
        assert "headache" in items
        assert "3 days" in items
        # Both from seg_0001
        for r in results:
            assert r["evidence"]["segment_id"] == "seg_0001"

    def test_empty_segments(self):
        seen: set[str] = set()
        results = _run_extractors_on_segments([], ["symptoms"], seen)
        assert results == []

    def test_no_matches(self):
        segs = [
            {"seg_id": "seg_0001", "t0": 0.0, "t1": 1.0,
             "speaker_id": "spk_0", "normalized_text": "the weather is nice"},
        ]
        seen: set[str] = set()
        results = _run_extractors_on_segments(segs, ["symptoms"], seen)
        assert results == []

    def test_negation_evidence(self):
        seen: set[str] = set()
        results = _run_extractors_on_segments(
            _SEGMENTS[2:], ["negations"], seen,
        )
        assert len(results) >= 1
        assert results[0]["evidence"]["segment_id"] == "seg_0003"
        assert results[0]["evidence"]["speaker_id"] == "spk_0"

    def test_missing_text_field_skipped(self):
        segs = [
            {"seg_id": "seg_0001", "t0": 0.0, "t1": 1.0, "speaker_id": "spk_0"},
        ]
        seen: set[str] = set()
        results = _run_extractors_on_segments(segs, ["symptoms"], seen)
        assert results == []


# ── _fmt_evidence tests ──────────────────────────────────────────────


class TestFmtEvidence:
    def test_full_evidence(self):
        ev = {"segment_id": "seg_0003", "speaker_id": "spk_1", "t_start": 102.5}
        result = _fmt_evidence(ev)
        assert result.startswith("[seg_0003, spk_1,")
        assert "]" in result

    def test_missing_speaker(self):
        ev = {"segment_id": "seg_0001", "t_start": 0.0}
        result = _fmt_evidence(ev)
        assert "seg_0001" in result
        assert result.startswith("[")
        assert result.endswith("]")

    def test_missing_t_start(self):
        ev = {"segment_id": "seg_0001", "speaker_id": "spk_0"}
        result = _fmt_evidence(ev)
        assert result == "[seg_0001, spk_0]"

    def test_empty_evidence(self):
        assert _fmt_evidence({}) == ""

    def test_all_empty_strings(self):
        ev = {"segment_id": "", "speaker_id": "", "t_start": None}
        assert _fmt_evidence(ev) == ""


# ── build_clinical_note with evidence ────────────────────────────────


class TestBuildNoteWithEvidence:
    def test_evidence_rendered_in_markdown(self):
        template = _make_template(["symptoms"], show_evidence=True)
        note = build_clinical_note(_SEGMENTS, template)
        assert "[seg_0001" in note
        assert "headache" in note
        # Evidence bracket on same line as item
        for line in note.splitlines():
            if "headache" in line and line.startswith("- "):
                assert "[seg_0001" in line
                break
        else:
            pytest.fail("headache line with evidence not found")

    def test_evidence_rendered_in_text_format(self):
        template = _make_template(
            ["symptoms"], show_evidence=True, fmt="text",
        )
        note = build_clinical_note(_SEGMENTS, template)
        assert "[seg_0001" in note
        assert "headache" in note

    def test_no_evidence_without_flag(self):
        template = _make_template(["symptoms"], show_evidence=False)
        note = build_clinical_note(_SEGMENTS, template)
        assert "headache" in note
        assert "[seg_" not in note

    def test_no_evidence_when_key_absent(self):
        """Backward compat: template without show_evidence key."""
        template = {
            "name": "Legacy",
            "format": "markdown",
            "sections": [
                {"title": "S", "extractors": ["symptoms"]},
            ],
        }
        note = build_clinical_note(_SEGMENTS, template)
        assert "headache" in note
        assert "[seg_" not in note

    def test_evidence_matches_correct_segment(self):
        template = _make_template(
            ["symptoms"], show_evidence=True,
        )
        note = build_clinical_note(_SEGMENTS, template)
        lines = note.splitlines()
        for line in lines:
            if "nausea" in line:
                assert "seg_0002" in line
                assert "spk_1" in line
                break
        else:
            pytest.fail("nausea line not found")

    def test_evidence_with_scoped_sections(self):
        """Scoped extraction evidence points to correct speaker's segment."""
        roles = {
            "spk_0": {"role": "patient", "confidence": 0.9, "evidence": {}},
            "spk_1": {"role": "clinician", "confidence": 0.9, "evidence": {}},
        }
        template = _make_template(
            ["symptoms"], show_evidence=True, scope="patient_only",
        )
        note = build_clinical_note(_SEGMENTS, template, speaker_roles=roles)
        # Patient segments are seg_0001, seg_0003 (spk_0)
        # headache is in seg_0001 (patient)
        assert "seg_0001" in note
        # nausea is in seg_0002 (clinician) — should NOT appear in scoped results
        # unless soft scoping kicks in
        lines = [l for l in note.splitlines() if "nausea" in l]
        if lines:
            # If nausea appears via soft scoping, it should reference seg_0002
            assert "seg_0002" in lines[0]

    def test_soft_scoping_evidence_from_actual_segment(self):
        """Supplemented items from fallback have evidence from their source."""
        # Only one segment for patient with no extractable symptoms
        segs = [
            {"seg_id": "seg_0001", "t0": 0.0, "t1": 1.0,
             "speaker_id": "spk_0", "normalized_text": "hello"},
            {"seg_id": "seg_0002", "t0": 2.0, "t1": 3.0,
             "speaker_id": "spk_1", "normalized_text": "headache noted"},
        ]
        roles = {
            "spk_0": {"role": "patient", "confidence": 0.9, "evidence": {}},
            "spk_1": {"role": "clinician", "confidence": 0.9, "evidence": {}},
        }
        template = _make_template(
            ["symptoms"], show_evidence=True, scope="patient_only",
        )
        note = build_clinical_note(segs, template, speaker_roles=roles)
        # headache should come from seg_0002 via soft scoping
        if "headache" in note:
            lines = [l for l in note.splitlines() if "headache" in l]
            assert "seg_0002" in lines[0]

    def test_no_items_detected_with_evidence(self):
        segs = [
            {"seg_id": "seg_0001", "t0": 0.0, "t1": 1.0,
             "speaker_id": "spk_0", "normalized_text": "the weather is nice"},
        ]
        template = _make_template(["symptoms"], show_evidence=True)
        note = build_clinical_note(segs, template)
        assert "No items detected" in note

    def test_evidence_graceful_with_missing_fields(self):
        """Segments missing optional fields don't crash."""
        segs = [
            {"normalized_text": "headache"},
        ]
        template = _make_template(["symptoms"], show_evidence=True)
        note = build_clinical_note(segs, template)
        assert "headache" in note
        # Should not crash even without seg_id, speaker_id, t0

    def test_sections_without_extractors_no_evidence(self):
        """Raw transcript sections (no extractors) never show evidence."""
        template = {
            "name": "Test",
            "format": "markdown",
            "show_evidence": True,
            "sections": [
                {"title": "Notes", "extractors": []},
            ],
        }
        note = build_clinical_note(_SEGMENTS, template)
        # Raw lines should not have [seg_ references
        assert "[seg_" not in note


# ── write_clinical_note with evidence ────────────────────────────────


class TestWriteNoteWithEvidence:
    def test_write_file_with_evidence(self, tmp_path):
        template = _make_template(["symptoms", "durations"], show_evidence=True)
        path = write_clinical_note(
            _SEGMENTS, template, str(tmp_path),
            "2026-03-01_10-00-00", "test",
        )
        content = path.read_text("utf-8")
        assert "headache" in content
        assert "[seg_0001" in content
        assert path.suffix == ".md"
