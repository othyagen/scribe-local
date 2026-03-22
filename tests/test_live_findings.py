"""Tests for live findings preview module."""

from __future__ import annotations

import io

import pytest

from app.streaming_buffer import StreamingBuffer
from app.live_findings import (
    LiveFindings,
    extract_findings_from_text,
    extract_qualifiers_from_segment,
    format_finding_line,
    format_qualifier_line,
)


# ── helpers ─────────────────────────────────────────────────────────


def _seg(
    text: str,
    seg_id: str = "seg_0001",
    t0: float = 0.0,
    t1: float = 1.0,
    speaker_id: str = "spk_0",
) -> dict:
    return {
        "seg_id": seg_id,
        "t0": t0,
        "t1": t1,
        "speaker_id": speaker_id,
        "normalized_text": text,
    }


def _make_live(buf: StreamingBuffer | None = None) -> tuple[LiveFindings, io.StringIO]:
    """Create a LiveFindings instance with captured output."""
    if buf is None:
        buf = StreamingBuffer()
    out = io.StringIO()
    return LiveFindings(buf, output=out), out


# ══════════════════════════════════════════════════════════════════
# extract_findings_from_text
# ══════════════════════════════════════════════════════════════════


class TestExtractFindingsFromText:
    def test_symptom_extracted(self):
        findings = extract_findings_from_text("patient has headache.")
        categories = [c for c, _ in findings]
        values = [v for _, v in findings]
        assert "symptom" in categories
        assert "headache" in values

    def test_negation_extracted(self):
        findings = extract_findings_from_text("no fever.")
        categories = [c for c, _ in findings]
        assert "negation" in categories

    def test_duration_extracted(self):
        findings = extract_findings_from_text("headache for 3 days.")
        values = [v for _, v in findings]
        assert "3 days" in values

    def test_medication_extracted(self):
        findings = extract_findings_from_text("prescribed ibuprofen.")
        values = [v for _, v in findings]
        assert "ibuprofen" in values

    def test_empty_text(self):
        assert extract_findings_from_text("") == []

    def test_no_findings(self):
        assert extract_findings_from_text("the weather is nice.") == []

    def test_multiple_categories(self):
        findings = extract_findings_from_text(
            "headache for 3 days, prescribed ibuprofen."
        )
        categories = {c for c, _ in findings}
        assert "symptom" in categories
        assert "duration" in categories
        assert "medication" in categories

    def test_stable_category_order(self):
        findings = extract_findings_from_text(
            "headache for 3 days, no fever, ibuprofen."
        )
        categories = [c for c, _ in findings]
        first_symptom = next(i for i, c in enumerate(categories) if c == "symptom")
        first_negation = next(i for i, c in enumerate(categories) if c == "negation")
        first_duration = next(i for i, c in enumerate(categories) if c == "duration")
        first_med = next(i for i, c in enumerate(categories) if c == "medication")
        assert first_symptom < first_negation < first_duration < first_med


# ══════════════════════════════════════════════════════════════════
# extract_qualifiers_from_segment
# ══════════════════════════════════════════════════════════════════


class TestExtractQualifiersFromSegment:
    def test_severity_extracted(self):
        seg = _seg("severe headache.")
        quals = extract_qualifiers_from_segment(seg, {"headache"})
        types = [t for t, _ in quals]
        assert "severity" in types
        values = [v for t, v in quals if t == "severity"]
        assert "severe" in values

    def test_onset_extracted(self):
        seg = _seg("sudden headache.")
        quals = extract_qualifiers_from_segment(seg, {"headache"})
        assert any(t == "onset" and v == "sudden" for t, v in quals)

    def test_laterality_extracted(self):
        seg = _seg("left-sided pain.")
        quals = extract_qualifiers_from_segment(seg, {"pain"})
        assert any(t == "laterality" and v == "left" for t, v in quals)

    def test_aggravating_factor_extracted(self):
        seg = _seg("headache worse with movement.")
        quals = extract_qualifiers_from_segment(seg, {"headache"})
        types = [t for t, _ in quals]
        assert "aggravating_factor" in types

    def test_relieving_factor_extracted(self):
        seg = _seg("headache relieved by rest.")
        quals = extract_qualifiers_from_segment(seg, {"headache"})
        types = [t for t, _ in quals]
        assert "relieving_factor" in types

    def test_no_symptoms_returns_empty(self):
        seg = _seg("severe and constant.")
        quals = extract_qualifiers_from_segment(seg, set())
        assert quals == []

    def test_no_qualifiers_returns_empty(self):
        seg = _seg("headache.")
        quals = extract_qualifiers_from_segment(seg, {"headache"})
        assert quals == []

    def test_radiation_extracted(self):
        seg = _seg("pain radiating to the back.")
        quals = extract_qualifiers_from_segment(seg, {"pain"})
        assert any(t == "radiation" for t, _ in quals)

    def test_pattern_extracted(self):
        seg = _seg("intermittent headache.")
        quals = extract_qualifiers_from_segment(seg, {"headache"})
        assert any(t == "pattern" and v == "intermittent" for t, v in quals)

    def test_progression_extracted(self):
        seg = _seg("worsening headache.")
        quals = extract_qualifiers_from_segment(seg, {"headache"})
        assert any(t == "progression" and v == "worsening" for t, v in quals)


# ══════════════════════════════════════════════════════════════════
# format_finding_line / format_qualifier_line
# ══════════════════════════════════════════════════════════════════


class TestFormatFindingLine:
    def test_basic_format(self):
        line = format_finding_line("symptom", "headache")
        assert line == "  + symptom: headache"

    def test_negation_format(self):
        line = format_finding_line("negation", "No fever")
        assert line == "  + negation: No fever"

    def test_duration_format(self):
        line = format_finding_line("duration", "3 days")
        assert line == "  + duration: 3 days"

    def test_medication_format(self):
        line = format_finding_line("medication", "ibuprofen 400 mg")
        assert line == "  + medication: ibuprofen 400 mg"


class TestFormatQualifierLine:
    def test_severity_format(self):
        line = format_qualifier_line("severity", "severe")
        assert line == "  + severity: severe"

    def test_onset_format(self):
        line = format_qualifier_line("onset", "sudden")
        assert line == "  + onset: sudden"

    def test_aggravating_factor_format(self):
        line = format_qualifier_line("aggravating_factor", "movement")
        assert line == "  + aggravating_factor: movement"

    def test_relieving_factor_format(self):
        line = format_qualifier_line("relieving_factor", "rest")
        assert line == "  + relieving_factor: rest"


# ══════════════════════════════════════════════════════════════════
# LiveFindings — unprocessed segment handling
# ══════════════════════════════════════════════════════════════════


class TestUnprocessedHandling:
    def test_no_segments_returns_empty(self):
        lf, _ = _make_live()
        assert lf.process_new_segments() == []

    def test_empty_text_segment_skipped(self):
        buf = StreamingBuffer()
        buf.append_segment(_seg("", seg_id="seg_0001"))
        lf, _ = _make_live(buf)
        assert lf.process_new_segments() == []

    def test_segment_without_findings(self):
        buf = StreamingBuffer()
        buf.append_segment(_seg("hello world.", seg_id="seg_0001"))
        lf, _ = _make_live(buf)
        assert lf.process_new_segments() == []


# ══════════════════════════════════════════════════════════════════
# LiveFindings — incremental extraction
# ══════════════════════════════════════════════════════════════════


class TestIncrementalExtraction:
    def test_single_segment_symptom(self):
        buf = StreamingBuffer()
        buf.append_segment(_seg("patient reports headache.", seg_id="seg_0001"))
        lf, out = _make_live(buf)
        new = lf.process_new_segments()
        assert any(c == "symptom" and v == "headache" for c, v in new)
        assert "headache" in out.getvalue()

    def test_single_segment_negation(self):
        buf = StreamingBuffer()
        buf.append_segment(_seg("denies fever.", seg_id="seg_0001"))
        lf, out = _make_live(buf)
        new = lf.process_new_segments()
        assert any(c == "negation" for c, _ in new)
        assert "negation" in out.getvalue()

    def test_single_segment_duration(self):
        buf = StreamingBuffer()
        buf.append_segment(_seg("headache for 3 days.", seg_id="seg_0001"))
        lf, _ = _make_live(buf)
        new = lf.process_new_segments()
        values = [v for c, v in new if c == "duration"]
        assert "3 days" in values

    def test_single_segment_medication(self):
        buf = StreamingBuffer()
        buf.append_segment(_seg("prescribed ibuprofen.", seg_id="seg_0001"))
        lf, _ = _make_live(buf)
        new = lf.process_new_segments()
        values = [v for c, v in new if c == "medication"]
        assert "ibuprofen" in values

    def test_two_segments_incremental(self):
        buf = StreamingBuffer()
        lf, out = _make_live(buf)

        buf.append_segment(_seg("headache.", seg_id="seg_0001", t0=0.0, t1=1.0))
        new1 = lf.process_new_segments()
        assert any(v == "headache" for _, v in new1)

        buf.append_segment(_seg("nausea for 2 days.", seg_id="seg_0002", t0=1.0, t1=2.0))
        new2 = lf.process_new_segments()
        assert any(v == "nausea" for _, v in new2)
        assert any(v == "2 days" for _, v in new2)
        assert not any(c == "symptom" and v == "headache" for c, v in new2)

    def test_three_segments_incremental(self):
        buf = StreamingBuffer()
        lf, _ = _make_live(buf)

        buf.append_segment(_seg("headache.", seg_id="seg_0001", t0=0.0, t1=1.0))
        lf.process_new_segments()

        buf.append_segment(_seg("nausea.", seg_id="seg_0002", t0=1.0, t1=2.0))
        lf.process_new_segments()

        buf.append_segment(_seg("dizziness.", seg_id="seg_0003", t0=2.0, t1=3.0))
        new3 = lf.process_new_segments()
        assert any(v == "dizziness" for _, v in new3)
        assert not any(v == "headache" for _, v in new3)
        assert not any(v == "nausea" for _, v in new3)


# ══════════════════════════════════════════════════════════════════
# LiveFindings — qualifier integration
# ══════════════════════════════════════════════════════════════════


class TestQualifierIntegration:
    def test_severity_detected_with_symptom(self):
        buf = StreamingBuffer()
        buf.append_segment(_seg("severe headache.", seg_id="seg_0001"))
        lf, out = _make_live(buf)
        new = lf.process_new_segments()

        # Both symptom and severity should appear
        assert any(c == "symptom" and v == "headache" for c, v in new)
        assert any(c == "severity" and v == "severe" for c, v in new)
        output = out.getvalue()
        assert "symptom: headache" in output
        assert "severity: severe" in output

    def test_onset_detected(self):
        buf = StreamingBuffer()
        buf.append_segment(_seg("sudden headache.", seg_id="seg_0001"))
        lf, _ = _make_live(buf)
        new = lf.process_new_segments()
        assert any(c == "onset" and v == "sudden" for c, v in new)

    def test_laterality_detected(self):
        buf = StreamingBuffer()
        buf.append_segment(_seg("left-sided chest pain.", seg_id="seg_0001"))
        lf, _ = _make_live(buf)
        new = lf.process_new_segments()
        assert any(c == "laterality" and v == "left" for c, v in new)

    def test_pattern_detected(self):
        buf = StreamingBuffer()
        buf.append_segment(_seg("intermittent headache.", seg_id="seg_0001"))
        lf, _ = _make_live(buf)
        new = lf.process_new_segments()
        assert any(c == "pattern" and v == "intermittent" for c, v in new)

    def test_progression_detected(self):
        buf = StreamingBuffer()
        buf.append_segment(_seg("worsening pain.", seg_id="seg_0001"))
        lf, _ = _make_live(buf)
        new = lf.process_new_segments()
        assert any(c == "progression" and v == "worsening" for c, v in new)

    def test_aggravating_factor_detected(self):
        buf = StreamingBuffer()
        buf.append_segment(_seg("headache worse with movement.", seg_id="seg_0001"))
        lf, out = _make_live(buf)
        new = lf.process_new_segments()
        assert any(c == "aggravating_factor" for c, _ in new)
        assert "aggravating_factor" in out.getvalue()

    def test_relieving_factor_detected(self):
        buf = StreamingBuffer()
        buf.append_segment(_seg("headache relieved by rest.", seg_id="seg_0001"))
        lf, out = _make_live(buf)
        new = lf.process_new_segments()
        assert any(c == "relieving_factor" for c, _ in new)
        assert "relieving_factor" in out.getvalue()

    def test_multiple_qualifiers_in_one_segment(self):
        buf = StreamingBuffer()
        buf.append_segment(_seg(
            "severe sudden left headache.",
            seg_id="seg_0001",
        ))
        lf, _ = _make_live(buf)
        new = lf.process_new_segments()
        types = {c for c, _ in new}
        assert "severity" in types
        assert "onset" in types
        assert "laterality" in types

    def test_qualifiers_in_second_segment(self):
        """Qualifiers for symptoms detected in earlier segments."""
        buf = StreamingBuffer()
        lf, _ = _make_live(buf)

        # First segment: symptom only
        buf.append_segment(_seg("headache.", seg_id="seg_0001", t0=0.0, t1=1.0))
        new1 = lf.process_new_segments()
        assert any(c == "symptom" and v == "headache" for c, v in new1)

        # Second segment: qualifier for known symptom
        buf.append_segment(_seg("severe headache.", seg_id="seg_0002", t0=1.0, t1=2.0))
        new2 = lf.process_new_segments()
        assert any(c == "severity" and v == "severe" for c, v in new2)

    def test_no_qualifiers_without_symptom(self):
        buf = StreamingBuffer()
        buf.append_segment(_seg("severe and constant.", seg_id="seg_0001"))
        lf, _ = _make_live(buf)
        new = lf.process_new_segments()
        assert not any(c == "severity" for c, _ in new)

    def test_qualifier_count(self):
        buf = StreamingBuffer()
        buf.append_segment(_seg("severe sudden headache.", seg_id="seg_0001"))
        lf, _ = _make_live(buf)
        lf.process_new_segments()
        assert lf.qualifier_count >= 2

    def test_all_qualifiers_property(self):
        buf = StreamingBuffer()
        buf.append_segment(_seg("severe headache.", seg_id="seg_0001"))
        lf, _ = _make_live(buf)
        lf.process_new_segments()
        all_q = lf.all_qualifiers
        assert any(t == "severity" and v == "severe" for t, v in all_q)

    def test_get_qualifiers_by_type(self):
        buf = StreamingBuffer()
        buf.append_segment(_seg("severe headache.", seg_id="seg_0001"))
        lf, _ = _make_live(buf)
        lf.process_new_segments()
        severity_vals = lf.get_qualifiers_by_type("severity")
        assert "severe" in severity_vals


# ══════════════════════════════════════════════════════════════════
# LiveFindings — duplicate prevention
# ══════════════════════════════════════════════════════════════════


class TestDuplicatePrevention:
    def test_same_symptom_in_two_segments(self):
        buf = StreamingBuffer()
        lf, _ = _make_live(buf)

        buf.append_segment(_seg("headache.", seg_id="seg_0001", t0=0.0, t1=1.0))
        new1 = lf.process_new_segments()
        assert any(v == "headache" for _, v in new1)

        buf.append_segment(_seg("still headache.", seg_id="seg_0002", t0=1.0, t1=2.0))
        new2 = lf.process_new_segments()
        assert not any(c == "symptom" and v == "headache" for c, v in new2)

    def test_case_insensitive_dedup(self):
        buf = StreamingBuffer()
        lf, _ = _make_live(buf)

        buf.append_segment(_seg("Headache.", seg_id="seg_0001", t0=0.0, t1=1.0))
        lf.process_new_segments()

        buf.append_segment(_seg("HEADACHE.", seg_id="seg_0002", t0=1.0, t1=2.0))
        new2 = lf.process_new_segments()
        assert not any(c == "symptom" and v == "headache" for c, v in new2)

    def test_different_category_same_value_not_deduped(self):
        buf = StreamingBuffer()
        lf, _ = _make_live(buf)

        buf.append_segment(_seg("headache.", seg_id="seg_0001", t0=0.0, t1=1.0))
        new1 = lf.process_new_segments()
        assert any(c == "symptom" and v == "headache" for c, v in new1)

        buf.append_segment(_seg("denies headache.", seg_id="seg_0002", t0=1.0, t1=2.0))
        new2 = lf.process_new_segments()
        assert any(c == "negation" for c, _ in new2)

    def test_all_findings_accumulates(self):
        buf = StreamingBuffer()
        lf, _ = _make_live(buf)

        buf.append_segment(_seg("headache.", seg_id="seg_0001", t0=0.0, t1=1.0))
        lf.process_new_segments()
        buf.append_segment(_seg("nausea.", seg_id="seg_0002", t0=1.0, t1=2.0))
        lf.process_new_segments()

        all_f = lf.all_findings
        values = [v for _, v in all_f]
        assert "headache" in values
        assert "nausea" in values

    def test_finding_count(self):
        buf = StreamingBuffer()
        lf, _ = _make_live(buf)

        buf.append_segment(_seg("headache for 3 days.", seg_id="seg_0001"))
        lf.process_new_segments()
        assert lf.finding_count >= 2

    def test_qualifier_dedup_across_segments(self):
        buf = StreamingBuffer()
        lf, _ = _make_live(buf)

        buf.append_segment(_seg("severe headache.", seg_id="seg_0001", t0=0.0, t1=1.0))
        new1 = lf.process_new_segments()
        assert any(c == "severity" and v == "severe" for c, v in new1)

        buf.append_segment(_seg("severe headache again.", seg_id="seg_0002", t0=1.0, t1=2.0))
        new2 = lf.process_new_segments()
        assert not any(c == "severity" and v == "severe" for c, v in new2)


# ══════════════════════════════════════════════════════════════════
# LiveFindings — processed marking
# ══════════════════════════════════════════════════════════════════


class TestProcessedMarking:
    def test_segments_marked_after_processing(self):
        buf = StreamingBuffer()
        buf.append_segment(_seg("headache.", seg_id="seg_0001"))
        lf, _ = _make_live(buf)
        lf.process_new_segments()
        assert buf.processed_count == 1
        assert buf.get_unprocessed_segments() == []

    def test_multiple_segments_all_marked(self):
        buf = StreamingBuffer()
        buf.append_segment(_seg("headache.", seg_id="seg_0001", t0=0.0, t1=1.0))
        buf.append_segment(_seg("nausea.", seg_id="seg_0002", t0=1.0, t1=2.0))
        lf, _ = _make_live(buf)
        lf.process_new_segments()
        assert buf.processed_count == 2

    def test_second_call_processes_only_new(self):
        buf = StreamingBuffer()
        lf, _ = _make_live(buf)

        buf.append_segment(_seg("headache.", seg_id="seg_0001", t0=0.0, t1=1.0))
        lf.process_new_segments()
        assert buf.processed_count == 1

        buf.append_segment(_seg("nausea.", seg_id="seg_0002", t0=1.0, t1=2.0))
        lf.process_new_segments()
        assert buf.processed_count == 2

    def test_empty_process_does_not_change_count(self):
        buf = StreamingBuffer()
        lf, _ = _make_live(buf)
        lf.process_new_segments()
        assert buf.processed_count == 0


# ══════════════════════════════════════════════════════════════════
# LiveFindings — console output
# ══════════════════════════════════════════════════════════════════


class TestConsoleOutput:
    def test_finding_printed_to_output(self):
        buf = StreamingBuffer()
        buf.append_segment(_seg("headache.", seg_id="seg_0001"))
        lf, out = _make_live(buf)
        lf.process_new_segments()
        output = out.getvalue()
        assert "+ symptom: headache" in output

    def test_multiple_findings_printed(self):
        buf = StreamingBuffer()
        buf.append_segment(_seg(
            "headache for 3 days, prescribed ibuprofen.",
            seg_id="seg_0001",
        ))
        lf, out = _make_live(buf)
        lf.process_new_segments()
        output = out.getvalue()
        assert "symptom" in output
        assert "duration" in output
        assert "medication" in output

    def test_no_output_for_no_findings(self):
        buf = StreamingBuffer()
        buf.append_segment(_seg("hello world.", seg_id="seg_0001"))
        lf, out = _make_live(buf)
        lf.process_new_segments()
        assert out.getvalue() == ""

    def test_duplicate_not_printed_again(self):
        buf = StreamingBuffer()
        lf, out = _make_live(buf)

        buf.append_segment(_seg("headache.", seg_id="seg_0001", t0=0.0, t1=1.0))
        lf.process_new_segments()

        buf.append_segment(_seg("headache again.", seg_id="seg_0002", t0=1.0, t1=2.0))
        lf.process_new_segments()
        assert out.getvalue().count("+ symptom: headache") == 1

    def test_qualifier_printed_to_output(self):
        buf = StreamingBuffer()
        buf.append_segment(_seg("severe headache.", seg_id="seg_0001"))
        lf, out = _make_live(buf)
        lf.process_new_segments()
        output = out.getvalue()
        assert "+ severity: severe" in output

    def test_findings_and_qualifiers_interleaved(self):
        buf = StreamingBuffer()
        buf.append_segment(_seg("severe headache.", seg_id="seg_0001"))
        lf, out = _make_live(buf)
        lf.process_new_segments()
        output = out.getvalue()
        # Findings printed before qualifiers
        symptom_pos = output.index("symptom: headache")
        severity_pos = output.index("severity: severe")
        assert symptom_pos < severity_pos


# ══════════════════════════════════════════════════════════════════
# LiveFindings — multi-segment flow
# ══════════════════════════════════════════════════════════════════


class TestMultiSegmentFlow:
    def test_realistic_session(self):
        """Simulate a realistic 4-segment recording session."""
        buf = StreamingBuffer()
        lf, out = _make_live(buf)

        # Segment 1: symptom + duration + qualifier
        buf.append_segment(_seg(
            "patient reports severe headache for 3 days.",
            seg_id="seg_0001", t0=0.0, t1=3.0,
        ))
        new1 = lf.process_new_segments()
        assert any(c == "symptom" and v == "headache" for c, v in new1)
        assert any(v == "3 days" for _, v in new1)
        assert any(c == "severity" and v == "severe" for c, v in new1)

        # Segment 2: negations
        buf.append_segment(_seg(
            "no fever, no vomiting.",
            seg_id="seg_0002", t0=3.0, t1=5.0,
        ))
        new2 = lf.process_new_segments()
        assert any(c == "negation" for c, _ in new2)
        assert not any(c == "symptom" and v == "headache" for c, v in new2)
        assert not any(v == "3 days" for _, v in new2)

        # Segment 3: medication
        buf.append_segment(_seg(
            "prescribed ibuprofen 400 mg.",
            seg_id="seg_0003", t0=5.0, t1=7.0,
        ))
        new3 = lf.process_new_segments()
        assert any(c == "medication" for c, _ in new3)

        # Segment 4: aggravating factor
        buf.append_segment(_seg(
            "headache worse with movement.",
            seg_id="seg_0004", t0=7.0, t1=9.0,
        ))
        new4 = lf.process_new_segments()
        assert any(c == "aggravating_factor" for c, _ in new4)
        # headache symptom should not repeat
        assert not any(c == "symptom" and v == "headache" for c, v in new4)

        # Final state
        assert lf.finding_count >= 4
        assert lf.qualifier_count >= 1
        assert buf.processed_count == 4
        assert buf.get_unprocessed_segments() == []

    def test_findings_by_category(self):
        buf = StreamingBuffer()
        lf, _ = _make_live(buf)

        buf.append_segment(_seg(
            "headache and nausea.",
            seg_id="seg_0001", t0=0.0, t1=2.0,
        ))
        lf.process_new_segments()

        symptoms = lf.get_findings_by_category("symptom")
        assert "headache" in symptoms
        assert "nausea" in symptoms

    def test_batch_of_unprocessed(self):
        """Multiple segments appended before processing — all handled."""
        buf = StreamingBuffer()
        lf, _ = _make_live(buf)

        buf.append_segment(_seg("headache.", seg_id="seg_0001", t0=0.0, t1=1.0))
        buf.append_segment(_seg("nausea.", seg_id="seg_0002", t0=1.0, t1=2.0))
        buf.append_segment(_seg("dizziness.", seg_id="seg_0003", t0=2.0, t1=3.0))

        new = lf.process_new_segments()
        symptoms = [v for c, v in new if c == "symptom"]
        assert "headache" in symptoms
        assert "nausea" in symptoms
        assert "dizziness" in symptoms
        assert buf.processed_count == 3

    def test_qualifier_accumulates_across_segments(self):
        buf = StreamingBuffer()
        lf, _ = _make_live(buf)

        buf.append_segment(_seg("severe headache.", seg_id="seg_0001", t0=0.0, t1=1.0))
        lf.process_new_segments()

        buf.append_segment(_seg("mild nausea.", seg_id="seg_0002", t0=1.0, t1=2.0))
        lf.process_new_segments()

        all_q = lf.all_qualifiers
        types = [t for t, _ in all_q]
        assert "severity" in types
        assert len([t for t in types if t == "severity"]) == 2  # severe + mild
