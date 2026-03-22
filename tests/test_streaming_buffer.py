"""Tests for streaming buffer and live preview modules."""

from __future__ import annotations

import io
import threading

import pytest

from app.streaming_buffer import StreamingBuffer, _validate_segment
from app.live_preview import (
    LivePreview,
    format_preview_line,
    format_status_line,
    format_timestamp,
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


# ══════════════════════════════════════════════════════════════════
# StreamingBuffer tests
# ══════════════════════════════════════════════════════════════════


# ── append behavior ─────────────────────────────────────────────────


class TestAppendBehavior:
    def test_append_single(self):
        buf = StreamingBuffer()
        assert buf.append_segment(_seg("hello.")) is True
        assert buf.get_segment_count() == 1

    def test_append_multiple(self):
        buf = StreamingBuffer()
        buf.append_segment(_seg("one.", seg_id="seg_0001", t0=0.0, t1=1.0))
        buf.append_segment(_seg("two.", seg_id="seg_0002", t0=1.0, t1=2.0))
        buf.append_segment(_seg("three.", seg_id="seg_0003", t0=2.0, t1=3.0))
        assert buf.get_segment_count() == 3

    def test_append_returns_segment_data(self):
        buf = StreamingBuffer()
        buf.append_segment(_seg("hello.", seg_id="seg_0001"))
        segs = buf.get_full_transcript()
        assert segs[0]["normalized_text"] == "hello."
        assert segs[0]["seg_id"] == "seg_0001"

    def test_append_preserves_extra_keys(self):
        seg = _seg("hello.")
        seg["paragraph_id"] = "para_0001"
        seg["raw_text"] = "helo."
        buf = StreamingBuffer()
        buf.append_segment(seg)
        result = buf.get_full_transcript()[0]
        assert result["paragraph_id"] == "para_0001"
        assert result["raw_text"] == "helo."


# ── segment ordering ───────────────────────────────────────────────


class TestSegmentOrdering:
    def test_in_order_append(self):
        buf = StreamingBuffer()
        buf.append_segment(_seg("a.", seg_id="seg_0001", t0=0.0, t1=1.0))
        buf.append_segment(_seg("b.", seg_id="seg_0002", t0=1.0, t1=2.0))
        buf.append_segment(_seg("c.", seg_id="seg_0003", t0=2.0, t1=3.0))
        texts = [s["normalized_text"] for s in buf.get_full_transcript()]
        assert texts == ["a.", "b.", "c."]

    def test_out_of_order_insertion(self):
        buf = StreamingBuffer()
        buf.append_segment(_seg("b.", seg_id="seg_0002", t0=1.0, t1=2.0))
        buf.append_segment(_seg("a.", seg_id="seg_0001", t0=0.0, t1=1.0))
        buf.append_segment(_seg("c.", seg_id="seg_0003", t0=2.0, t1=3.0))
        texts = [s["normalized_text"] for s in buf.get_full_transcript()]
        assert texts == ["a.", "b.", "c."]

    def test_same_t0_preserves_order(self):
        buf = StreamingBuffer()
        buf.append_segment(_seg("a.", seg_id="seg_0001", t0=1.0, t1=2.0))
        buf.append_segment(_seg("b.", seg_id="seg_0002", t0=1.0, t1=3.0))
        segs = buf.get_full_transcript()
        assert len(segs) == 2
        # Both at t0=1.0 — insertion order preserved by bisect_right
        assert segs[0]["seg_id"] == "seg_0001"
        assert segs[1]["seg_id"] == "seg_0002"

    def test_ordering_after_mixed_inserts(self):
        buf = StreamingBuffer()
        buf.append_segment(_seg("c.", seg_id="seg_0003", t0=5.0, t1=6.0))
        buf.append_segment(_seg("a.", seg_id="seg_0001", t0=1.0, t1=2.0))
        buf.append_segment(_seg("d.", seg_id="seg_0004", t0=7.0, t1=8.0))
        buf.append_segment(_seg("b.", seg_id="seg_0002", t0=3.0, t1=4.0))
        t0s = [s["t0"] for s in buf.get_full_transcript()]
        assert t0s == [1.0, 3.0, 5.0, 7.0]


# ── duplicate prevention ───────────────────────────────────────────


class TestDuplicatePrevention:
    def test_duplicate_seg_id_rejected(self):
        buf = StreamingBuffer()
        assert buf.append_segment(_seg("first.", seg_id="seg_0001")) is True
        assert buf.append_segment(_seg("second.", seg_id="seg_0001")) is False
        assert buf.get_segment_count() == 1

    def test_duplicate_does_not_modify_buffer(self):
        buf = StreamingBuffer()
        buf.append_segment(_seg("first.", seg_id="seg_0001"))
        buf.append_segment(_seg("different text.", seg_id="seg_0001"))
        assert buf.get_full_transcript()[0]["normalized_text"] == "first."

    def test_different_seg_ids_accepted(self):
        buf = StreamingBuffer()
        buf.append_segment(_seg("a.", seg_id="seg_0001"))
        buf.append_segment(_seg("b.", seg_id="seg_0002"))
        assert buf.get_segment_count() == 2


# ── recent segment retrieval ──────────────────────────────────────


class TestRecentSegments:
    def test_get_recent_all(self):
        buf = StreamingBuffer()
        buf.append_segment(_seg("a.", seg_id="seg_0001", t0=0.0, t1=1.0))
        buf.append_segment(_seg("b.", seg_id="seg_0002", t0=1.0, t1=2.0))
        recent = buf.get_recent_segments(10)
        assert len(recent) == 2

    def test_get_recent_subset(self):
        buf = StreamingBuffer()
        for i in range(5):
            buf.append_segment(_seg(
                f"seg {i}.", seg_id=f"seg_{i:04d}", t0=float(i), t1=float(i + 1),
            ))
        recent = buf.get_recent_segments(2)
        assert len(recent) == 2
        assert recent[0]["seg_id"] == "seg_0003"
        assert recent[1]["seg_id"] == "seg_0004"

    def test_get_recent_zero(self):
        buf = StreamingBuffer()
        buf.append_segment(_seg("a."))
        assert buf.get_recent_segments(0) == []

    def test_get_recent_negative(self):
        buf = StreamingBuffer()
        buf.append_segment(_seg("a."))
        assert buf.get_recent_segments(-1) == []

    def test_get_recent_empty_buffer(self):
        buf = StreamingBuffer()
        assert buf.get_recent_segments(5) == []

    def test_get_recent_returns_copy(self):
        buf = StreamingBuffer()
        buf.append_segment(_seg("a."))
        recent = buf.get_recent_segments(1)
        recent.append({"fake": True})
        assert buf.get_segment_count() == 1


# ── transcript reconstruction ─────────────────────────────────────


class TestTranscriptReconstruction:
    def test_full_transcript_returns_all(self):
        buf = StreamingBuffer()
        buf.append_segment(_seg("a.", seg_id="seg_0001", t0=0.0, t1=1.0))
        buf.append_segment(_seg("b.", seg_id="seg_0002", t0=1.0, t1=2.0))
        transcript = buf.get_full_transcript()
        assert len(transcript) == 2

    def test_full_text_concatenation(self):
        buf = StreamingBuffer()
        buf.append_segment(_seg("hello", seg_id="seg_0001", t0=0.0, t1=1.0))
        buf.append_segment(_seg("world", seg_id="seg_0002", t0=1.0, t1=2.0))
        assert buf.get_full_text() == "hello world"

    def test_full_text_empty(self):
        buf = StreamingBuffer()
        assert buf.get_full_text() == ""

    def test_full_transcript_returns_copy(self):
        buf = StreamingBuffer()
        buf.append_segment(_seg("a."))
        transcript = buf.get_full_transcript()
        transcript.clear()
        assert buf.get_segment_count() == 1

    def test_word_count(self):
        buf = StreamingBuffer()
        buf.append_segment(_seg("hello world", seg_id="seg_0001", t0=0.0, t1=1.0))
        buf.append_segment(_seg("foo bar baz", seg_id="seg_0002", t0=1.0, t1=2.0))
        assert buf.get_word_count() == 5

    def test_word_count_empty(self):
        buf = StreamingBuffer()
        assert buf.get_word_count() == 0

    def test_elapsed_seconds(self):
        buf = StreamingBuffer()
        buf.append_segment(_seg("a.", seg_id="seg_0001", t0=1.0, t1=3.0))
        buf.append_segment(_seg("b.", seg_id="seg_0002", t0=5.0, t1=8.0))
        assert buf.get_elapsed_seconds() == 7.0  # 8.0 - 1.0

    def test_elapsed_seconds_empty(self):
        buf = StreamingBuffer()
        assert buf.get_elapsed_seconds() == 0.0


# ── incremental processing ────────────────────────────────────────


class TestIncrementalProcessing:
    def test_initial_unprocessed(self):
        buf = StreamingBuffer()
        buf.append_segment(_seg("a.", seg_id="seg_0001"))
        buf.append_segment(_seg("b.", seg_id="seg_0002"))
        unprocessed = buf.get_unprocessed_segments()
        assert len(unprocessed) == 2

    def test_mark_all_processed(self):
        buf = StreamingBuffer()
        buf.append_segment(_seg("a.", seg_id="seg_0001"))
        buf.append_segment(_seg("b.", seg_id="seg_0002"))
        buf.mark_processed()
        assert buf.get_unprocessed_segments() == []
        assert buf.processed_count == 2

    def test_mark_partial_processed(self):
        buf = StreamingBuffer()
        buf.append_segment(_seg("a.", seg_id="seg_0001"))
        buf.append_segment(_seg("b.", seg_id="seg_0002"))
        buf.append_segment(_seg("c.", seg_id="seg_0003"))
        buf.mark_processed(2)
        unprocessed = buf.get_unprocessed_segments()
        assert len(unprocessed) == 1
        assert unprocessed[0]["seg_id"] == "seg_0003"

    def test_incremental_append_and_process(self):
        buf = StreamingBuffer()
        buf.append_segment(_seg("a.", seg_id="seg_0001", t0=0.0, t1=1.0))
        buf.mark_processed()
        buf.append_segment(_seg("b.", seg_id="seg_0002", t0=1.0, t1=2.0))
        unprocessed = buf.get_unprocessed_segments()
        assert len(unprocessed) == 1
        assert unprocessed[0]["seg_id"] == "seg_0002"

    def test_mark_processed_exceeds_count(self):
        buf = StreamingBuffer()
        buf.append_segment(_seg("a.", seg_id="seg_0001"))
        buf.mark_processed(100)
        assert buf.processed_count == 1

    def test_processed_count_property(self):
        buf = StreamingBuffer()
        assert buf.processed_count == 0
        buf.append_segment(_seg("a.", seg_id="seg_0001"))
        buf.mark_processed(1)
        assert buf.processed_count == 1


# ── clear ──────────────────────────────────────────────────────────


class TestClear:
    def test_clear_empties_buffer(self):
        buf = StreamingBuffer()
        buf.append_segment(_seg("a.", seg_id="seg_0001"))
        buf.append_segment(_seg("b.", seg_id="seg_0002"))
        buf.clear()
        assert buf.get_segment_count() == 0
        assert buf.get_full_transcript() == []

    def test_clear_resets_processed_count(self):
        buf = StreamingBuffer()
        buf.append_segment(_seg("a.", seg_id="seg_0001"))
        buf.mark_processed()
        buf.clear()
        assert buf.processed_count == 0

    def test_clear_allows_re_adding_same_id(self):
        buf = StreamingBuffer()
        buf.append_segment(_seg("a.", seg_id="seg_0001"))
        buf.clear()
        assert buf.append_segment(_seg("b.", seg_id="seg_0001")) is True


# ── validation ─────────────────────────────────────────────────────


class TestValidation:
    def test_missing_seg_id(self):
        seg = {"t0": 0.0, "t1": 1.0, "speaker_id": "spk_0",
               "normalized_text": "hi."}
        with pytest.raises(ValueError, match="seg_id"):
            _validate_segment(seg)

    def test_missing_t0(self):
        seg = {"seg_id": "seg_0001", "t1": 1.0, "speaker_id": "spk_0",
               "normalized_text": "hi."}
        with pytest.raises(ValueError, match="t0"):
            _validate_segment(seg)

    def test_missing_normalized_text(self):
        seg = {"seg_id": "seg_0001", "t0": 0.0, "t1": 1.0,
               "speaker_id": "spk_0"}
        with pytest.raises(ValueError, match="normalized_text"):
            _validate_segment(seg)

    def test_valid_segment_passes(self):
        _validate_segment(_seg("hello."))  # Should not raise

    def test_append_invalid_raises(self):
        buf = StreamingBuffer()
        with pytest.raises(ValueError):
            buf.append_segment({"t0": 0.0})


# ── thread safety ──────────────────────────────────────────────────


class TestThreadSafety:
    def test_concurrent_appends(self):
        buf = StreamingBuffer()
        errors: list[Exception] = []

        def append_batch(start: int):
            try:
                for i in range(50):
                    idx = start + i
                    buf.append_segment(_seg(
                        f"seg {idx}.", seg_id=f"seg_{idx:04d}",
                        t0=float(idx), t1=float(idx + 1),
                    ))
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=append_batch, args=(0,))
        t2 = threading.Thread(target=append_batch, args=(50,))
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors
        assert buf.get_segment_count() == 100


# ══════════════════════════════════════════════════════════════════
# LivePreview tests
# ══════════════════════════════════════════════════════════════════


class TestFormatTimestamp:
    def test_zero(self):
        assert format_timestamp(0.0) == "00:00"

    def test_seconds(self):
        assert format_timestamp(45.0) == "00:45"

    def test_minutes(self):
        assert format_timestamp(125.0) == "02:05"

    def test_negative_clamped(self):
        assert format_timestamp(-5.0) == "00:00"

    def test_large_value(self):
        assert format_timestamp(3661.0) == "61:01"


class TestFormatPreviewLine:
    def test_basic_format(self):
        seg = _seg("hello world.", seg_id="seg_0001", t0=12.5,
                    t1=14.0, speaker_id="spk_1")
        line = format_preview_line(seg)
        assert line == "[00:12] spk_1: hello world."

    def test_missing_fields(self):
        seg = {"seg_id": "x", "t0": 0.0, "t1": 1.0,
               "speaker_id": "spk_0", "normalized_text": ""}
        line = format_preview_line(seg)
        assert "[00:00] spk_0:" in line


class TestFormatStatusLine:
    def test_empty_buffer(self):
        buf = StreamingBuffer()
        line = format_status_line(buf)
        assert "0 words" in line
        assert "0 segments" in line

    def test_with_data(self):
        buf = StreamingBuffer()
        buf.append_segment(_seg("hello world", seg_id="seg_0001",
                                t0=0.0, t1=5.0))
        buf.append_segment(_seg("foo bar baz", seg_id="seg_0002",
                                t0=5.0, t1=10.0))
        line = format_status_line(buf)
        assert "5 words" in line
        assert "2 segments" in line
        assert "00:10" in line


class TestLivePreview:
    def test_print_segment(self):
        buf = StreamingBuffer()
        output = io.StringIO()
        preview = LivePreview(buf, output=output)
        seg = _seg("hello.", t0=5.0, speaker_id="spk_1")
        line = preview.print_segment(seg)
        assert "[00:05] spk_1: hello." in line
        assert "[00:05] spk_1: hello." in output.getvalue()
        assert preview.segments_printed == 1

    def test_print_new_segments(self):
        buf = StreamingBuffer()
        output = io.StringIO()
        preview = LivePreview(buf, output=output)

        buf.append_segment(_seg("a.", seg_id="seg_0001", t0=0.0, t1=1.0))
        buf.append_segment(_seg("b.", seg_id="seg_0002", t0=1.0, t1=2.0))

        lines = preview.print_new_segments()
        assert len(lines) == 2

        # Second call — no new segments
        lines2 = preview.print_new_segments()
        assert lines2 == []

    def test_print_new_after_additional_append(self):
        buf = StreamingBuffer()
        output = io.StringIO()
        preview = LivePreview(buf, output=output)

        buf.append_segment(_seg("a.", seg_id="seg_0001", t0=0.0, t1=1.0))
        preview.print_new_segments()

        buf.append_segment(_seg("b.", seg_id="seg_0002", t0=1.0, t1=2.0))
        lines = preview.print_new_segments()
        assert len(lines) == 1
        assert "b." in lines[0]

    def test_print_status(self):
        buf = StreamingBuffer()
        buf.append_segment(_seg("hello world", seg_id="seg_0001",
                                t0=0.0, t1=5.0))
        output = io.StringIO()
        preview = LivePreview(buf, output=output)
        line = preview.print_status()
        assert "2 words" in line
        assert "1 segments" in line

    def test_segments_printed_counter(self):
        buf = StreamingBuffer()
        output = io.StringIO()
        preview = LivePreview(buf, output=output)
        assert preview.segments_printed == 0
        preview.print_segment(_seg("a."))
        preview.print_segment(_seg("b."))
        assert preview.segments_printed == 2
