"""Tests for conservative ICE (Ideas, Concerns, Expectations) extraction."""

from __future__ import annotations

import pytest

from app.ice_extraction import extract_ice


def _seg(text: str, seg_id: str = "seg_0001", speaker_id: str = "spk_0",
         t0: float = 0.0, t1: float = 1.0) -> dict:
    return {
        "seg_id": seg_id,
        "t0": t0,
        "t1": t1,
        "speaker_id": speaker_id,
        "normalized_text": text,
    }


# ── empty / structure ────────────────────────────────────────────


class TestStructure:
    def test_empty_segments(self):
        result = extract_ice([])
        assert result == {"ideas": [], "concerns": [], "expectations": []}

    def test_no_ice_content(self):
        result = extract_ice([_seg("patient has headache.")])
        assert result["ideas"] == []
        assert result["concerns"] == []
        assert result["expectations"] == []

    def test_keys_present(self):
        result = extract_ice([])
        assert set(result.keys()) == {"ideas", "concerns", "expectations"}


# ── ideas ────────────────────────────────────────────────────────


class TestIdeas:
    def test_i_think_it_might_be(self):
        result = extract_ice([
            _seg("I think it might be a migraine or something."),
        ])
        assert len(result["ideas"]) == 1
        assert "migraine" in result["ideas"][0]["text"].lower()

    def test_i_wonder_if_its(self):
        result = extract_ice([
            _seg("I wonder if it's related to stress somehow."),
        ])
        assert len(result["ideas"]) == 1

    def test_could_this_be(self):
        result = extract_ice([
            _seg("could this be something serious happening here?"),
        ])
        assert len(result["ideas"]) == 1

    def test_maybe_its(self):
        result = extract_ice([
            _seg("maybe it's the new medication causing this."),
        ])
        assert len(result["ideas"]) == 1

    def test_seg_id_preserved(self):
        result = extract_ice([
            _seg("I think it might be a virus or infection.",
                 seg_id="seg_0005", speaker_id="spk_1", t0=10.0),
        ])
        item = result["ideas"][0]
        assert item["seg_id"] == "seg_0005"
        assert item["speaker_id"] == "spk_1"
        assert item["t_start"] == 10.0


# ── concerns ─────────────────────────────────────────────────────


class TestConcerns:
    def test_im_worried_about(self):
        result = extract_ice([
            _seg("I'm worried about it being something serious."),
        ])
        assert len(result["concerns"]) == 1

    def test_im_afraid_it_might_be(self):
        result = extract_ice([
            _seg("I'm afraid it might be cancer or something bad."),
        ])
        assert len(result["concerns"]) == 1

    def test_im_concerned_about(self):
        result = extract_ice([
            _seg("I'm concerned about the spreading of the rash."),
        ])
        assert len(result["concerns"]) == 1

    def test_my_concern_is(self):
        result = extract_ice([
            _seg("my concern is that it keeps coming back regularly."),
        ])
        assert len(result["concerns"]) == 1


# ── expectations ─────────────────────────────────────────────────


class TestExpectations:
    def test_i_was_hoping_for(self):
        result = extract_ice([
            _seg("I was hoping for a referral to a specialist."),
        ])
        assert len(result["expectations"]) == 1

    def test_id_like_to_get(self):
        result = extract_ice([
            _seg("I'd like to get an MRI scan done."),
        ])
        assert len(result["expectations"]) == 1

    def test_could_i_get(self):
        result = extract_ice([
            _seg("could I get a blood test done please."),
        ])
        assert len(result["expectations"]) == 1


# ── no false positives ───────────────────────────────────────────


class TestNoFalsePositives:
    def test_bare_i_think_no_match(self):
        """Bare 'I think' without clinical continuation should not match."""
        result = extract_ice([_seg("I think so.")])
        assert result["ideas"] == []

    def test_i_want_to_go_home(self):
        """General desire should not be extracted as expectation."""
        result = extract_ice([_seg("I want to go home now.")])
        assert result["expectations"] == []

    def test_can_you_pass_the_water(self):
        """Casual request should not match."""
        result = extract_ice([_seg("can you pass the water please.")])
        assert result["expectations"] == []

    def test_bare_trigger_insufficient_content(self):
        """Trigger with fewer than 2 words of content should not match."""
        result = extract_ice([_seg("I'm worried about it.")])
        # "it" is only 1 word — should not match
        assert result["concerns"] == []


# ── deduplication ────────────────────────────────────────────────


class TestDeduplication:
    def test_duplicate_text_deduplicated(self):
        result = extract_ice([
            _seg("I'm worried about the headache coming back.",
                 seg_id="seg_0001"),
            _seg("I'm worried about the headache coming back.",
                 seg_id="seg_0002"),
        ])
        assert len(result["concerns"]) == 1
