"""Tests for history extraction module."""

from __future__ import annotations

import pytest

from app.history_extraction import extract_history_context, _empty_structure
from app.clinical_state import build_clinical_state


def _seg(text: str) -> dict:
    return {
        "seg_id": "seg_0001",
        "t0": 0.0,
        "t1": 1.0,
        "speaker_id": "spk_0",
        "normalized_text": text,
    }


# ── structure tests ──────────────────────────────────────────────────


class TestStructure:
    def test_empty_segments(self):
        result = extract_history_context([])
        assert result == _empty_structure()

    def test_all_top_level_keys(self):
        result = extract_history_context([_seg("hello.")])
        expected_keys = {
            "conditions", "procedures", "medications", "allergies",
            "immunizations", "family_history", "social_history",
            "exposures", "other_history",
        }
        assert set(result.keys()) == expected_keys

    def test_social_history_subkeys(self):
        result = extract_history_context([_seg("hello.")])
        sh = result["social_history"]
        expected = {
            "substance_use", "occupation", "living_situation",
            "support_network", "sexual_history", "other_social_history",
        }
        assert set(sh.keys()) == expected

    def test_substance_use_subkeys(self):
        result = extract_history_context([_seg("hello.")])
        su = result["social_history"]["substance_use"]
        expected = {
            "tobacco", "alcohol", "recreational_drugs",
            "non_prescription_substances", "other_substance_use",
        }
        assert set(su.keys()) == expected

    def test_exposure_subkeys(self):
        result = extract_history_context([_seg("hello.")])
        ex = result["exposures"]
        expected = {
            "travel", "animals", "infectious_contacts", "environmental",
            "food_and_water", "insects_and_ticks", "occupational_exposures",
            "other_exposures",
        }
        assert set(ex.keys()) == expected


# ── conditions ────────────────────────────────────────────────────────


class TestConditions:
    def test_history_of_diabetes(self):
        result = extract_history_context([_seg("history of diabetes.")])
        assert any("diabetes" in c.lower() for c in result["conditions"])

    def test_known_hypertension(self):
        result = extract_history_context([_seg("known hypertension.")])
        assert any("hypertension" in c.lower() for c in result["conditions"])

    def test_keyword_match(self):
        result = extract_history_context([_seg("patient has asthma.")])
        assert any("asthma" in c.lower() for c in result["conditions"])


# ── allergies ─────────────────────────────────────────────────────────


class TestAllergies:
    def test_allergic_to(self):
        result = extract_history_context([_seg("allergic to penicillin.")])
        assert any("penicillin" in a.lower() for a in result["allergies"])

    def test_allergy_to(self):
        result = extract_history_context([_seg("allergy to sulfa.")])
        assert any("sulfa" in a.lower() for a in result["allergies"])


# ── substance use: tobacco ───────────────────────────────────────────


class TestTobacco:
    def test_smokes(self):
        result = extract_history_context([_seg("patient smokes 10 cigarettes a day.")])
        tobacco = result["social_history"]["substance_use"]["tobacco"]
        assert len(tobacco) >= 1

    def test_former_smoker(self):
        result = extract_history_context([_seg("former smoker.")])
        tobacco = result["social_history"]["substance_use"]["tobacco"]
        assert any("former smoker" in t.lower() for t in tobacco)

    def test_never_smoked(self):
        result = extract_history_context([_seg("never smoked.")])
        tobacco = result["social_history"]["substance_use"]["tobacco"]
        assert any("never smoked" in t.lower() for t in tobacco)

    def test_vaping(self):
        result = extract_history_context([_seg("patient is vaping.")])
        tobacco = result["social_history"]["substance_use"]["tobacco"]
        assert len(tobacco) >= 1


# ── substance use: alcohol ───────────────────────────────────────────


class TestAlcohol:
    def test_drinks_per_week(self):
        result = extract_history_context([_seg("drinks 5 drinks per week.")])
        alcohol = result["social_history"]["substance_use"]["alcohol"]
        assert len(alcohol) >= 1

    def test_social_drinker(self):
        result = extract_history_context([_seg("social drinker.")])
        alcohol = result["social_history"]["substance_use"]["alcohol"]
        assert any("social drinker" in a.lower() for a in alcohol)


# ── substance use: supplements ───────────────────────────────────────


class TestSupplements:
    def test_vitamin(self):
        result = extract_history_context([_seg("takes vitamin D supplement.")])
        nps = result["social_history"]["substance_use"]["non_prescription_substances"]
        assert len(nps) >= 1

    def test_herbal(self):
        result = extract_history_context([_seg("uses herbal remedies.")])
        nps = result["social_history"]["substance_use"]["non_prescription_substances"]
        assert any("herbal" in s.lower() for s in nps)

    def test_fish_oil(self):
        result = extract_history_context([_seg("takes fish oil daily.")])
        nps = result["social_history"]["substance_use"]["non_prescription_substances"]
        assert any("fish oil" in s.lower() for s in nps)


# ── substance use: recreational drugs ────────────────────────────────


class TestRecreationalDrugs:
    def test_cannabis(self):
        result = extract_history_context([_seg("uses cannabis occasionally.")])
        drugs = result["social_history"]["substance_use"]["recreational_drugs"]
        assert any("cannabis" in d.lower() for d in drugs)

    def test_cocaine(self):
        result = extract_history_context([_seg("history of cocaine use.")])
        drugs = result["social_history"]["substance_use"]["recreational_drugs"]
        assert any("cocaine" in d.lower() for d in drugs)


# ── exposures: travel ────────────────────────────────────────────────


class TestTravelExposure:
    def test_travelled_to(self):
        result = extract_history_context([_seg("travelled to India last month.")])
        travel = result["exposures"]["travel"]
        assert any("india" in t.lower() for t in travel)

    def test_returned_from(self):
        result = extract_history_context([_seg("returned from Thailand.")])
        travel = result["exposures"]["travel"]
        assert any("thailand" in t.lower() for t in travel)

    def test_recent_travel(self):
        result = extract_history_context([_seg("recent travel.")])
        travel = result["exposures"]["travel"]
        assert any("recent travel" in t.lower() for t in travel)


# ── exposures: animals ───────────────────────────────────────────────


class TestAnimalExposure:
    def test_pet_cat(self):
        result = extract_history_context([_seg("has a pet cat at home.")])
        animals = result["exposures"]["animals"]
        assert len(animals) >= 1

    def test_dog(self):
        result = extract_history_context([_seg("owns a dog.")])
        animals = result["exposures"]["animals"]
        assert any("dog" in a.lower() for a in animals)

    def test_animal_bite(self):
        result = extract_history_context([_seg("animal bite on left hand.")])
        animals = result["exposures"]["animals"]
        assert any("animal bite" in a.lower() for a in animals)


# ── exposures: occupational ──────────────────────────────────────────


class TestOccupationalExposure:
    def test_occupational_exposure(self):
        result = extract_history_context([_seg("occupational exposure to asbestos.")])
        occ = result["exposures"]["occupational_exposures"]
        assert any("occupational exposure" in o.lower() for o in occ)

    def test_healthcare_worker(self):
        result = extract_history_context([_seg("patient is a healthcare worker.")])
        occ = result["exposures"]["occupational_exposures"]
        assert any("healthcare worker" in o.lower() for o in occ)


# ── exposures: insects and ticks ─────────────────────────────────────


class TestInsectExposure:
    def test_tick_bite(self):
        result = extract_history_context([_seg("tick bite two weeks ago.")])
        insects = result["exposures"]["insects_and_ticks"]
        assert any("tick bite" in i.lower() for i in insects)

    def test_mosquito_bite(self):
        result = extract_history_context([_seg("mosquito bite on arm.")])
        insects = result["exposures"]["insects_and_ticks"]
        assert any("mosquito bite" in i.lower() for i in insects)


# ── exposures: environmental ─────────────────────────────────────────


class TestEnvironmentalExposure:
    def test_mold(self):
        result = extract_history_context([_seg("mold in the bathroom.")])
        env = result["exposures"]["environmental"]
        assert any("mold" in e.lower() for e in env)

    def test_dust(self):
        result = extract_history_context([_seg("exposure to dust at work.")])
        env = result["exposures"]["environmental"]
        assert any("dust" in e.lower() for e in env)


# ── exposures: food and water ────────────────────────────────────────


class TestFoodWaterExposure:
    def test_contaminated_food(self):
        result = extract_history_context([_seg("ate contaminated food.")])
        fw = result["exposures"]["food_and_water"]
        assert any("contaminated food" in f.lower() for f in fw)

    def test_raw_fish(self):
        result = extract_history_context([_seg("ate raw fish yesterday.")])
        fw = result["exposures"]["food_and_water"]
        assert any("raw fish" in f.lower() for f in fw)


# ── exposures: infectious contacts ───────────────────────────────────


class TestInfectiousContacts:
    def test_sick_contact(self):
        result = extract_history_context([_seg("had a sick contact last week.")])
        ic = result["exposures"]["infectious_contacts"]
        assert any("sick contact" in c.lower() for c in ic)


# ── fallback fields ──────────────────────────────────────────────────


class TestFallbackFields:
    def test_other_history_is_list(self):
        result = extract_history_context([_seg("hello.")])
        assert isinstance(result["other_history"], list)

    def test_other_social_history_is_list(self):
        result = extract_history_context([_seg("hello.")])
        assert isinstance(result["social_history"]["other_social_history"], list)

    def test_other_exposures_is_list(self):
        result = extract_history_context([_seg("hello.")])
        assert isinstance(result["exposures"]["other_exposures"], list)

    def test_other_substance_use_is_list(self):
        result = extract_history_context([_seg("hello.")])
        su = result["social_history"]["substance_use"]
        assert isinstance(su["other_substance_use"], list)


# ── family history ───────────────────────────────────────────────────


class TestFamilyHistory:
    def test_family_history_of(self):
        result = extract_history_context([
            _seg("family history of breast cancer."),
        ])
        assert any("breast cancer" in f.lower() for f in result["family_history"])

    def test_mother_had(self):
        result = extract_history_context([_seg("mother had diabetes.")])
        assert any("diabetes" in f.lower() for f in result["family_history"])


# ── procedures ───────────────────────────────────────────────────────


class TestProcedures:
    def test_previous_surgery(self):
        result = extract_history_context([_seg("previous surgery on knee.")])
        assert any("surgery" in p.lower() for p in result["procedures"])

    def test_appendectomy(self):
        result = extract_history_context([_seg("had a appendectomy.")])
        assert any("appendectomy" in p.lower() for p in result["procedures"])


# ── immunizations ────────────────────────────────────────────────────


class TestImmunizations:
    def test_flu_shot(self):
        result = extract_history_context([_seg("received flu shot.")])
        assert len(result["immunizations"]) >= 1

    def test_covid_vaccine(self):
        result = extract_history_context([_seg("got covid vaccine.")])
        assert any("covid" in i.lower() for i in result["immunizations"])


# ── occupation ───────────────────────────────────────────────────────


class TestOccupation:
    def test_works_as(self):
        result = extract_history_context([_seg("works as a carpenter.")])
        occ = result["social_history"]["occupation"]
        assert any("carpenter" in o.lower() for o in occ)

    def test_retired(self):
        result = extract_history_context([_seg("patient is retired.")])
        occ = result["social_history"]["occupation"]
        assert any("retired" in o.lower() for o in occ)


# ── living situation ─────────────────────────────────────────────────


class TestLivingSituation:
    def test_lives_alone(self):
        result = extract_history_context([_seg("lives alone.")])
        ls = result["social_history"]["living_situation"]
        assert any("lives alone" in l.lower() for l in ls)

    def test_nursing_home(self):
        result = extract_history_context([_seg("resident of nursing home.")])
        ls = result["social_history"]["living_situation"]
        assert any("nursing home" in l.lower() for l in ls)


# ── clinical state integration ───────────────────────────────────────


class TestClinicalStateIntegration:
    def test_history_in_clinical_state(self):
        state = build_clinical_state([_seg("history of diabetes. smokes.")])
        assert "history" in state
        assert isinstance(state["history"], dict)
        assert any("diabetes" in c.lower() for c in state["history"]["conditions"])
        tobacco = state["history"]["social_history"]["substance_use"]["tobacco"]
        assert len(tobacco) >= 1

    def test_history_empty_segments(self):
        state = build_clinical_state([])
        assert "history" in state
        assert state["history"]["conditions"] == []

    def test_history_does_not_break_other_keys(self):
        state = build_clinical_state([_seg("headache for 3 days.")])
        assert "symptoms" in state
        assert "history" in state
        assert "headache" in state["symptoms"]
