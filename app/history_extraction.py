"""History extraction — deterministic PMH, social history, and exposure detection.

Extracts background clinical context from transcript segments using
keyword and regex matching.  No ML, no LLM, no external APIs.

Items that cannot be confidently classified are placed in fallback
fields (``other_social_history``, ``other_exposures``, ``other_history``)
so that no extracted information is discarded.
"""

from __future__ import annotations

import re


# ── pattern tables ───────────────────────────────────────────────────
# Each entry: (compiled regex, capture-group index for the extracted phrase).
# Group 0 = full match when no sub-group is needed.

def _kw(pattern: str) -> re.Pattern[str]:
    """Compile a case-insensitive word-boundary pattern."""
    return re.compile(pattern, re.IGNORECASE)


# -- conditions (PMH) -------------------------------------------------

_CONDITION_PATTERNS: list[tuple[re.Pattern[str], int]] = [
    (_kw(r"\b(?:history of|known|diagnosed with|has|had)\s+([a-z][a-z ]{2,30})"), 1),
    (_kw(r"\b(diabetes|hypertension|asthma|copd|epilepsy|heart failure"
         r"|coronary artery disease|atrial fibrillation|hypothyroidism"
         r"|hyperthyroidism|chronic kidney disease|stroke|cancer"
         r"|depression|anxiety disorder|rheumatoid arthritis"
         r"|osteoarthritis|gout|migraine|hepatitis)\b"), 1),
]

# -- procedures --------------------------------------------------------

_PROCEDURE_PATTERNS: list[tuple[re.Pattern[str], int]] = [
    (_kw(r"\b(?:previous|prior|had a|underwent|history of)\s+"
         r"(surgery|appendectomy|cholecystectomy|hysterectomy"
         r"|knee replacement|hip replacement|bypass|stent"
         r"|cesarean|c-section|tonsillectomy|colonoscopy"
         r"|endoscopy|biopsy|catheterization)\b"), 1),
]

# -- allergies ---------------------------------------------------------

_ALLERGY_PATTERNS: list[tuple[re.Pattern[str], int]] = [
    (_kw(r"\ballergic to\s+([a-z][a-z ]{1,30})"), 1),
    (_kw(r"\b(?:allergy|allergies) to\s+([a-z][a-z ]{1,30})"), 1),
    (_kw(r"\b(penicillin allergy|sulfa allergy|latex allergy"
         r"|peanut allergy|shellfish allergy)\b"), 1),
]

# -- immunizations -----------------------------------------------------

_IMMUNIZATION_PATTERNS: list[tuple[re.Pattern[str], int]] = [
    (_kw(r"\b(?:vaccinated|immunized|received)\s+(?:against |for |with )?"
         r"(covid|influenza|flu|tetanus|hepatitis [ab]|pneumonia"
         r"|mmr|hpv|shingles|measles|polio)\b"), 1),
    (_kw(r"\b(flu shot|covid vaccine|booster|vaccination)\b"), 0),
]

# -- family history ----------------------------------------------------

_FAMILY_HISTORY_PATTERNS: list[tuple[re.Pattern[str], int]] = [
    (_kw(r"\b(?:family history of|mother had|father had|brother had"
         r"|sister had|parent had|grandparent had|grandmother had"
         r"|grandfather had)\s+([a-z][a-z ]{2,30})"), 1),
    (_kw(r"\b(?:runs in the family|familial)\b"), 0),
]

# -- substance use: tobacco --------------------------------------------

_TOBACCO_PATTERNS: list[tuple[re.Pattern[str], int]] = [
    (_kw(r"\b(smok(?:es?|ing|ed)|cigarettes?|tobacco|nicotine"
         r"|pack[- ]?(?:years?|a day)|vap(?:es?|ing)|e-cigarette)\b"), 0),
    (_kw(r"\b(quit(?:ted)? smoking|former smoker|never smoked"
         r"|non-?smoker|current smoker)\b"), 0),
]

# -- substance use: alcohol --------------------------------------------

_ALCOHOL_PATTERNS: list[tuple[re.Pattern[str], int]] = [
    (_kw(r"\b(alcohol|drinks?(?:\s+per\s+(?:day|week))?|beer|wine"
         r"|spirits|units?\s+(?:per|a)\s+week|binge drinking"
         r"|social drinker|heavy drinker|teetotal|abstinent)\b"), 0),
]

# -- substance use: recreational drugs ---------------------------------

_RECREATIONAL_DRUG_PATTERNS: list[tuple[re.Pattern[str], int]] = [
    (_kw(r"\b(cannabis|marijuana|cocaine|amphetamine|methamphetamine"
         r"|heroin|opioid abuse|fentanyl|ecstasy|mdma|lsd"
         r"|recreational drug|illicit drug|intravenous drug)\b"), 0),
]

# -- substance use: non-prescription substances -----------------------

_NON_PRESCRIPTION_PATTERNS: list[tuple[re.Pattern[str], int]] = [
    (_kw(r"\b(supplement|herbal|over-the-counter|otc|vitamin"
         r"|fish oil|probiotics|melatonin|st\.?\s*john'?s?\s*wort"
         r"|ginkgo|echinacea|turmeric|glucosamine)\b"), 0),
]

# -- occupation --------------------------------------------------------

_OCCUPATION_PATTERNS: list[tuple[re.Pattern[str], int]] = [
    (_kw(r"\b(?:works? (?:as|in|at)|occupation|employed as|job is)\s+"
         r"([a-z][a-z ]{2,30})"), 1),
    (_kw(r"\b(retired|unemployed|self-employed|on disability"
         r"|student|homemaker)\b"), 0),
]

# -- living situation --------------------------------------------------

_LIVING_PATTERNS: list[tuple[re.Pattern[str], int]] = [
    (_kw(r"\b(lives? (?:alone|with|at home)|homeless|shelter"
         r"|nursing home|assisted living|care home"
         r"|group home|independent(?:ly)?)\b"), 0),
]

# -- support network ---------------------------------------------------

_SUPPORT_PATTERNS: list[tuple[re.Pattern[str], int]] = [
    (_kw(r"\b((?:good |strong |no |limited )?(?:social )?support"
         r"|caregiver|carer|next of kin|emergency contact)\b"), 0),
]

# -- sexual history ----------------------------------------------------

_SEXUAL_HISTORY_PATTERNS: list[tuple[re.Pattern[str], int]] = [
    (_kw(r"\b(sexually active|sexual partner|std|sti"
         r"|contraception|condom|birth control|pregnant"
         r"|pregnancy|last menstrual period|lmp)\b"), 0),
]

# -- exposures: travel -------------------------------------------------

_TRAVEL_PATTERNS: list[tuple[re.Pattern[str], int]] = [
    (_kw(r"\b(?:travel(?:led|ed|ing|s)? to|returned from|visited|trip to"
         r"|been to|flew (?:to|from))\s+([a-z][a-z ]{2,30})"), 1),
    (_kw(r"\b(recent travel|travel history|abroad)\b"), 0),
]

# -- exposures: animals ------------------------------------------------

_ANIMAL_PATTERNS: list[tuple[re.Pattern[str], int]] = [
    (_kw(r"\b(pet|pets|dog|cat|bird|parrot|reptile|rodent|hamster"
         r"|rabbit|farm animal|livestock|horse|cattle"
         r"|animal (?:bite|scratch|contact|exposure))\b"), 0),
]

# -- exposures: infectious contacts -----------------------------------

_INFECTIOUS_CONTACT_PATTERNS: list[tuple[re.Pattern[str], int]] = [
    (_kw(r"\b(sick contact|exposed to|contact with (?:someone|a person)"
         r"|household contact|close contact|outbreak|epidemic)\b"), 0),
]

# -- exposures: environmental -----------------------------------------

_ENVIRONMENTAL_PATTERNS: list[tuple[re.Pattern[str], int]] = [
    (_kw(r"\b(mold|mould|dust|asbestos|lead|chemical|fumes"
         r"|pollution|radiation|radon|pesticide|herbicide"
         r"|toxic|hazardous)\b"), 0),
]

# -- exposures: food and water ----------------------------------------

_FOOD_WATER_PATTERNS: list[tuple[re.Pattern[str], int]] = [
    (_kw(r"\b(contaminated (?:food|water)|food poisoning|unsafe water"
         r"|raw (?:meat|fish|shellfish|eggs)|unpasteurized"
         r"|street food|tap water)\b"), 0),
]

# -- exposures: insects and ticks --------------------------------------

_INSECT_TICK_PATTERNS: list[tuple[re.Pattern[str], int]] = [
    (_kw(r"\b(tick (?:bite|exposure)|mosquito (?:bite|exposure)"
         r"|insect bite|flea bite|bed bug|lice|scabies)\b"), 0),
]

# -- exposures: occupational ------------------------------------------

_OCCUPATIONAL_EXPOSURE_PATTERNS: list[tuple[re.Pattern[str], int]] = [
    (_kw(r"\b(occupational exposure|workplace exposure|work-related"
         r"|industrial exposure|mining|construction work"
         r"|healthcare worker|lab(?:oratory)? worker)\b"), 0),
]

# -- medications (history-specific, e.g. "takes metformin daily") ------

_HISTORY_MED_PATTERNS: list[tuple[re.Pattern[str], int]] = [
    (_kw(r"\b(?:takes?|taking|on|using|current(?:ly)? (?:on|taking))\s+"
         r"([a-z][a-z ]{2,30})\b"), 1),
]


# ── extraction engine ────────────────────────────────────────────────

def _extract_matches(
    text: str,
    patterns: list[tuple[re.Pattern[str], int]],
) -> list[str]:
    """Run patterns against text, return unique matches in order."""
    seen: set[str] = set()
    results: list[str] = []
    for pat, group in patterns:
        for m in pat.finditer(text):
            value = m.group(group).strip().rstrip(".,;!?")
            key = value.lower()
            if key and key not in seen:
                seen.add(key)
                results.append(value)
    return results


def _empty_structure() -> dict:
    """Return the empty history structure."""
    return {
        "conditions": [],
        "procedures": [],
        "medications": [],
        "allergies": [],
        "immunizations": [],
        "family_history": [],
        "social_history": {
            "substance_use": {
                "tobacco": [],
                "alcohol": [],
                "recreational_drugs": [],
                "non_prescription_substances": [],
                "other_substance_use": [],
            },
            "occupation": [],
            "living_situation": [],
            "support_network": [],
            "sexual_history": [],
            "other_social_history": [],
        },
        "exposures": {
            "travel": [],
            "animals": [],
            "infectious_contacts": [],
            "environmental": [],
            "food_and_water": [],
            "insects_and_ticks": [],
            "occupational_exposures": [],
            "other_exposures": [],
        },
        "other_history": [],
    }


def extract_history_context(segments: list[dict]) -> dict:
    """Extract history context from normalized segments.

    Args:
        segments: list of normalized segment dicts (with ``normalized_text``).

    Returns:
        dict following the history structure with extracted items.
    """
    full_text = " ".join(
        seg.get("normalized_text", "") for seg in segments
    ).strip()

    if not full_text:
        return _empty_structure()

    result = _empty_structure()

    result["conditions"] = _extract_matches(full_text, _CONDITION_PATTERNS)
    result["procedures"] = _extract_matches(full_text, _PROCEDURE_PATTERNS)
    result["allergies"] = _extract_matches(full_text, _ALLERGY_PATTERNS)
    result["immunizations"] = _extract_matches(full_text, _IMMUNIZATION_PATTERNS)
    result["family_history"] = _extract_matches(full_text, _FAMILY_HISTORY_PATTERNS)
    result["medications"] = _extract_matches(full_text, _HISTORY_MED_PATTERNS)

    # Social history
    sh = result["social_history"]
    su = sh["substance_use"]
    su["tobacco"] = _extract_matches(full_text, _TOBACCO_PATTERNS)
    su["alcohol"] = _extract_matches(full_text, _ALCOHOL_PATTERNS)
    su["recreational_drugs"] = _extract_matches(full_text, _RECREATIONAL_DRUG_PATTERNS)
    su["non_prescription_substances"] = _extract_matches(
        full_text, _NON_PRESCRIPTION_PATTERNS,
    )
    sh["occupation"] = _extract_matches(full_text, _OCCUPATION_PATTERNS)
    sh["living_situation"] = _extract_matches(full_text, _LIVING_PATTERNS)
    sh["support_network"] = _extract_matches(full_text, _SUPPORT_PATTERNS)
    sh["sexual_history"] = _extract_matches(full_text, _SEXUAL_HISTORY_PATTERNS)

    # Exposures
    ex = result["exposures"]
    ex["travel"] = _extract_matches(full_text, _TRAVEL_PATTERNS)
    ex["animals"] = _extract_matches(full_text, _ANIMAL_PATTERNS)
    ex["infectious_contacts"] = _extract_matches(
        full_text, _INFECTIOUS_CONTACT_PATTERNS,
    )
    ex["environmental"] = _extract_matches(full_text, _ENVIRONMENTAL_PATTERNS)
    ex["food_and_water"] = _extract_matches(full_text, _FOOD_WATER_PATTERNS)
    ex["insects_and_ticks"] = _extract_matches(full_text, _INSECT_TICK_PATTERNS)
    ex["occupational_exposures"] = _extract_matches(
        full_text, _OCCUPATIONAL_EXPOSURE_PATTERNS,
    )

    return result
