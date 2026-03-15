"""Scenario definitions for synthetic clinical case generation.

Each scenario is a plain dict with clearly separated concerns:
  - encounter_type: in_person_consultation | telephone_triage
  - participants: list of speaker dicts (role, id, voice hints)
  - theme: clinical theme tag (chest_pain, cough_fever, etc.)
  - dialogue: list of turns (speaker_id, text)
  - ground_truth: expected clinical facts extracted by SCRIBE

Scenarios are deterministic data — no randomness, no I/O.
"""

from __future__ import annotations


# ── participant templates ────────────────────────────────────────


def _doctor(speaker_id: str = "spk_0") -> dict:
    return {
        "speaker_id": speaker_id,
        "role": "clinician",
        "label": "Doctor",
        "voice_hint": "male",
    }


def _patient(speaker_id: str = "spk_1") -> dict:
    return {
        "speaker_id": speaker_id,
        "role": "patient",
        "label": "Patient",
        "voice_hint": "female",
    }


# ── scenario registry ───────────────────────────────────────────


SCENARIOS: dict[str, dict] = {}


def _register(case_id: str, scenario: dict) -> None:
    scenario["case_id"] = case_id
    SCENARIOS[case_id] = scenario


# ── chest_pain_consultation ──────────────────────────────────────

_register("chest_pain_consultation", {
    "encounter_type": "in_person_consultation",
    "theme": "chest_pain",
    "language": "en",
    "participants": [_doctor("spk_0"), _patient("spk_1")],
    "dialogue": [
        {"speaker_id": "spk_0", "text": "Good morning. What brings you in today?"},
        {"speaker_id": "spk_1", "text": "I have been having chest pain for three days now."},
        {"speaker_id": "spk_0", "text": "Can you describe the pain? Is it sharp or dull?"},
        {"speaker_id": "spk_1", "text": "It is a dull pressure in the center of my chest."},
        {"speaker_id": "spk_0", "text": "Does the pain get worse with exertion or physical activity?"},
        {"speaker_id": "spk_1", "text": "Yes it gets worse when I walk up stairs. It gets better when I rest."},
        {"speaker_id": "spk_0", "text": "Do you have any shortness of breath?"},
        {"speaker_id": "spk_1", "text": "Yes some shortness of breath especially with the chest pain."},
        {"speaker_id": "spk_0", "text": "Any nausea or sweating?"},
        {"speaker_id": "spk_1", "text": "I have had some nausea but no sweating."},
        {"speaker_id": "spk_0", "text": "Do you have any history of heart disease?"},
        {"speaker_id": "spk_1", "text": "No I do not have any heart problems. No diabetes either."},
        {"speaker_id": "spk_0", "text": "Are you taking any medications currently?"},
        {"speaker_id": "spk_1", "text": "I take ibuprofen occasionally for headaches."},
        {"speaker_id": "spk_0", "text": "I would like to run an ECG and some blood tests. We should rule out cardiac causes given your symptoms."},
    ],
    "ground_truth": {
        "symptoms": ["chest pain", "shortness of breath", "nausea"],
        "negations": ["sweating", "heart disease", "diabetes"],
        "medications": ["ibuprofen"],
        "durations": ["three days"],
        "qualifiers": [
            {
                "symptom": "chest pain",
                "severity": None,
                "onset": None,
                "character": "dull",
                "pattern": None,
                "progression": None,
                "laterality": None,
                "radiation": None,
                "aggravating_factors": ["walk up stairs"],
                "relieving_factors": ["rest"],
            },
        ],
        "speaker_roles": {
            "spk_0": "clinician",
            "spk_1": "patient",
        },
        "expected_patterns": ["angina_like"],
        "expected_red_flags": ["chest_pain_with_dyspnea"],
    },
})


# ── cough_fever_telephone ────────────────────────────────────────

_register("cough_fever_telephone", {
    "encounter_type": "telephone_triage",
    "theme": "cough_fever",
    "language": "en",
    "participants": [_doctor("spk_0"), _patient("spk_1")],
    "dialogue": [
        {"speaker_id": "spk_0", "text": "Hello this is the triage nurse. How can I help you today?"},
        {"speaker_id": "spk_1", "text": "I have had a bad cough and fever for about five days."},
        {"speaker_id": "spk_0", "text": "Can you tell me more about the cough? Is it dry or productive?"},
        {"speaker_id": "spk_1", "text": "It is a productive cough with yellow mucus."},
        {"speaker_id": "spk_0", "text": "And the fever, how high has it been?"},
        {"speaker_id": "spk_1", "text": "It has been around thirty eight point five degrees."},
        {"speaker_id": "spk_0", "text": "Do you have any difficulty breathing or chest tightness?"},
        {"speaker_id": "spk_1", "text": "No difficulty breathing. But I do have some sore throat."},
        {"speaker_id": "spk_0", "text": "Any body aches or fatigue?"},
        {"speaker_id": "spk_1", "text": "Yes I have been feeling very tired and my muscles ache."},
        {"speaker_id": "spk_0", "text": "Are you taking any medications for this?"},
        {"speaker_id": "spk_1", "text": "I have been taking paracetamol for the fever."},
        {"speaker_id": "spk_0", "text": "I recommend you come in for a physical examination. Please drink plenty of fluids and continue the paracetamol."},
    ],
    "ground_truth": {
        "symptoms": ["cough", "fever", "sore throat", "fatigue"],
        "negations": ["difficulty breathing"],
        "medications": ["paracetamol"],
        "durations": ["five days"],
        "qualifiers": [
            {
                "symptom": "cough",
                "severity": None,
                "onset": None,
                "character": "productive",
                "pattern": None,
                "progression": None,
                "laterality": None,
                "radiation": None,
                "aggravating_factors": [],
                "relieving_factors": [],
            },
        ],
        "speaker_roles": {
            "spk_0": "clinician",
            "spk_1": "patient",
        },
        "expected_patterns": [],
        "expected_red_flags": [],
    },
})


# ── abdominal_pain_consultation ──────────────────────────────────

_register("abdominal_pain_consultation", {
    "encounter_type": "in_person_consultation",
    "theme": "abdominal_pain",
    "language": "en",
    "participants": [_doctor("spk_0"), _patient("spk_1")],
    "dialogue": [
        {"speaker_id": "spk_0", "text": "Good afternoon. What is the problem today?"},
        {"speaker_id": "spk_1", "text": "I have had abdominal pain for about two weeks now."},
        {"speaker_id": "spk_0", "text": "Where exactly is the pain located?"},
        {"speaker_id": "spk_1", "text": "It is mostly in the upper right part of my abdomen."},
        {"speaker_id": "spk_0", "text": "How would you describe the pain?"},
        {"speaker_id": "spk_1", "text": "It is a cramping pain that comes and goes. Sometimes it is quite sharp."},
        {"speaker_id": "spk_0", "text": "Does it get worse after eating?"},
        {"speaker_id": "spk_1", "text": "Yes it is much worse after fatty meals."},
        {"speaker_id": "spk_0", "text": "Do you have any nausea or vomiting?"},
        {"speaker_id": "spk_1", "text": "I have nausea sometimes but no vomiting. I also have some bloating."},
        {"speaker_id": "spk_0", "text": "Any changes in your bowel movements? Any diarrhea?"},
        {"speaker_id": "spk_1", "text": "No diarrhea. No blood in the stool either."},
        {"speaker_id": "spk_0", "text": "Are you taking any medications?"},
        {"speaker_id": "spk_1", "text": "Just omeprazole that my previous doctor prescribed."},
        {"speaker_id": "spk_0", "text": "I would like to order an ultrasound of your abdomen. The symptoms suggest we should check the gallbladder."},
    ],
    "ground_truth": {
        "symptoms": ["abdominal pain", "nausea", "bloating"],
        "negations": ["vomiting", "diarrhea", "blood in stool"],
        "medications": ["omeprazole"],
        "durations": ["two weeks"],
        "qualifiers": [
            {
                "symptom": "abdominal pain",
                "severity": None,
                "onset": None,
                "character": "cramping",
                "pattern": "intermittent",
                "progression": None,
                "laterality": "right",
                "radiation": None,
                "aggravating_factors": ["fatty meals"],
                "relieving_factors": [],
            },
        ],
        "speaker_roles": {
            "spk_0": "clinician",
            "spk_1": "patient",
        },
        "expected_patterns": [],
        "expected_red_flags": [],
    },
})


def get_scenario(case_id: str) -> dict:
    """Return a scenario by case_id.  Raises KeyError if not found."""
    return SCENARIOS[case_id]


def list_scenarios() -> list[str]:
    """Return sorted list of all registered case IDs."""
    return sorted(SCENARIOS.keys())
