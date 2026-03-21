"""Hypothesis evidence gaps — missing-evidence identification and targeted questions.

For each hypothesis with a known condition profile, identifies which
expected findings are present, negated, or absent, and generates
condition-discriminating follow-up questions for absent findings.

Purely additive post-processing layer — never modifies hypotheses,
prioritization, ranking, or any other existing output.

Pure function — no I/O, no ML, no input mutation, deterministic.
"""

from __future__ import annotations

import re

from app.diagnostic_hints import _extract_negated_symptoms


# ── expected-findings registry (lean v1) ─────────────────────────────
#
# Keys are lowercase condition names matching diagnostic_hints.RULES
# and/or hypothesis_prioritization.DANGEROUS_CONDITIONS.
#
# Each entry lists clinically discriminating findings beyond what the
# RULES required-symptoms already capture.  Keep lean: 3-5 per condition,
# focused on findings that change clinical management.

CONDITION_FINDINGS: dict[str, list[dict]] = {
    # ── dangerous conditions (must-not-miss safety value) ────────────
    "acute coronary syndrome": [
        {"finding": "chest pain", "question": "Do you have chest pain or pressure?",
         "reason": "Chest pain is the cardinal symptom of acute coronary syndrome"},
        {"finding": "exertional pattern", "question": "Does the pain come on with exertion?",
         "reason": "Exertional chest pain suggests cardiac ischaemia and helps distinguish ACS from musculoskeletal causes"},
        {"finding": "radiation", "question": "Does it spread to the arm, jaw, or back?",
         "reason": "Radiation to arm, jaw, or back is a classic ACS feature that increases diagnostic likelihood"},
        {"finding": "diaphoresis", "question": "Have you been sweating excessively?",
         "reason": "Diaphoresis with chest pain is a high-specificity sign for myocardial infarction"},
        {"finding": "nausea", "question": "Have you had nausea?",
         "reason": "Nausea accompanies ACS due to vagal stimulation and supports the diagnosis"},
    ],
    "pulmonary embolism": [
        {"finding": "dyspnea", "question": "Are you short of breath?",
         "reason": "Dyspnea is the most common symptom of pulmonary embolism"},
        {"finding": "chest pain", "question": "Do you have chest pain?",
         "reason": "Pleuritic chest pain suggests pulmonary infarction and helps confirm PE"},
        {"finding": "hemoptysis", "question": "Have you coughed up blood?",
         "reason": "Hemoptysis indicates pulmonary infarction and is a key discriminating feature for PE"},
        {"finding": "unilateral leg swelling", "question": "Is one leg swollen or painful?",
         "reason": "Unilateral leg swelling suggests DVT, a common source of pulmonary embolism"},
        {"finding": "sudden onset", "question": "Did the symptoms start suddenly?",
         "reason": "Sudden onset is characteristic of PE and helps distinguish it from gradual respiratory conditions"},
    ],
    "meningitis": [
        {"finding": "headache", "question": "Do you have a headache?",
         "reason": "Headache is a core feature of the meningitis triad"},
        {"finding": "fever", "question": "Have you had a fever?",
         "reason": "Fever indicates systemic infection and is part of the classic meningitis triad"},
        {"finding": "neck stiffness", "question": "Is your neck stiff?",
         "reason": "Neck stiffness (nuchal rigidity) is a hallmark sign of meningeal irritation"},
        {"finding": "photophobia", "question": "Does light bother your eyes?",
         "reason": "Photophobia suggests meningeal inflammation and helps confirm meningitis"},
    ],
    "sepsis": [
        {"finding": "fever", "question": "Have you had a fever or chills?",
         "reason": "Fever or hypothermia is a SIRS criterion indicating systemic infection"},
        {"finding": "tachycardia", "question": "Has your heart been racing?",
         "reason": "Tachycardia is a SIRS criterion and an early sign of haemodynamic compromise in sepsis"},
        {"finding": "confusion", "question": "Have you felt confused or disoriented?",
         "reason": "Altered mental status indicates end-organ dysfunction and raises sepsis severity"},
        {"finding": "hypotension", "question": "Have you felt faint or lightheaded?",
         "reason": "Hypotension suggests septic shock and requires urgent intervention"},
    ],
    "aortic dissection": [
        {"finding": "chest pain", "question": "Do you have chest or back pain?",
         "reason": "Severe chest or back pain is the presenting symptom in most aortic dissections"},
        {"finding": "sudden onset", "question": "Did the pain start suddenly?",
         "reason": "Sudden-onset maximal pain is the classic presentation that distinguishes dissection from other causes"},
        {"finding": "tearing quality", "question": "Does it feel like a tearing or ripping sensation?",
         "reason": "Tearing or ripping quality is highly characteristic of aortic dissection"},
    ],
    "stroke": [
        {"finding": "weakness", "question": "Do you have weakness in your face, arm, or leg?",
         "reason": "Focal weakness is the most common stroke presentation and localises the lesion"},
        {"finding": "speech difficulty", "question": "Have you had trouble speaking?",
         "reason": "Speech difficulty suggests cortical involvement and is a key stroke recognition sign"},
        {"finding": "sudden onset", "question": "Did the symptoms start suddenly?",
         "reason": "Sudden onset of neurological deficit is the hallmark of stroke versus gradual conditions"},
        {"finding": "vision changes", "question": "Have you had any vision changes?",
         "reason": "Visual field deficits suggest posterior circulation stroke"},
    ],
    "tension pneumothorax": [
        {"finding": "dyspnea", "question": "Are you short of breath?",
         "reason": "Progressive dyspnea is an early sign of tension pneumothorax"},
        {"finding": "chest pain", "question": "Do you have chest pain?",
         "reason": "Acute pleuritic chest pain helps localise the pneumothorax"},
        {"finding": "sudden onset", "question": "Did the symptoms start suddenly?",
         "reason": "Sudden onset of dyspnea and chest pain is characteristic of pneumothorax"},
    ],
    "ectopic pregnancy": [
        {"finding": "abdominal pain", "question": "Do you have abdominal or pelvic pain?",
         "reason": "Lower abdominal or pelvic pain is the most common symptom of ectopic pregnancy"},
        {"finding": "vaginal bleeding", "question": "Have you had any vaginal bleeding?",
         "reason": "Vaginal bleeding with pain in early pregnancy is the classic ectopic presentation"},
        {"finding": "missed period", "question": "Have you missed a menstrual period?",
         "reason": "Missed period establishes pregnancy status, which is required to consider ectopic"},
    ],
    "gastrointestinal bleeding": [
        {"finding": "hematemesis", "question": "Have you vomited blood?",
         "reason": "Hematemesis confirms upper GI bleeding and indicates the source is proximal"},
        {"finding": "melena", "question": "Have your stools been dark or tarry?",
         "reason": "Melena indicates significant upper GI haemorrhage requiring evaluation"},
        {"finding": "dizziness", "question": "Have you felt dizzy or lightheaded?",
         "reason": "Dizziness suggests haemodynamic compromise from blood loss"},
    ],
    "gi bleed": [
        {"finding": "hematemesis", "question": "Have you vomited blood?",
         "reason": "Hematemesis confirms upper GI bleeding and indicates the source is proximal"},
        {"finding": "melena", "question": "Have your stools been dark or tarry?",
         "reason": "Melena indicates significant upper GI haemorrhage requiring evaluation"},
        {"finding": "dizziness", "question": "Have you felt dizzy or lightheaded?",
         "reason": "Dizziness suggests haemodynamic compromise from blood loss"},
    ],
    # ── common RULES conditions with discriminating value ────────────
    "pneumonia": [
        {"finding": "fever", "question": "Have you had a fever?",
         "reason": "Fever supports an infectious aetiology and helps confirm pneumonia"},
        {"finding": "cough", "question": "Do you have a cough?",
         "reason": "Cough is a core respiratory symptom expected in pneumonia"},
        {"finding": "dyspnea", "question": "Are you short of breath?",
         "reason": "Dyspnea indicates respiratory compromise and supports pneumonia over upper tract infection"},
        {"finding": "chest pain", "question": "Do you have chest pain when breathing?",
         "reason": "Pleuritic chest pain suggests pleural involvement and helps distinguish pneumonia from bronchitis"},
        {"finding": "sputum", "question": "Are you coughing up sputum or phlegm?",
         "reason": "Productive sputum suggests lower respiratory infection and helps differentiate from viral URI"},
    ],
    "migraine": [
        {"finding": "headache", "question": "Do you have a headache?",
         "reason": "Headache is the defining symptom of migraine"},
        {"finding": "photophobia", "question": "Does light bother you?",
         "reason": "Photophobia is a key migraine feature that helps distinguish it from tension headache"},
        {"finding": "nausea", "question": "Have you felt nauseous?",
         "reason": "Nausea is part of the migraine symptom complex and supports the diagnosis"},
        {"finding": "aura", "question": "Did you see flashing lights or spots before the headache?",
         "reason": "Visual aura is a highly specific migraine feature that confirms the diagnosis subtype"},
    ],
    "urinary tract infection": [
        {"finding": "dysuria", "question": "Does it burn when you urinate?",
         "reason": "Dysuria is the most predictive symptom for urinary tract infection"},
        {"finding": "urinary frequency", "question": "Are you urinating more often?",
         "reason": "Increased frequency supports lower urinary tract irritation from infection"},
        {"finding": "fever", "question": "Have you had a fever?",
         "reason": "Fever with urinary symptoms suggests upper tract involvement (pyelonephritis)"},
        {"finding": "flank pain", "question": "Do you have pain in your side or back?",
         "reason": "Flank pain suggests pyelonephritis and indicates the infection has ascended"},
    ],
}


# ── priority class ordering ──────────────────────────────────────────

_PRIORITY_ORDER: dict[str, int] = {
    "must_not_miss": 0,
    "most_likely": 1,
    "less_likely": 2,
}


# ── public API ───────────────────────────────────────────────────────


def identify_evidence_gaps(
    hypotheses: list[dict],
    prioritization: list[dict],
    observations: list[dict],
    negations: list[str],
) -> dict:
    """Identify missing evidence and generate targeted questions per hypothesis.

    Args:
        hypotheses: ranked hypothesis list from ``state["hypotheses"]``.
        prioritization: priority list from ``state["hypothesis_prioritization"]``.
        observations: observation list from ``state["observations"]``.
        negations: negation strings from ``state["negations"]``.

    Returns:
        dict with ``missing_evidence`` (list) and ``suggested_questions`` (list).
    """
    if not hypotheses:
        return {"missing_evidence": [], "suggested_questions": []}

    # Build lookup structures.
    prio_index = _build_priority_index(prioritization)
    obs_values = _collect_observation_values(observations)
    obs_index = _build_obs_value_index(observations)
    negated = _extract_negated_symptoms(negations or [])

    missing_evidence: list[dict] = []
    raw_questions: list[tuple[int, int, dict]] = []  # (prio_sort, rank, question)

    for hyp in hypotheses:
        title = hyp.get("title", "")
        title_lower = title.strip().lower()
        if not title_lower or title_lower not in CONDITION_FINDINGS:
            continue

        hyp_id = hyp.get("id", "")
        rank = hyp.get("rank", 0)
        prio_class = prio_index.get(hyp_id, "less_likely")
        prio_sort = _PRIORITY_ORDER.get(prio_class, 2)

        # Collect supporting evidence values for this hypothesis.
        supporting_values = _collect_supporting_values(hyp, obs_index)

        expected = CONDITION_FINDINGS[title_lower]
        present: list[str] = []
        absent: list[str] = []
        negated_list: list[str] = []
        findings: list[dict] = []

        for entry in expected:
            finding = entry["finding"]
            reason = entry.get("reason", "")
            status = _classify_finding(finding, obs_values, supporting_values, negated)
            findings.append({
                "name": finding,
                "status": status,
                "reason": reason,
            })
            if status == "present":
                present.append(finding)
            elif status == "negated":
                negated_list.append(finding)
            else:
                absent.append(finding)
                raw_questions.append((prio_sort, rank, {
                    "question": entry["question"],
                    "target_hypothesis": title,
                    "target_finding": finding,
                    "priority_class": prio_class,
                    "reason": reason,
                }))

        missing_evidence.append({
            "hypothesis_id": hyp_id,
            "title": title,
            "priority_class": prio_class,
            "present": present,
            "absent": absent,
            "negated": negated_list,
            "findings": findings,
        })

    # Sort questions: priority class, then rank, then stable order.
    raw_questions.sort(key=lambda t: (t[0], t[1]))

    # Deduplicate by question text (keep higher-priority version).
    seen_questions: set[str] = set()
    suggested_questions: list[dict] = []
    for _, _, q in raw_questions:
        key = q["question"].lower()
        if key not in seen_questions:
            seen_questions.add(key)
            suggested_questions.append(q)

    return {
        "missing_evidence": missing_evidence,
        "suggested_questions": suggested_questions,
    }


# ── internal helpers ─────────────────────────────────────────────────


def _build_priority_index(prioritization: list[dict]) -> dict[str, str]:
    """Map hypothesis_id → priority_class from prioritization list."""
    return {
        p.get("hypothesis_id", ""): p.get("priority_class", "less_likely")
        for p in prioritization
    }


def _collect_observation_values(observations: list[dict]) -> set[str]:
    """Collect lowercase symptom observation values."""
    return {
        obs.get("value", "").strip().lower()
        for obs in observations
        if obs.get("finding_type") == "symptom" and obs.get("value")
    }


def _build_obs_value_index(observations: list[dict]) -> dict[str, str]:
    """Map observation_id → lowercase value for symptom observations."""
    return {
        obs["observation_id"]: obs.get("value", "").strip().lower()
        for obs in observations
        if obs.get("observation_id") and obs.get("finding_type") == "symptom"
    }


def _collect_supporting_values(
    hyp: dict,
    obs_index: dict[str, str],
) -> set[str]:
    """Collect lowercase values of a hypothesis's supporting observations."""
    values: set[str] = set()
    for ev in hyp.get("supporting_observations", []):
        if isinstance(ev, dict):
            obs_id = ev.get("observation_id", "")
        else:
            obs_id = ev
        val = obs_index.get(obs_id, "")
        if val:
            values.add(val)
    return values


def _classify_finding(
    finding: str,
    observation_values: set[str],
    supporting_values: set[str],
    negated_symptoms: set[str],
) -> str:
    """Classify a finding as 'present', 'negated', or 'absent'.

    Matching strategy (conservative):
    1. Exact canonical match (case-insensitive) against observation values
       and supporting evidence values.
    2. Limited fallback: if the finding is a multi-word phrase, check if it
       appears as a whole-word boundary within any observation value.
    3. Check negated symptoms.
    4. Otherwise absent.
    """
    finding_lower = finding.strip().lower()

    # 1. Exact match against observations and supporting evidence.
    all_values = observation_values | supporting_values
    if finding_lower in all_values:
        return "present"

    # 2. Limited substring fallback for multi-word findings only.
    if " " in finding_lower:
        pattern = re.compile(r"\b" + re.escape(finding_lower) + r"\b")
        for val in all_values:
            if pattern.search(val):
                return "present"

    # 3. Negation check.
    if finding_lower in negated_symptoms:
        return "negated"

    return "absent"
