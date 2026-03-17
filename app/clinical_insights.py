"""Clinical insights — gap analysis and next-step prompts from state.

Rule-based identification of missing information, uncertainties,
suggested follow-up questions, and data quality issues.  Derived
only from existing clinical state — no new reasoning or AI.

Pure function — no I/O, no ML, no input mutation, deterministic.
"""

from __future__ import annotations


def derive_clinical_insights(clinical_state: dict) -> dict:
    """Derive clinical insights from the full clinical state.

    Args:
        clinical_state: dict produced by :func:`build_clinical_state`.

    Returns:
        dict with keys: ``missing_information``, ``uncertainties``,
        ``suggested_questions``, ``data_quality_issues``.
    """
    return {
        "missing_information": _find_missing_information(clinical_state),
        "uncertainties": _find_uncertainties(clinical_state),
        "suggested_questions": _suggest_questions(clinical_state),
        "data_quality_issues": _find_data_quality_issues(clinical_state),
    }


# ── missing information ─────────────────────────────────────────────


def _find_missing_information(state: dict) -> list[dict]:
    """Identify gaps in the clinical record."""
    gaps: list[dict] = []

    symptoms = state.get("symptoms", [])
    durations = state.get("durations", [])
    medications = state.get("medications", [])
    timeline = state.get("timeline", [])
    derived = state.get("derived", {})
    structured = derived.get("structured_symptoms", [])

    # Symptoms without duration
    symptoms_with_duration = {
        e.get("symptom", "").lower()
        for e in timeline
        if e.get("time_expression")
    }
    for symptom in symptoms:
        if symptom.lower() not in symptoms_with_duration:
            gaps.append({
                "category": "missing_duration",
                "detail": f"No duration recorded for: {symptom}",
                "related": symptom,
            })

    # Symptoms without severity
    symptoms_with_severity = set()
    for ss in structured:
        if ss.get("severity"):
            symptoms_with_severity.add(ss.get("symptom", "").lower())
    for symptom in symptoms:
        if symptom.lower() not in symptoms_with_severity:
            gaps.append({
                "category": "missing_severity",
                "detail": f"No severity recorded for: {symptom}",
                "related": symptom,
            })

    # Medications without dosage (from review flags)
    review_flags = state.get("review_flags", [])
    for flag in review_flags:
        if flag.get("type") == "medication_without_dosage":
            med = flag.get("medication", flag.get("detail", ""))
            gaps.append({
                "category": "missing_dosage",
                "detail": f"No dosage recorded for medication: {med}",
                "related": med,
            })

    # No medications documented despite active problems
    problems = state.get("problems", [])
    if problems and not medications:
        gaps.append({
            "category": "missing_medications",
            "detail": "Active problems present but no medications documented",
            "related": None,
        })

    return gaps


# ── uncertainties ───────────────────────────────────────────────────


def _find_uncertainties(state: dict) -> list[dict]:
    """Identify areas of diagnostic uncertainty."""
    uncertainties: list[dict] = []

    hypotheses = state.get("hypotheses", [])
    problems = state.get("problems", [])

    # Multiple competing hypotheses with similar scores
    if len(hypotheses) >= 2:
        scores = [h.get("score", 0) for h in hypotheses]
        if scores[0] == scores[1]:
            titles = [h.get("title", "") for h in hypotheses[:2]]
            uncertainties.append({
                "category": "competing_hypotheses",
                "detail": f"Equal scoring between: {titles[0]}, {titles[1]}",
                "related": titles,
            })

    # Low-confidence hypotheses
    for hyp in hypotheses:
        if hyp.get("confidence") == "low" and hyp.get("rank") == 1:
            uncertainties.append({
                "category": "low_confidence_top_hypothesis",
                "detail": f"Top hypothesis has low confidence: {hyp.get('title', '')}",
                "related": hyp.get("title", ""),
            })

    # Symptoms without any hypothesis coverage
    hypothesis_symptoms: set[str] = set()
    observations = state.get("observations", [])
    obs_index = {o.get("observation_id", ""): o for o in observations}
    for hyp in hypotheses:
        for ev in hyp.get("supporting_observations", []):
            obs_id = ev.get("observation_id", "") if isinstance(ev, dict) else ev
            obs = obs_index.get(obs_id, {})
            if obs.get("finding_type") == "symptom":
                hypothesis_symptoms.add(obs.get("value", "").lower())

    symptoms = state.get("symptoms", [])
    for symptom in symptoms:
        if symptom.lower() not in hypothesis_symptoms and hypotheses:
            uncertainties.append({
                "category": "unexplained_symptom",
                "detail": f"Symptom not covered by any hypothesis: {symptom}",
                "related": symptom,
            })

    return uncertainties


# ── suggested questions ─────────────────────────────────────────────


def _suggest_questions(state: dict) -> list[dict]:
    """Generate follow-up question suggestions based on gaps."""
    questions: list[dict] = []

    symptoms = state.get("symptoms", [])
    durations = state.get("durations", [])
    medications = state.get("medications", [])
    timeline = state.get("timeline", [])
    negations = state.get("negations", [])
    derived = state.get("derived", {})

    # Duration questions for symptoms without timing
    symptoms_with_duration = {
        e.get("symptom", "").lower()
        for e in timeline
        if e.get("time_expression")
    }
    for symptom in symptoms:
        if symptom.lower() not in symptoms_with_duration:
            questions.append({
                "category": "duration",
                "question": f"How long have you had {symptom}?",
                "related": symptom,
            })

    # Severity questions
    structured = derived.get("structured_symptoms", [])
    symptoms_with_severity = set()
    for ss in structured:
        if ss.get("severity"):
            symptoms_with_severity.add(ss.get("symptom", "").lower())
    for symptom in symptoms:
        if symptom.lower() not in symptoms_with_severity:
            questions.append({
                "category": "severity",
                "question": f"How would you rate the severity of your {symptom}?",
                "related": symptom,
            })

    # Allergy question if medications present but no allergy info
    history = state.get("history", {})
    allergy_mentioned = bool(history.get("allergies"))
    if medications and not allergy_mentioned:
        questions.append({
            "category": "allergy",
            "question": "Do you have any known drug allergies?",
            "related": None,
        })

    return questions


# ── data quality issues ─────────────────────────────────────────────


def _find_data_quality_issues(state: dict) -> list[dict]:
    """Identify data quality concerns in the clinical record."""
    issues: list[dict] = []

    review_flags = state.get("review_flags", [])
    observations = state.get("observations", [])
    segments_count = len([
        o for o in observations if o.get("finding_type") == "symptom"
    ])

    # Low-confidence ASR segments
    low_conf_count = sum(
        1 for f in review_flags
        if f.get("type") == "low_confidence_segment"
    )
    if low_conf_count > 0:
        issues.append({
            "category": "low_confidence_transcription",
            "detail": f"{low_conf_count} segment(s) flagged for low ASR confidence",
            "count": low_conf_count,
        })

    # Very short session (few observations)
    if 0 < segments_count < 2:
        issues.append({
            "category": "limited_data",
            "detail": "Very few symptom observations recorded",
            "count": segments_count,
        })

    # Missing metrics in review flags
    missing_metrics = sum(
        1 for f in review_flags
        if f.get("type") == "missing_metrics"
    )
    if missing_metrics > 0:
        issues.append({
            "category": "missing_asr_metrics",
            "detail": f"{missing_metrics} segment(s) missing ASR quality metrics",
            "count": missing_metrics,
        })

    return issues
