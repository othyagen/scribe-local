#!/usr/bin/env python3
"""Interactive clinical case simulation runner.

Uses the clinical session facade and metrics module to simulate a
consultation loop: initialise → view → answer questions → update → repeat.

Run from the project root:
    python -m scripts.run_case_simulation
"""

from __future__ import annotations

import re
import sys

from app.clinical_session import (
    initialize_session,
    get_app_view,
    submit_answers,
    apply_manual_update,
)
from app.clinical_metrics import derive_clinical_metrics


# ── case data ────────────────────────────────────────────────────────


def _build_case_segments() -> list[dict]:
    """Return hardcoded case segments."""
    texts = [
        "65-year-old male with cough and fever for 3 days.",
        "Feels more short of breath today.",
    ]
    segments = []
    t = 0.0
    for i, text in enumerate(texts):
        seg_id = f"seg_{i + 1:04d}"
        segments.append({
            "seg_id": seg_id,
            "t0": t,
            "t1": t + 3.0,
            "speaker_id": "spk_0",
            "normalized_text": text,
        })
        t += 3.0
    return segments


_DEFAULT_CONFIG: dict = {
    "mode": "assist",
    "update_strategy": "manual",
    "show_summary_views": True,
    "show_insights": True,
    "show_questions": True,
}


# ── question type mapping ────────────────────────────────────────────


_TYPE_PATTERNS: list[tuple[str, list[str]]] = [
    ("duration", ["duration", "how long", "when did it start"]),
    ("severity", ["severity", "how severe", "how bad"]),
    ("allergy", ["allergy", "allergies"]),
    ("dosage", ["dose", "dosage", "medication"]),
]


def classify_question_type(question_text: str) -> str | None:
    """Map a question string to a supported structured answer type.

    Returns the answer type string, or ``None`` if unsupported.
    """
    lower = question_text.lower()
    for answer_type, keywords in _TYPE_PATTERNS:
        for kw in keywords:
            if kw in lower:
                return answer_type
    return None


def build_structured_answer(
    question: dict,
    answer_text: str,
) -> dict | None:
    """Convert a question + user answer into a structured answer dict.

    Returns ``None`` if the question type is not supported.
    """
    q_type = classify_question_type(question.get("question", ""))
    if q_type is None:
        return None

    related = question.get("related") if "related" in question else None
    # Infer related from reason if not directly available.
    if related is None:
        reason = question.get("reason", "")
        # Try to extract the symptom/medication name from reason text.
        match = re.search(r"for\s+(.+)$", reason, re.IGNORECASE)
        if match:
            related = match.group(1).strip()

    return {
        "type": q_type,
        "value": answer_text.strip(),
        "related": related,
    }


# ── formatting functions ─────────────────────────────────────────────


def format_summary_text(app_view: dict) -> str:
    """Format the clinical summary section."""
    lines: list[str] = ["=== SUMMARY ==="]
    vo = app_view.get("orchestrated", {}).get("visible_outputs", {})
    summary = vo.get("clinical_summary")
    if not summary:
        lines.append("  (no summary available)")
        return "\n".join(lines)

    narrative = summary.get("problem_narrative", {})
    narrative_text = narrative.get("narrative", "") if isinstance(narrative, dict) else ""
    if narrative_text:
        lines.append(f"  {narrative_text}")
    else:
        # Fallback: list key findings.
        findings = summary.get("key_findings", [])
        if findings:
            for f in findings[:6]:
                lines.append(f"  - [{f.get('type', '?')}] {f.get('value', '')}")
        else:
            lines.append("  (no findings)")

    return "\n".join(lines)


def format_hypotheses_text(app_view: dict) -> str:
    """Format the top hypotheses section."""
    lines: list[str] = ["=== TOP HYPOTHESES ==="]
    vo = app_view.get("orchestrated", {}).get("visible_outputs", {})
    summary = vo.get("clinical_summary", {})
    hypotheses = summary.get("ranked_hypotheses", []) if summary else []

    if not hypotheses:
        lines.append("  (none)")
        return "\n".join(lines)

    for h in hypotheses[:3]:
        title = h.get("title", "?")
        score = h.get("score", 0)
        rank = h.get("rank", "?")
        conf = h.get("confidence", "?")
        lines.append(f"  #{rank} {title} (score={score}, confidence={conf})")

    return "\n".join(lines)


def format_red_flags_text(app_view: dict) -> str:
    """Format the red flags section."""
    lines: list[str] = ["=== RED FLAGS ==="]
    vo = app_view.get("orchestrated", {}).get("visible_outputs", {})
    summary = vo.get("clinical_summary", {})
    red_flags = summary.get("red_flags", []) if summary else []

    if not red_flags:
        lines.append("  None")
        return "\n".join(lines)

    for rf in red_flags:
        label = rf.get("label", "?")
        evidence = rf.get("evidence", [])
        lines.append(f"  * {label} (evidence: {', '.join(str(e) for e in evidence)})")

    return "\n".join(lines)


def format_questions_text(questions: list[dict]) -> str:
    """Format the suggested questions section."""
    lines: list[str] = ["=== QUESTIONS ==="]
    if not questions:
        lines.append("  (no questions)")
        return "\n".join(lines)

    for i, q in enumerate(questions, 1):
        priority = q.get("priority", "?")
        text = q.get("question", "?")
        lines.append(f"  {i}. [{priority}] {text}")

    return "\n".join(lines)


def format_metrics_text(metrics: dict) -> str:
    """Format the metrics section."""
    lines: list[str] = ["=== METRICS ==="]

    om = metrics.get("observation_metrics", {})
    pm = metrics.get("problem_metrics", {})
    hm = metrics.get("hypothesis_metrics", {})
    rm = metrics.get("risk_metrics", {})
    im = metrics.get("interaction_metrics", {})
    dq = metrics.get("data_quality_metrics", {})
    um = metrics.get("update_metrics", {})

    lines.append(f"  observations:         {om.get('observation_count', 0)}")
    lines.append(f"  problems:             {pm.get('problem_count', 0)}")
    lines.append(f"  hypotheses:           {hm.get('hypothesis_count', 0)}")
    lines.append(f"  red flags:            {rm.get('red_flag_count', 0)}")
    lines.append(f"  questions:            {im.get('question_count', 0)}")
    lines.append(f"  missing information:  {dq.get('missing_information_count', 0)}")

    pending = um.get("pending_count")
    lines.append(f"  pending observations: {pending if pending is not None else 'n/a'}")

    return "\n".join(lines)


def format_pending_status(session: dict) -> str:
    """Format pending update status."""
    pending = session.get("pending_observations", [])
    count = len(pending)
    if count == 0:
        return "  No pending updates."
    return f"  ** {count} pending observation(s) — type 'update' to apply **"


def build_display_bundle(
    app_view: dict,
    questions: list[dict],
    metrics: dict,
    session: dict,
) -> dict:
    """Build a structured display bundle for all sections.

    Returns a dict of section names to formatted text strings.
    Can be consumed by print, TTS, or any other presentation layer.
    """
    return {
        "summary": format_summary_text(app_view),
        "hypotheses": format_hypotheses_text(app_view),
        "red_flags": format_red_flags_text(app_view),
        "questions": format_questions_text(questions),
        "metrics": format_metrics_text(metrics),
        "pending_status": format_pending_status(session),
    }


def render_display(bundle: dict) -> str:
    """Render the display bundle as a single text block."""
    sections = [
        bundle["summary"],
        bundle["hypotheses"],
        bundle["red_flags"],
        bundle["questions"],
        bundle["metrics"],
        bundle["pending_status"],
    ]
    return "\n\n".join(sections)


# ── main loop ────────────────────────────────────────────────────────


def _get_questions_from_view(app_view: dict) -> list[dict]:
    """Extract questions from the app view."""
    vo = app_view.get("orchestrated", {}).get("visible_outputs", {})
    return vo.get("clinical_questions") or []


def run_simulation() -> None:
    """Run the interactive case simulation."""
    segments = _build_case_segments()
    session = initialize_session(segments, config=_DEFAULT_CONFIG)

    print("Clinical Case Simulation")
    print("=" * 40)
    print()

    while True:
        # Build view and metrics.
        app_view = get_app_view(session)
        metrics = derive_clinical_metrics(session["clinical_state"])
        questions = _get_questions_from_view(app_view)

        # Build and render display.
        bundle = build_display_bundle(app_view, questions, metrics, session)
        print(render_display(bundle))
        print()

        # Prompt.
        try:
            user_input = input(
                "Choose a question number, or type 'update', 'view', or 'exit': "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue

        if user_input.lower() == "exit":
            print("Exiting.")
            break

        if user_input.lower() == "view":
            continue

        if user_input.lower() == "update":
            session = apply_manual_update(session)
            print("\n>> Update applied.\n")
            continue

        # Try to parse as question number.
        try:
            choice = int(user_input)
        except ValueError:
            print(f"\n>> Unknown command: {user_input}\n")
            continue

        if choice < 1 or choice > len(questions):
            print(f"\n>> Invalid question number: {choice}\n")
            continue

        selected = questions[choice - 1]
        q_type = classify_question_type(selected.get("question", ""))

        if q_type is None:
            print(
                f"\n>> Question type not supported for structured answers: "
                f"'{selected.get('question', '')}'\n"
            )
            continue

        try:
            answer_text = input(f"Your answer for '{selected['question']}': ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not answer_text:
            print("\n>> Empty answer, skipping.\n")
            continue

        structured = build_structured_answer(selected, answer_text)
        if structured is None:
            print("\n>> Could not build structured answer.\n")
            continue

        session = submit_answers(session, [structured])
        result = session.get("answer_result", {})
        new_count = len(result.get("new_observations", []))
        unparsed = len(result.get("unparsed_answers", []))
        print(f"\n>> Answer submitted. {new_count} observation(s) created.")
        if unparsed:
            print(f"   {unparsed} answer(s) could not be parsed.")
        print()


if __name__ == "__main__":
    run_simulation()
