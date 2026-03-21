"""Case compare — text vs TTS execution comparison.

Runs the same case through both text mode and TTS mode, then produces
a structured diff isolating the effect of the audio path on clinical
understanding.

Pure orchestration — delegates to :func:`run_case` and
:func:`run_case_tts`, adds no new clinical logic.
"""

from __future__ import annotations

from pathlib import Path

from app.case_system import run_case
from app.case_tts import run_case_tts


# ── helpers ────────────────────────────────────────────────────────


def _set_diff(a: set, b: set) -> dict:
    """Return shared / a_only / b_only."""
    return {
        "shared": sorted(a & b),
        "text_only": sorted(a - b),
        "tts_only": sorted(b - a),
    }


def _extract_state(result: dict) -> dict:
    return result.get("session", {}).get("clinical_state", {})


def _extract_encounter(state: dict) -> dict:
    return state.get("encounter_output", {})


# ── comparison builders ────────────────────────────────────────────


def _compare_key_findings(text_eo: dict, tts_eo: dict) -> dict:
    text_kf = set(text_eo.get("key_findings", []))
    tts_kf = set(tts_eo.get("key_findings", []))
    return _set_diff(text_kf, tts_kf)


def _compare_red_flags(text_eo: dict, tts_eo: dict) -> dict:
    text_rf = {rf.get("label", "") for rf in text_eo.get("red_flags", [])}
    tts_rf = {rf.get("label", "") for rf in tts_eo.get("red_flags", [])}
    text_rf.discard("")
    tts_rf.discard("")
    return _set_diff(text_rf, tts_rf)


def _compare_hypotheses(text_eo: dict, tts_eo: dict) -> dict:
    text_hyps = {h["title"]: h for h in text_eo.get("hypotheses", []) if h.get("title")}
    tts_hyps = {h["title"]: h for h in tts_eo.get("hypotheses", []) if h.get("title")}

    text_titles = set(text_hyps.keys())
    tts_titles = set(tts_hyps.keys())
    shared = text_titles & tts_titles

    rank_changes = []
    for title in sorted(shared):
        text_rank = text_hyps[title].get("rank", 0)
        tts_rank = tts_hyps[title].get("rank", 0)
        if text_rank != tts_rank:
            rank_changes.append({
                "title": title,
                "text_rank": text_rank,
                "tts_rank": tts_rank,
            })

    return {
        "shared_titles": sorted(shared),
        "text_only_titles": sorted(text_titles - tts_titles),
        "tts_only_titles": sorted(tts_titles - text_titles),
        "rank_changes": rank_changes,
    }


def _compare_prioritization(text_eo: dict, tts_eo: dict) -> dict:
    text_prio = {
        h["title"]: h.get("priority_class", "")
        for h in text_eo.get("hypotheses", []) if h.get("title")
    }
    tts_prio = {
        h["title"]: h.get("priority_class", "")
        for h in tts_eo.get("hypotheses", []) if h.get("title")
    }

    all_titles = set(text_prio.keys()) | set(tts_prio.keys())

    unchanged = []
    changed = []
    dropped = []
    added = []

    for title in sorted(all_titles):
        in_text = title in text_prio
        in_tts = title in tts_prio
        if in_text and in_tts:
            if text_prio[title] == tts_prio[title]:
                unchanged.append({"title": title, "priority_class": text_prio[title]})
            else:
                changed.append({
                    "title": title,
                    "text_priority": text_prio[title],
                    "tts_priority": tts_prio[title],
                })
        elif in_text:
            dropped.append({"title": title, "priority_class": text_prio[title]})
        else:
            added.append({"title": title, "priority_class": tts_prio[title]})

    return {
        "unchanged": unchanged,
        "changed": changed,
        "dropped": dropped,
        "added": added,
    }


def _compare_questions(text_state: dict, tts_state: dict) -> dict:
    text_qs = {
        sq.get("question", "")
        for sq in (
            text_state.get("hypothesis_evidence_gaps", {})
            .get("suggested_questions", [])
        )
    }
    tts_qs = {
        sq.get("question", "")
        for sq in (
            tts_state.get("hypothesis_evidence_gaps", {})
            .get("suggested_questions", [])
        )
    }
    text_qs.discard("")
    tts_qs.discard("")
    return _set_diff(text_qs, tts_qs)


def _compare_metrics(text_result: dict, tts_result: dict) -> dict | None:
    text_metrics = text_result.get("metrics")
    tts_metrics = tts_result.get("metrics")
    if not text_metrics or not tts_metrics:
        return None
    return {"text": text_metrics, "tts": tts_metrics}


# ── public API ─────────────────────────────────────────────────────


def compare_case_modes(
    case: dict,
    output_dir: Path,
    *,
    provider: str = "edge",
    voice: str | None = None,
    lang: str | None = None,
    asr_engine: object | None = None,
) -> dict:
    """Run a case in text and TTS modes, return a structured comparison.

    Args:
        case: parsed case dict.
        output_dir: directory for TTS audio output.
        provider: TTS provider name (explicit).
        voice: optional provider-specific voice identifier.
        lang: optional language code hint.
        asr_engine: optional pre-initialised ASR engine.

    Returns:
        Dict with ``text_result``, ``tts_result``, and ``comparison``.
    """
    text_result = run_case(case)
    tts_result = run_case_tts(
        case, output_dir,
        provider=provider, voice=voice, lang=lang,
        asr_engine=asr_engine,
    )

    text_state = _extract_state(text_result)
    tts_state = _extract_state(tts_result)
    text_eo = _extract_encounter(text_state)
    tts_eo = _extract_encounter(tts_state)

    comparison = {
        "key_findings": _compare_key_findings(text_eo, tts_eo),
        "red_flags": _compare_red_flags(text_eo, tts_eo),
        "hypotheses": _compare_hypotheses(text_eo, tts_eo),
        "prioritization": _compare_prioritization(text_eo, tts_eo),
        "questions": _compare_questions(text_state, tts_state),
    }

    metrics = _compare_metrics(text_result, tts_result)
    if metrics is not None:
        comparison["metrics"] = metrics

    return {
        "text_result": text_result,
        "tts_result": tts_result,
        "comparison": comparison,
    }
