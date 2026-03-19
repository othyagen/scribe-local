"""Case scoring — ground-truth comparison for case execution results.

Compares case execution outputs against declared ground truth and
produces structured evaluation scores.  Evaluation-only — never
affects reasoning, ranking, or case execution.

Pure functions — no I/O, no ML, no input mutation, deterministic.
"""

from __future__ import annotations

from app.case_system import run_case, run_case_script


# ── scoring ─────────────────────────────────────────────────────────


def score_result_against_ground_truth(result_bundle: dict) -> dict:
    """Score a result bundle against its ground truth.

    Args:
        result_bundle: result from :func:`run_case` or
            :func:`run_case_script`.

    Returns:
        Structured score dict with stable schema.
    """
    gt = result_bundle.get("ground_truth") or {}
    state = result_bundle.get("session", {}).get("clinical_state", {})
    has_gt = bool(
        gt.get("expected_hypotheses")
        or gt.get("red_flags")
        or gt.get("key_findings")
    )

    hyp_score = _score_hypotheses(gt, state)
    rf_score = _score_red_flags(gt, state)
    kf_score = _score_key_findings(gt, state)

    return {
        "case_id": result_bundle.get("case_id", ""),
        "has_ground_truth": has_gt,
        "hypotheses": hyp_score,
        "red_flags": rf_score,
        "key_findings": kf_score,
        "summary": {
            "hypothesis_hit_rate": hyp_score["hit_rate"],
            "hypothesis_expected_count": hyp_score["expected_count"],
            "hypothesis_matched_count": hyp_score["matched_count"],
            "red_flag_hit_rate": rf_score["hit_rate"],
            "red_flag_expected_count": rf_score["expected_count"],
            "red_flag_matched_count": rf_score["matched_count"],
            "key_finding_hit_rate": kf_score["hit_rate"],
            "key_finding_expected_count": kf_score["expected_count"],
            "key_finding_matched_count": kf_score["matched_count"],
            "top_hypothesis_expected": hyp_score["top_hypothesis_expected"],
        },
    }


def score_case_run(case: dict) -> dict:
    """Run a case and score the result.

    Args:
        case: parsed case dict.

    Returns:
        Dict with ``case_id``, ``result_bundle``, and ``score``.
    """
    result = run_case(case)
    score = score_result_against_ground_truth(result)
    return {
        "case_id": case.get("case_id", ""),
        "result_bundle": result,
        "score": score,
    }


def score_case_script_run(case: dict) -> dict:
    """Run a case with its answer script and score the result.

    Args:
        case: parsed case dict (should have ``answer_script``).

    Returns:
        Dict with ``case_id``, ``result_bundle``, and ``score``.
    """
    result = run_case_script(case)
    score = score_result_against_ground_truth(result)
    return {
        "case_id": case.get("case_id", ""),
        "result_bundle": result,
        "score": score,
    }


def summarize_score(score: dict) -> dict:
    """Return a compact summary of a score dict.

    Args:
        score: score dict from :func:`score_result_against_ground_truth`.

    Returns:
        Compact summary dict.
    """
    summary = score.get("summary", {})
    return {
        "case_id": score.get("case_id", ""),
        "has_ground_truth": score.get("has_ground_truth", False),
        "hypothesis_hit_rate": summary.get("hypothesis_hit_rate", 0.0),
        "red_flag_hit_rate": summary.get("red_flag_hit_rate", 0.0),
        "key_finding_hit_rate": summary.get("key_finding_hit_rate", 0.0),
        "top_hypothesis_expected": summary.get("top_hypothesis_expected", False),
    }


# ── internal helpers ────────────────────────────────────────────────


def _normalize(s: str) -> str:
    """Normalize a string for matching: lowercase + strip."""
    return s.strip().lower() if isinstance(s, str) else ""


def _score_hypotheses(gt: dict, state: dict) -> dict:
    """Score hypothesis presence against ground truth."""
    expected_raw = gt.get("expected_hypotheses") or []
    expected = [_normalize(h) for h in expected_raw]

    actual_hyps = state.get("hypotheses", [])
    actual_titles = [_normalize(h.get("title", "")) for h in actual_hyps]
    actual_set = set(actual_titles)

    present: list[str] = []
    missing: list[str] = []
    expected_ranks: dict[str, int | None] = {}

    for i, exp in enumerate(expected):
        raw = expected_raw[i] if i < len(expected_raw) else exp
        if exp in actual_set:
            present.append(raw)
            # Find rank (1-based).
            rank = next(
                (j + 1 for j, t in enumerate(actual_titles) if t == exp),
                None,
            )
            expected_ranks[raw] = rank
        else:
            missing.append(raw)
            expected_ranks[raw] = None

    # Top hypothesis check.
    top_hyp = actual_titles[0] if actual_titles else ""
    top_hyp_raw = actual_hyps[0].get("title", "") if actual_hyps else ""
    top_expected = top_hyp in expected if expected else False

    hit_rate = len(present) / len(expected) if expected else 0.0

    return {
        "expected": list(expected_raw),
        "present": present,
        "missing": missing,
        "expected_count": len(expected),
        "matched_count": len(present),
        "top_hypothesis": top_hyp_raw,
        "top_hypothesis_expected": top_expected,
        "expected_ranks": expected_ranks,
        "hit_rate": hit_rate,
    }


def _score_red_flags(gt: dict, state: dict) -> dict:
    """Score red flag presence against ground truth."""
    expected_raw = gt.get("red_flags") or []
    expected = [_normalize(rf) for rf in expected_raw]

    derived = state.get("derived", {})
    actual_flags = derived.get("red_flags", [])
    actual_labels = {_normalize(rf.get("label", "")) for rf in actual_flags}

    present: list[str] = []
    missing: list[str] = []

    for i, exp in enumerate(expected):
        raw = expected_raw[i] if i < len(expected_raw) else exp
        if exp in actual_labels:
            present.append(raw)
        else:
            missing.append(raw)

    hit_rate = len(present) / len(expected) if expected else 0.0

    return {
        "expected": list(expected_raw),
        "present": present,
        "missing": missing,
        "expected_count": len(expected),
        "matched_count": len(present),
        "hit_rate": hit_rate,
    }


def _score_key_findings(gt: dict, state: dict) -> dict:
    """Score key finding presence against ground truth."""
    expected_raw = gt.get("key_findings") or []
    expected = [_normalize(f) for f in expected_raw]

    actual_symptoms = {_normalize(s) for s in state.get("symptoms", [])}

    present: list[str] = []
    missing: list[str] = []

    for i, exp in enumerate(expected):
        raw = expected_raw[i] if i < len(expected_raw) else exp
        if exp in actual_symptoms:
            present.append(raw)
        else:
            missing.append(raw)

    hit_rate = len(present) / len(expected) if expected else 0.0

    return {
        "expected": list(expected_raw),
        "present": present,
        "missing": missing,
        "expected_count": len(expected),
        "matched_count": len(present),
        "hit_rate": hit_rate,
    }
