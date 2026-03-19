"""Case analysis — failure analysis and evaluation reporting.

Aggregates scoring results across multiple cases to identify failure
patterns, weaknesses, and performance trends.  Strictly analytical —
never influences clinical reasoning or scoring logic.

Pure functions — no I/O, no ML, no input mutation, deterministic.
"""

from __future__ import annotations

import copy


# ── helpers ─────────────────────────────────────────────────────────


def safe_get_score_fields(case_result: dict) -> dict:
    """Extract score fields from a case result, tolerating missing data.

    Args:
        case_result: dict with optional ``score`` key.

    Returns:
        Dict with normalised score fields (defaults for missing data).
    """
    score = case_result.get("score") or {}
    summary = score.get("summary") or {}
    hyp = score.get("hypotheses") or {}
    rf = score.get("red_flags") or {}
    kf = score.get("key_findings") or {}
    adv = case_result.get("adversarial") or {}

    return {
        "case_id": case_result.get("case_id", ""),
        "has_ground_truth": score.get("has_ground_truth", False),
        "hypothesis_hit_rate": summary.get("hypothesis_hit_rate", 0.0),
        "red_flag_hit_rate": summary.get("red_flag_hit_rate", 0.0),
        "key_finding_hit_rate": summary.get("key_finding_hit_rate", 0.0),
        "top_hypothesis_expected": summary.get("top_hypothesis_expected", False),
        "missing_hypotheses": list(hyp.get("missing") or []),
        "missing_red_flags": list(rf.get("missing") or []),
        "missing_key_findings": list(kf.get("missing") or []),
        "strategy": adv.get("strategy", ""),
    }


# ── main analysis ──────────────────────────────────────────────────


def analyze_case_results(case_results: list[dict]) -> dict:
    """Aggregate scoring results and identify failure patterns.

    Args:
        case_results: list of dicts, each containing at minimum
            ``case_id`` and ``score`` (from :mod:`app.case_scoring`).
            May also contain ``adversarial`` metadata.

    Returns:
        Structured analysis dict with stable schema.
    """
    records = [safe_get_score_fields(cr) for cr in case_results]
    scored = [r for r in records if r["has_ground_truth"]]

    return {
        "overall": _compute_overall(records, scored),
        "worst_cases": _compute_worst_cases(scored),
        "strategy_breakdown": _compute_strategy_breakdown(records),
        "hypothesis_failures": _compute_frequency("missing_hypotheses", scored),
        "red_flag_failures": _compute_frequency("missing_red_flags", scored),
        "key_finding_failures": _compute_frequency("missing_key_findings", scored),
        "score_distribution": _compute_distribution(scored),
    }


# ── overall metrics ─────────────────────────────────────────────────


def _compute_overall(records: list[dict], scored: list[dict]) -> dict:
    n = len(records)
    n_scored = len(scored)

    if n_scored == 0:
        return {
            "total_cases": n,
            "scored_cases": 0,
            "avg_score": 0.0,
            "avg_hypothesis_hit_rate": 0.0,
            "avg_red_flag_hit_rate": 0.0,
            "avg_key_finding_hit_rate": 0.0,
        }

    avg_hyp = sum(r["hypothesis_hit_rate"] for r in scored) / n_scored
    avg_rf = sum(r["red_flag_hit_rate"] for r in scored) / n_scored
    avg_kf = sum(r["key_finding_hit_rate"] for r in scored) / n_scored
    avg_score = (avg_hyp + avg_rf + avg_kf) / 3.0

    return {
        "total_cases": n,
        "scored_cases": n_scored,
        "avg_score": round(avg_score, 4),
        "avg_hypothesis_hit_rate": round(avg_hyp, 4),
        "avg_red_flag_hit_rate": round(avg_rf, 4),
        "avg_key_finding_hit_rate": round(avg_kf, 4),
    }


# ── worst cases ─────────────────────────────────────────────────────


def _case_score(record: dict) -> float:
    """Compute a composite score for ranking."""
    return (
        record["hypothesis_hit_rate"]
        + record["red_flag_hit_rate"]
        + record["key_finding_hit_rate"]
    ) / 3.0


def _compute_worst_cases(scored: list[dict], n: int = 5) -> list[dict]:
    ranked = sorted(scored, key=_case_score)
    worst: list[dict] = []
    for r in ranked[:n]:
        worst.append({
            "case_id": r["case_id"],
            "score": round(_case_score(r), 4),
            "missing_hypotheses": r["missing_hypotheses"],
            "missing_red_flags": r["missing_red_flags"],
            "missing_key_findings": r["missing_key_findings"],
        })
    return worst


# ── strategy breakdown ──────────────────────────────────────────────


def _compute_strategy_breakdown(records: list[dict]) -> list[dict]:
    strategy_groups: dict[str, list[dict]] = {}
    for r in records:
        strat = r["strategy"]
        if not strat:
            continue
        strategy_groups.setdefault(strat, []).append(r)

    breakdown: list[dict] = []
    for strat in sorted(strategy_groups.keys()):
        group = strategy_groups[strat]
        scored = [r for r in group if r["has_ground_truth"]]
        count = len(group)

        if scored:
            avg = sum(_case_score(r) for r in scored) / len(scored)
        else:
            avg = 0.0

        breakdown.append({
            "strategy": strat,
            "count": count,
            "avg_score": round(avg, 4),
        })

    return breakdown


# ── failure frequency ───────────────────────────────────────────────


def _compute_frequency(field: str, scored: list[dict], n: int = 5) -> list[dict]:
    counts: dict[str, int] = {}
    for r in scored:
        for item in r.get(field, []):
            key = item.strip().lower()
            if key:
                counts[key] = counts.get(key, 0) + 1

    # Sort by count desc, then alphabetically.
    ranked = sorted(counts.items(), key=lambda x: (-x[1], x[0]))
    return [{"item": item, "count": count} for item, count in ranked[:n]]


# ── score distribution ──────────────────────────────────────────────


_BUCKETS = [
    (0.0, 0.2),
    (0.2, 0.4),
    (0.4, 0.6),
    (0.6, 0.8),
    (0.8, 1.0),
]


def _compute_distribution(scored: list[dict]) -> list[dict]:
    bucket_counts = [0] * len(_BUCKETS)

    for r in scored:
        s = _case_score(r)
        for i, (lo, hi) in enumerate(_BUCKETS):
            if lo <= s < hi or (i == len(_BUCKETS) - 1 and s == hi):
                bucket_counts[i] += 1
                break

    return [
        {
            "range": f"{lo:.1f}-{hi:.1f}",
            "count": bucket_counts[i],
        }
        for i, (lo, hi) in enumerate(_BUCKETS)
    ]
