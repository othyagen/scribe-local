#!/usr/bin/env python3
"""Unified evaluation dashboard — single-entrypoint consolidated report.

Orchestrates existing case replay, variation, adversarial, Synthea
evaluation, scoring, and analysis into one combined report.

Run from the project root::

    python -m scripts.run_evaluation_dashboard
"""

from __future__ import annotations

from pathlib import Path

from app.case_system import (
    load_all_cases,
    validate_case,
    run_case,
    run_case_script,
)
from app.case_variations import generate_case_variations
from app.case_adversarial import generate_adversarial_cases
from app.case_scoring import score_result_against_ground_truth
from app.case_analysis import analyze_case_results
from app.synthea_import import load_synthea_patients, convert_patients_to_cases


_CASE_DIR = Path(__file__).resolve().parent.parent / "resources" / "cases"
_SYNTHEA_PATH = Path(__file__).resolve().parent.parent / "resources" / "synthea_sample.json"


# ── scoring helper ──────────────────────────────────────────────────


def _score_cases(cases: list[dict]) -> list[dict]:
    """Run and score a list of cases, returning scored result entries."""
    scored: list[dict] = []
    for case in cases:
        validation = validate_case(case)
        if not validation["valid"]:
            continue

        has_script = bool(case.get("answer_script"))
        result = run_case_script(case) if has_script else run_case(case)

        if not result.get("validation", {}).get("valid", False):
            continue

        score = score_result_against_ground_truth(result)
        entry: dict = {
            "case_id": case.get("case_id", ""),
            "score": score,
            "result_bundle": result,
        }
        if "adversarial" in case:
            entry["adversarial"] = case["adversarial"]
        scored.append(entry)

    return scored


# ── group runners ───────────────────────────────────────────────────


def run_base_case_group(case_dir: Path = _CASE_DIR) -> dict:
    """Load and evaluate base seed cases."""
    cases = load_all_cases(case_dir)
    # Exclude generated subdirectories by filtering to top-level only.
    cases = [c for c in cases if True]  # load_all_cases already non-recursive
    scored = _score_cases(cases)
    analysis = analyze_case_results(scored)
    return {
        "label": "base",
        "case_count": len(cases),
        "scored_count": len(scored),
        "scored_results": scored,
        "analysis": analysis,
    }


def run_variation_group(case_dir: Path = _CASE_DIR) -> dict:
    """Generate and evaluate all variations of base cases."""
    base_cases = load_all_cases(case_dir)
    all_variations: list[dict] = []
    for case in base_cases:
        all_variations.extend(generate_case_variations(case))
    scored = _score_cases(all_variations)
    analysis = analyze_case_results(scored)
    return {
        "label": "variations",
        "case_count": len(all_variations),
        "scored_count": len(scored),
        "scored_results": scored,
        "analysis": analysis,
    }


def run_adversarial_group(case_dir: Path = _CASE_DIR) -> dict:
    """Generate and evaluate all adversarial variants of base cases."""
    base_cases = load_all_cases(case_dir)
    all_adversarial: list[dict] = []
    for case in base_cases:
        all_adversarial.extend(generate_adversarial_cases(case))
    scored = _score_cases(all_adversarial)
    analysis = analyze_case_results(scored)
    return {
        "label": "adversarial",
        "case_count": len(all_adversarial),
        "scored_count": len(scored),
        "scored_results": scored,
        "analysis": analysis,
    }


def run_synthea_group(sample_path: Path = _SYNTHEA_PATH) -> dict:
    """Import and evaluate Synthea sample patients."""
    if not sample_path.exists():
        return {
            "label": "synthea",
            "case_count": 0,
            "scored_count": 0,
            "scored_results": [],
            "analysis": analyze_case_results([]),
        }

    patients = load_synthea_patients(sample_path)
    cases = convert_patients_to_cases(patients)
    scored = _score_cases(cases)
    analysis = analyze_case_results(scored)
    return {
        "label": "synthea",
        "case_count": len(cases),
        "scored_count": len(scored),
        "scored_results": scored,
        "analysis": analysis,
    }


# ── analysis helpers ────────────────────────────────────────────────


def summarize_group_analysis(group: dict) -> dict:
    """Extract a compact summary from a group result."""
    overall = group["analysis"]["overall"]
    worst = group["analysis"]["worst_cases"]
    strategies = group["analysis"]["strategy_breakdown"]

    worst_case_id = worst[0]["case_id"] if worst else "(none)"
    worst_score = worst[0]["score"] if worst else 0.0

    most_damaging = ""
    if strategies:
        lowest = min(strategies, key=lambda s: s["avg_score"])
        most_damaging = f"{lowest['strategy']} (avg={lowest['avg_score']:.2f})"

    return {
        "label": group["label"],
        "count": overall["total_cases"],
        "scored": overall["scored_cases"],
        "avg_hypothesis_hit_rate": overall["avg_hypothesis_hit_rate"],
        "avg_red_flag_hit_rate": overall["avg_red_flag_hit_rate"],
        "avg_key_finding_hit_rate": overall["avg_key_finding_hit_rate"],
        "avg_score": overall["avg_score"],
        "worst_case": worst_case_id,
        "worst_score": worst_score,
        "most_damaging_strategy": most_damaging,
    }


def build_global_analysis(groups: list[dict]) -> dict:
    """Combine all scored results into one global analysis."""
    all_scored: list[dict] = []
    for g in groups:
        all_scored.extend(g["scored_results"])
    return analyze_case_results(all_scored)


# ── rendering ───────────────────────────────────────────────────────


def render_dashboard_report(groups: list[dict], global_analysis: dict) -> str:
    """Render the full dashboard as a string."""
    lines: list[str] = []
    lines.append("=== EVALUATION DASHBOARD ===")
    lines.append("")

    for group in groups:
        summary = summarize_group_analysis(group)
        label = summary["label"].upper()
        lines.append(f"--- {label} ---")
        lines.append(f"  count:                    {summary['count']}")
        lines.append(f"  scored:                   {summary['scored']}")
        lines.append(f"  avg_hypothesis_hit_rate:  {summary['avg_hypothesis_hit_rate']:.2f}")
        lines.append(f"  avg_red_flag_hit_rate:    {summary['avg_red_flag_hit_rate']:.2f}")
        lines.append(f"  avg_key_finding_hit_rate: {summary['avg_key_finding_hit_rate']:.2f}")
        lines.append(f"  avg_score:                {summary['avg_score']:.2f}")
        lines.append(f"  worst_case:               {summary['worst_case']} ({summary['worst_score']:.2f})")

        if summary["most_damaging_strategy"]:
            lines.append(f"  most_damaging_strategy:   {summary['most_damaging_strategy']}")

        lines.append("")

    # Global summary.
    overall = global_analysis["overall"]
    lines.append("--- GLOBAL SUMMARY ---")
    lines.append(f"  total_cases:                      {overall['total_cases']}")
    lines.append(f"  scored_cases:                     {overall['scored_cases']}")
    lines.append(f"  overall_avg_hypothesis_hit_rate:  {overall['avg_hypothesis_hit_rate']:.2f}")
    lines.append(f"  overall_avg_red_flag_hit_rate:    {overall['avg_red_flag_hit_rate']:.2f}")
    lines.append(f"  overall_avg_key_finding_hit_rate: {overall['avg_key_finding_hit_rate']:.2f}")
    lines.append(f"  overall_avg_score:                {overall['avg_score']:.2f}")
    lines.append("")

    hyp_failures = global_analysis["hypothesis_failures"]
    if hyp_failures:
        lines.append("  most_frequently_missing_hypotheses:")
        for f in hyp_failures[:5]:
            lines.append(f"    {f['item']:<30} ({f['count']}x)")
        lines.append("")

    rf_failures = global_analysis["red_flag_failures"]
    if rf_failures:
        lines.append("  most_frequently_missed_red_flags:")
        for f in rf_failures[:5]:
            lines.append(f"    {f['item']:<30} ({f['count']}x)")
        lines.append("")

    kf_failures = global_analysis["key_finding_failures"]
    if kf_failures:
        lines.append("  most_frequently_missed_key_findings:")
        for f in kf_failures[:5]:
            lines.append(f"    {f['item']:<30} ({f['count']}x)")
        lines.append("")

    dist = global_analysis["score_distribution"]
    if dist:
        lines.append("  score_distribution:")
        for d in dist:
            bar = "#" * d["count"]
            lines.append(f"    {d['range']}  {bar} ({d['count']})")
        lines.append("")

    return "\n".join(lines)


# ── main ────────────────────────────────────────────────────────────


def run_dashboard(
    case_dir: Path = _CASE_DIR,
    synthea_path: Path = _SYNTHEA_PATH,
) -> dict:
    """Run the full evaluation dashboard.

    Returns:
        Dict with ``groups`` (list of group results) and
        ``global_analysis``.
    """
    groups = [
        run_base_case_group(case_dir),
        run_variation_group(case_dir),
        run_adversarial_group(case_dir),
        run_synthea_group(synthea_path),
    ]
    global_analysis = build_global_analysis(groups)
    return {
        "groups": groups,
        "global_analysis": global_analysis,
    }


def main() -> None:
    dashboard = run_dashboard()
    report = render_dashboard_report(
        dashboard["groups"],
        dashboard["global_analysis"],
    )
    print(report)


if __name__ == "__main__":
    main()
