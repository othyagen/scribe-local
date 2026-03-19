#!/usr/bin/env python3
"""Case analysis report — failure analysis across all case variants.

Loads seed cases, generates variations and adversarial variants,
runs scoring, and prints a structured evaluation report.

Run from the project root::

    python -m scripts.run_case_analysis
"""

from __future__ import annotations

import sys
from pathlib import Path

from app.case_system import load_all_cases, validate_case, run_case, run_case_script
from app.case_variations import generate_case_variations
from app.case_adversarial import generate_adversarial_cases
from app.case_scoring import score_result_against_ground_truth
from app.case_analysis import analyze_case_results


_CASE_DIR = Path(__file__).resolve().parent.parent / "resources" / "cases"


def _run_and_score(case: dict) -> dict:
    """Run a case and return a result with score and optional adversarial metadata."""
    has_script = bool(case.get("answer_script"))
    result = run_case_script(case) if has_script else run_case(case)

    if not result.get("validation", {}).get("valid", False):
        return None

    score = score_result_against_ground_truth(result)
    entry = {
        "case_id": case.get("case_id", ""),
        "score": score,
    }
    if "adversarial" in case:
        entry["adversarial"] = case["adversarial"]
    return entry


def run_analysis() -> None:
    """Run full analysis and print report."""
    base_cases = load_all_cases(_CASE_DIR)

    if not base_cases:
        print(f"No cases found in {_CASE_DIR}")
        sys.exit(1)

    # Collect all cases: base + variations + adversarial.
    all_cases: list[dict] = []
    all_cases.extend(base_cases)

    for case in base_cases:
        all_cases.extend(generate_case_variations(case))
        all_cases.extend(generate_adversarial_cases(case))

    print(f"Running analysis on {len(all_cases)} total cases")
    print(f"  ({len(base_cases)} base + variations + adversarial)")
    print("=" * 60)
    print()

    # Run and score all.
    case_results: list[dict] = []
    for case in all_cases:
        entry = _run_and_score(case)
        if entry is not None:
            case_results.append(entry)

    analysis = analyze_case_results(case_results)

    # Print report.
    overall = analysis["overall"]
    print("=== OVERALL ===")
    print(f"  Cases:          {overall['total_cases']}")
    print(f"  Scored:         {overall['scored_cases']}")
    print(f"  Avg score:      {overall['avg_score']:.2f}")
    print(f"  Avg hyp rate:   {overall['avg_hypothesis_hit_rate']:.2f}")
    print(f"  Avg rf rate:    {overall['avg_red_flag_hit_rate']:.2f}")
    print(f"  Avg kf rate:    {overall['avg_key_finding_hit_rate']:.2f}")
    print()

    worst = analysis["worst_cases"]
    if worst:
        print("=== WORST CASES ===")
        for w in worst:
            print(f"  {w['case_id']:<40} score={w['score']:.2f}")
            if w["missing_hypotheses"]:
                print(f"    missing hyp:      {', '.join(w['missing_hypotheses'])}")
            if w["missing_red_flags"]:
                print(f"    missing rf:       {', '.join(w['missing_red_flags'])}")
            if w["missing_key_findings"]:
                print(f"    missing findings: {', '.join(w['missing_key_findings'])}")
        print()

    breakdown = analysis["strategy_breakdown"]
    if breakdown:
        print("=== STRATEGY BREAKDOWN ===")
        for s in breakdown:
            print(f"  {s['strategy']:<30} avg={s['avg_score']:.2f}  n={s['count']}")
        print()

    hyp_failures = analysis["hypothesis_failures"]
    if hyp_failures:
        print("=== MISSING HYPOTHESES ===")
        for f in hyp_failures:
            print(f"  {f['item']:<30} ({f['count']}x)")
        print()

    rf_failures = analysis["red_flag_failures"]
    if rf_failures:
        print("=== MISSING RED FLAGS ===")
        for f in rf_failures:
            print(f"  {f['item']:<30} ({f['count']}x)")
        print()

    kf_failures = analysis["key_finding_failures"]
    if kf_failures:
        print("=== MISSING KEY FINDINGS ===")
        for f in kf_failures:
            print(f"  {f['item']:<30} ({f['count']}x)")
        print()

    dist = analysis["score_distribution"]
    if dist:
        print("=== SCORE DISTRIBUTION ===")
        for d in dist:
            bar = "#" * d["count"]
            print(f"  {d['range']}  {bar} ({d['count']})")
        print()


if __name__ == "__main__":
    run_analysis()
