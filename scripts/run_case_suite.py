#!/usr/bin/env python3
"""Batch clinical case runner.

Loads all YAML cases from ``resources/cases/``, runs each through the
clinical session pipeline, and prints a concise report.

Run from the project root::

    python -m scripts.run_case_suite
"""

from __future__ import annotations

import sys
from pathlib import Path

from app.case_system import (
    load_all_cases,
    validate_case,
    run_case,
    run_case_script,
    extract_top_hypotheses,
    compare_result_to_ground_truth,
)


_CASE_DIR = Path(__file__).resolve().parent.parent / "resources" / "cases"


def _format_hypothesis(result: dict) -> str:
    hyps = extract_top_hypotheses(result, n=1)
    if hyps:
        return hyps[0].get("title", "?")
    return "(none)"


def _format_gt_check(result: dict) -> str:
    gt = result.get("ground_truth", {})
    if not gt.get("expected_hypotheses") and not gt.get("red_flags") and not gt.get("key_findings"):
        return "no ground truth"

    comparison = compare_result_to_ground_truth(result)
    parts: list[str] = []

    hyp_matches = comparison.get("hypothesis_matches", [])
    if hyp_matches:
        found = sum(1 for m in hyp_matches if m["found"])
        parts.append(f"hyp: {found}/{len(hyp_matches)}")

    flag_matches = comparison.get("red_flag_matches", [])
    if flag_matches:
        found = sum(1 for m in flag_matches if m["found"])
        parts.append(f"flags: {found}/{len(flag_matches)}")

    finding_matches = comparison.get("finding_matches", [])
    if finding_matches:
        found = sum(1 for m in finding_matches if m["found"])
        parts.append(f"findings: {found}/{len(finding_matches)}")

    return ", ".join(parts) if parts else "no ground truth"


def run_suite() -> None:
    """Run all cases and print a report."""
    cases = load_all_cases(_CASE_DIR)

    if not cases:
        print(f"No cases found in {_CASE_DIR}")
        sys.exit(1)

    print(f"Case Suite: {len(cases)} case(s) from {_CASE_DIR}")
    print("=" * 70)
    print()

    valid_count = 0
    invalid_count = 0
    hyp_counts: list[int] = []
    question_counts: list[int] = []
    red_flag_counts: list[int] = []

    for case in cases:
        case_id = case.get("case_id", "(no id)")
        validation = validate_case(case)

        if not validation["valid"]:
            invalid_count += 1
            errors = "; ".join(validation["errors"])
            print(f"  {case_id:<25} INVALID  errors: {errors}")
            continue

        valid_count += 1

        # Use scripted run if answer_script is present.
        has_script = bool(case.get("answer_script"))
        result = run_case_script(case) if has_script else run_case(case)

        metrics = result.get("metrics", {})
        hm = metrics.get("hypothesis_metrics", {})
        rm = metrics.get("risk_metrics", {})
        im = metrics.get("interaction_metrics", {})
        um = metrics.get("update_metrics", {})

        hyp_count = hm.get("hypothesis_count", 0)
        rf_count = rm.get("red_flag_count", 0)
        q_count = im.get("question_count", 0)
        pending = um.get("pending_count")

        hyp_counts.append(hyp_count)
        question_counts.append(q_count)
        red_flag_counts.append(rf_count)

        top_hyp = _format_hypothesis(result)
        gt_check = _format_gt_check(result)

        mode = "scripted" if has_script else "basic"
        pending_str = str(pending) if pending is not None else "n/a"

        print(f"  {case_id:<25} OK ({mode})")
        print(f"    top hypothesis:  {top_hyp}")
        print(f"    hypotheses: {hyp_count}  red_flags: {rf_count}  "
              f"questions: {q_count}  pending: {pending_str}")
        print(f"    ground truth:    {gt_check}")
        print()

    # Summary.
    print("=" * 70)
    print("SUMMARY")
    print(f"  cases run:    {valid_count + invalid_count}")
    print(f"  valid:        {valid_count}")
    print(f"  invalid:      {invalid_count}")

    if hyp_counts:
        print(f"  avg hypotheses:   {sum(hyp_counts) / len(hyp_counts):.1f}")
        print(f"  avg questions:    {sum(question_counts) / len(question_counts):.1f}")
        print(f"  avg red flags:    {sum(red_flag_counts) / len(red_flag_counts):.1f}")
    print()


if __name__ == "__main__":
    run_suite()
