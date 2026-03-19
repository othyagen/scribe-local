#!/usr/bin/env python3
"""Batch case scoring runner.

Loads all YAML cases from ``resources/cases/``, executes each through
the clinical session pipeline, scores against ground truth, and prints
a concise evaluation report.

Run from the project root::

    python -m scripts.run_case_scoring
"""

from __future__ import annotations

import sys
from pathlib import Path

from app.case_system import load_all_cases, validate_case
from app.case_scoring import (
    score_case_run,
    score_case_script_run,
    summarize_score,
)


_CASE_DIR = Path(__file__).resolve().parent.parent / "resources" / "cases"


def run_scoring() -> None:
    """Run all cases, score, and print a report."""
    cases = load_all_cases(_CASE_DIR)

    if not cases:
        print(f"No cases found in {_CASE_DIR}")
        sys.exit(1)

    print(f"Case Scoring: {len(cases)} case(s) from {_CASE_DIR}")
    print("=" * 70)
    print()

    scored_summaries: list[dict] = []

    for case in cases:
        case_id = case.get("case_id", "(no id)")
        validation = validate_case(case)

        if not validation["valid"]:
            errors = "; ".join(validation["errors"])
            print(f"  {case_id:<25} INVALID  errors: {errors}")
            continue

        has_script = bool(case.get("answer_script"))
        if has_script:
            scored = score_case_script_run(case)
        else:
            scored = score_case_run(case)

        score = scored["score"]
        summary = summarize_score(score)
        scored_summaries.append(summary)

        mode = "scripted" if has_script else "basic"
        gt_str = "yes" if summary["has_ground_truth"] else "no"

        print(f"  {case_id:<25} ({mode})  ground_truth: {gt_str}")

        if summary["has_ground_truth"]:
            hyp = score["hypotheses"]
            rf = score["red_flags"]
            kf = score["key_findings"]

            print(f"    hypotheses:   {hyp['matched_count']}/{hyp['expected_count']}"
                  f"  hit_rate={summary['hypothesis_hit_rate']:.2f}"
                  f"  top_expected={summary['top_hypothesis_expected']}")
            if hyp["missing"]:
                print(f"      missing: {', '.join(hyp['missing'])}")

            print(f"    red_flags:    {rf['matched_count']}/{rf['expected_count']}"
                  f"  hit_rate={summary['red_flag_hit_rate']:.2f}")
            if rf["missing"]:
                print(f"      missing: {', '.join(rf['missing'])}")

            print(f"    key_findings: {kf['matched_count']}/{kf['expected_count']}"
                  f"  hit_rate={summary['key_finding_hit_rate']:.2f}")
            if kf["missing"]:
                print(f"      missing: {', '.join(kf['missing'])}")
        else:
            print("    (no ground truth to score)")

        print()

    # Summary averages.
    with_gt = [s for s in scored_summaries if s["has_ground_truth"]]

    print("=" * 70)
    print("SUMMARY")
    print(f"  cases scored:     {len(scored_summaries)}")
    print(f"  with ground truth: {len(with_gt)}")

    if with_gt:
        avg_hyp = sum(s["hypothesis_hit_rate"] for s in with_gt) / len(with_gt)
        avg_rf = sum(s["red_flag_hit_rate"] for s in with_gt) / len(with_gt)
        avg_kf = sum(s["key_finding_hit_rate"] for s in with_gt) / len(with_gt)
        top_count = sum(1 for s in with_gt if s["top_hypothesis_expected"])

        print(f"  avg hypothesis_hit_rate:  {avg_hyp:.2f}")
        print(f"  avg red_flag_hit_rate:    {avg_rf:.2f}")
        print(f"  avg key_finding_hit_rate: {avg_kf:.2f}")
        print(f"  top_hypothesis_expected:  {top_count}/{len(with_gt)}")
    print()


if __name__ == "__main__":
    run_scoring()
