#!/usr/bin/env python3
"""Synthea suite runner — end-to-end evaluation of imported Synthea cases.

Loads Synthea-style patients, converts to SCRIBE cases, executes through
the case system, scores against ground truth, and produces a structured
analysis report.

Run from the project root::

    python -m scripts.run_synthea_suite
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

from app.synthea_import import load_synthea_patients, convert_patients_to_cases
from app.case_system import validate_case, run_case, run_case_script
from app.case_scoring import score_result_against_ground_truth, summarize_score
from app.case_analysis import analyze_case_results


_SAMPLE_PATH = Path(__file__).resolve().parent.parent / "resources" / "synthea_sample.json"
_GENERATED_DIR = Path(__file__).resolve().parent.parent / "resources" / "cases" / "generated"


# ── export helper ───────────────────────────────────────────────────


def export_cases_to_yaml(cases: list[dict], output_dir: Path) -> list[Path]:
    """Export converted cases to YAML files.

    Args:
        cases: list of SCRIBE case dicts.
        output_dir: directory to write YAML files into.

    Returns:
        List of written file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for case in cases:
        case_id = case.get("case_id", "unknown")
        filename = f"{case_id}.yaml"
        path = output_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(case, f, default_flow_style=False, sort_keys=False)
        paths.append(path)
    return paths


# ── runner ──────────────────────────────────────────────────────────


def run_synthea_suite(
    sample_path: Path = _SAMPLE_PATH,
    export_dir: Path | None = _GENERATED_DIR,
) -> dict:
    """Run the full Synthea evaluation suite.

    Args:
        sample_path: path to Synthea sample JSON.
        export_dir: optional directory to export YAML cases.
            Set to ``None`` to skip export.

    Returns:
        Dict with ``patients``, ``cases``, ``scored_results``,
        and ``analysis``.
    """
    # Load and convert.
    patients = load_synthea_patients(sample_path)
    cases = convert_patients_to_cases(patients)

    # Validate.
    valid_cases: list[dict] = []
    invalid_cases: list[dict] = []
    for case in cases:
        validation = validate_case(case)
        if validation["valid"]:
            valid_cases.append(case)
        else:
            invalid_cases.append(case)

    # Export if requested.
    if export_dir is not None and valid_cases:
        export_cases_to_yaml(valid_cases, export_dir)

    # Execute and score.
    scored_results: list[dict] = []
    for case in valid_cases:
        has_script = bool(case.get("answer_script"))
        result = run_case_script(case) if has_script else run_case(case)

        if not result.get("validation", {}).get("valid", False):
            continue

        score = score_result_against_ground_truth(result)
        scored_results.append({
            "case_id": case.get("case_id", ""),
            "title": case.get("title", ""),
            "score": score,
            "result_bundle": result,
        })

    # Analyze.
    analysis = analyze_case_results(scored_results)

    return {
        "patient_count": len(patients),
        "case_count": len(cases),
        "valid_case_count": len(valid_cases),
        "invalid_case_count": len(invalid_cases),
        "cases": cases,
        "scored_results": scored_results,
        "analysis": analysis,
    }


# ── report printing ────────────────────────────────────────────────


def _extract_top_hypothesis(result: dict) -> str:
    state = result.get("result_bundle", {}).get("session", {}).get("clinical_state", {})
    hyps = state.get("hypotheses", [])
    if hyps:
        return hyps[0].get("title", "(none)")
    return "(none)"


def print_report(suite: dict) -> None:
    """Print a structured evaluation report."""
    print("=== SYNTHEA IMPORT SUMMARY ===")
    print(f"  patients loaded:   {suite['patient_count']}")
    print(f"  cases generated:   {suite['case_count']}")
    print(f"  valid cases:       {suite['valid_case_count']}")
    print(f"  invalid cases:     {suite['invalid_case_count']}")
    print()

    scored = suite["scored_results"]
    if scored:
        print("=== PER-CASE RESULTS ===")
        for sr in scored:
            case_id = sr["case_id"]
            title = sr.get("title", "")
            summary = summarize_score(sr["score"])
            top_hyp = _extract_top_hypothesis(sr)

            print(f"  {case_id}")
            if title:
                print(f"    title:     {title}")
            print(f"    top hyp:   {top_hyp}")
            print(f"    hyp rate:  {summary['hypothesis_hit_rate']:.2f}")
            print(f"    rf rate:   {summary['red_flag_hit_rate']:.2f}")
            print(f"    kf rate:   {summary['key_finding_hit_rate']:.2f}")
            print()

    analysis = suite["analysis"]
    overall = analysis["overall"]

    print("=== OVERALL ANALYSIS ===")
    print(f"  total cases:          {overall['total_cases']}")
    print(f"  scored cases:         {overall['scored_cases']}")
    print(f"  avg hyp hit rate:     {overall['avg_hypothesis_hit_rate']:.2f}")
    print(f"  avg rf hit rate:      {overall['avg_red_flag_hit_rate']:.2f}")
    print(f"  avg kf hit rate:      {overall['avg_key_finding_hit_rate']:.2f}")
    print()

    worst = analysis["worst_cases"]
    if worst:
        print("  Worst cases:")
        for w in worst[:3]:
            print(f"    {w['case_id']:<30} score={w['score']:.2f}")
            if w["missing_hypotheses"]:
                print(f"      missing hyp: {', '.join(w['missing_hypotheses'])}")
            if w["missing_key_findings"]:
                print(f"      missing kf:  {', '.join(w['missing_key_findings'])}")
        print()

    hyp_failures = analysis["hypothesis_failures"]
    if hyp_failures:
        print("  Most missed hypotheses:")
        for f in hyp_failures[:5]:
            print(f"    {f['item']:<30} ({f['count']}x)")
        print()

    rf_failures = analysis["red_flag_failures"]
    if rf_failures:
        print("  Most missed red flags:")
        for f in rf_failures[:5]:
            print(f"    {f['item']:<30} ({f['count']}x)")
        print()

    kf_failures = analysis["key_finding_failures"]
    if kf_failures:
        print("  Most missed key findings:")
        for f in kf_failures[:5]:
            print(f"    {f['item']:<30} ({f['count']}x)")
        print()


def main() -> None:
    if not _SAMPLE_PATH.exists():
        print(f"Sample file not found: {_SAMPLE_PATH}")
        sys.exit(1)

    suite = run_synthea_suite()
    print_report(suite)


if __name__ == "__main__":
    main()
