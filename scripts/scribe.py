#!/usr/bin/env python3
"""SCRIBE CLI — unified case browsing, validation, and execution.

Thin wrapper around existing case system functions.  Does not modify
any execution function signatures or core logic.

Usage::

    python -m scripts.scribe cases list [--tag X] [--origin X] ...
    python -m scripts.scribe cases show <case_ref>
    python -m scripts.scribe cases validate <case_ref>
    python -m scripts.scribe cases create
    python -m scripts.scribe run <case_ref>
    python -m scripts.scribe compare <case_ref> [--provider edge] [--output-dir DIR]
    python -m scripts.scribe dashboard [--case-dir DIR] [--synthea-path PATH]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CASE_DIRS = [_PROJECT_ROOT / "resources" / "cases"]
_DEFAULT_SYNTHEA = _PROJECT_ROOT / "resources" / "synthea_sample.json"

_CASE_TEMPLATE = """\
case_id: my_case_01
title: Short descriptive title
description: >
  Longer description of the clinical scenario.

segments:
  - seg_id: seg_0001
    t0: 0.0
    t1: 3.0
    speaker_id: spk_0
    normalized_text: "Patient presents with symptom."

config:
  mode: assist
  update_strategy: manual
  show_questions: true

ground_truth:
  expected_hypotheses: []
  red_flags: []
  key_findings: []

meta:
  tags: []
  difficulty: moderate
  source: synthetic

# Optional — add provenance for traceability:
# provenance:
#   origin: synthetic
#   created: "2026-03-22"

# Optional — add classification for registry filtering:
# classification:
#   organ_systems: [respiratory]
#   presenting_complaints: [cough]
#   diagnosis_targets:
#     icd10:
#       - code: "J18.9"
#         display: "Pneumonia, unspecified"

# Optional — add patient demographics:
# patient:
#   age: 50
#   sex: male
"""


# ── case resolution ────────────────────────────────────────────────


def _resolve(args: argparse.Namespace) -> Path:
    """Resolve case_ref from CLI args."""
    from app.case_registry import resolve_case

    case_dirs = [Path(d) for d in args.case_dirs] if hasattr(args, "case_dirs") and args.case_dirs else _DEFAULT_CASE_DIRS
    return resolve_case(args.case_ref, case_dirs)


def _get_case_dirs(args: argparse.Namespace) -> list[Path]:
    if hasattr(args, "case_dirs") and args.case_dirs:
        return [Path(d) for d in args.case_dirs]
    return list(_DEFAULT_CASE_DIRS)


# ── cases subcommands ──────────────────────────────────────────────


def cmd_cases_list(args: argparse.Namespace) -> None:
    """List cases with optional filters."""
    from app.case_registry import build_registry, filter_registry

    case_dirs = _get_case_dirs(args)
    entries = build_registry(case_dirs)
    entries = filter_registry(
        entries,
        organ_system=args.organ_system,
        complaint=args.complaint,
        tag=args.tag,
        origin=args.origin,
        difficulty=args.difficulty,
        icd=args.icd,
        icpc=args.icpc,
        snomed=args.snomed,
    )

    if not entries:
        print("No cases found.")
        return

    # Tabular output.
    header = f"{'CASE_ID':<25} {'TITLE':<40} {'ORIGIN':<12} {'DIFF':<12} {'SEGS':>4}"
    print(header)
    print("-" * len(header))
    for e in entries:
        title = e.title[:38] + ".." if len(e.title) > 40 else e.title
        print(f"{e.case_id:<25} {title:<40} {e.origin:<12} {e.difficulty:<12} {e.segment_count:>4}")
    print(f"\n{len(entries)} case(s) found.")


def cmd_cases_show(args: argparse.Namespace) -> None:
    """Show details of a single case."""
    from app.case_system import load_case

    path = _resolve(args)
    case = load_case(path)

    print(f"Case ID:     {case.get('case_id', '')}")
    print(f"Title:       {case.get('title', '')}")
    print(f"Description: {case.get('description', '').strip()}")
    print(f"Segments:    {len(case.get('segments', []))}")
    print(f"File:        {path}")

    prov = case.get("provenance")
    if prov:
        print(f"\nProvenance:  origin={prov.get('origin', '')}, created={prov.get('created', '')}")

    meta = case.get("meta")
    if meta:
        print(f"Tags:        {meta.get('tags', [])}")
        print(f"Difficulty:  {meta.get('difficulty', '')}")

    classification = case.get("classification")
    if classification:
        print(f"\nClassification:")
        for k, v in classification.items():
            print(f"  {k}: {v}")

    patient = case.get("patient")
    if patient:
        print(f"\nPatient:     {patient}")

    gt = case.get("ground_truth")
    if gt:
        print(f"\nGround truth:")
        for k, v in gt.items():
            print(f"  {k}: {v}")


def cmd_cases_validate(args: argparse.Namespace) -> None:
    """Validate a case (composed: core + extended schema)."""
    from app.case_system import load_case, validate_case
    from app.case_schema import validate_extended_schema

    path = _resolve(args)
    case = load_case(path)

    # Core validation (includes provenance).
    core = validate_case(case)
    # Extended validation (classification + patient).
    ext = validate_extended_schema(case)

    # Compose results.
    errors = core["errors"] + ext["errors"]
    warnings = core["warnings"] + ext["warnings"]
    valid = len(errors) == 0

    print(f"Validating: {case.get('case_id', '')} ({path})")
    print(f"Result:     {'VALID' if valid else 'INVALID'}")

    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")

    if warnings:
        print(f"\nWarnings ({len(warnings)}):")
        for w in warnings:
            print(f"  - {w}")

    if not errors and not warnings:
        print("  No issues found.")

    sys.exit(0 if valid else 1)


def cmd_cases_create(args: argparse.Namespace) -> None:
    """Print a YAML case template to stdout."""
    print(_CASE_TEMPLATE)


def cmd_cases_slice(args: argparse.Namespace) -> None:
    """Score all cases and slice results by a metadata field."""
    from app.case_registry import build_registry
    from app.case_scoring import score_case_run
    from app.case_system import load_case
    from app.evaluation_slicing import slice_evaluation

    case_dirs = _get_case_dirs(args)
    entries = build_registry(case_dirs)

    if not entries:
        print("No cases found.")
        return

    results = []
    for entry in entries:
        case = load_case(entry.path)
        scored = score_case_run(case)
        results.append({"case": case, "score": scored["score"]})

    sliced = slice_evaluation(results, args.by)

    if not sliced:
        print("No groups found.")
        return

    # Sort by score ascending (lowest first).
    sorted_groups = sorted(sliced.items(), key=lambda kv: kv[1])

    header = f"{'GROUP':<30} {'HIT_RATE':>10}"
    print(f"Slicing by: {args.by}\n")
    print(header)
    print("-" * len(header))
    for group, rate in sorted_groups:
        print(f"{group:<30} {rate:>10.2%}")
    print(f"\n{len(sorted_groups)} group(s).")


# ── run subcommand ─────────────────────────────────────────────────


def cmd_run(args: argparse.Namespace) -> None:
    """Run a case through the clinical pipeline."""
    from app.case_system import load_case, run_case, run_case_script

    path = _resolve(args)
    case = load_case(path)

    if case.get("answer_script"):
        result = run_case_script(case)
    else:
        result = run_case(case)

    # Print summary.
    print(f"Case:       {result.get('case_id', '')}")
    v = result.get("validation", {})
    print(f"Valid:      {v.get('valid', False)}")

    metrics = result.get("metrics", {})
    if metrics:
        print(f"Metrics:    {json.dumps(metrics, indent=2)}")

    state = result.get("session", {}).get("clinical_state", {})
    hyps = state.get("hypotheses", [])
    if hyps:
        print(f"\nHypotheses ({len(hyps)}):")
        for h in hyps[:5]:
            print(f"  - {h.get('title', '')}")


# ── compare subcommand ─────────────────────────────────────────────


def cmd_compare(args: argparse.Namespace) -> None:
    """Run text vs TTS comparison on a case."""
    from app.case_system import load_case
    from app.case_compare import compare_case_modes

    path = _resolve(args)
    case = load_case(path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = compare_case_modes(
        case, output_dir,
        provider=args.provider,
    )

    comp = result.get("comparison", {})
    print(f"Case:       {case.get('case_id', '')}")
    print(f"Provider:   {args.provider}")
    print(f"\nComparison:")
    print(json.dumps(comp, indent=2, default=str))


# ── dashboard subcommand ───────────────────────────────────────────


def cmd_dashboard(args: argparse.Namespace) -> None:
    """Run the full evaluation dashboard."""
    from scripts.run_evaluation_dashboard import (
        run_dashboard,
        render_dashboard_report,
    )

    case_dir = Path(args.case_dir) if args.case_dir else _DEFAULT_CASE_DIRS[0]
    synthea_path = Path(args.synthea_path) if args.synthea_path else _DEFAULT_SYNTHEA

    dashboard = run_dashboard(case_dir=case_dir, synthea_path=synthea_path)
    report = render_dashboard_report(
        dashboard["groups"],
        dashboard["global_analysis"],
        dashboard.get("mismatch_report"),
        dashboard.get("encounter_data"),
        dashboard.get("compare_data"),
    )
    print(report)


# ── argument parsing ───────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scribe",
        description="SCRIBE CLI — case browsing, validation, and execution.",
    )
    parser.add_argument(
        "--case-dir", dest="case_dirs", action="append", default=None,
        help="Case directory to scan (repeatable; default: resources/cases)",
    )
    subparsers = parser.add_subparsers(dest="command")

    # ── cases ──
    cases_parser = subparsers.add_parser("cases", help="Browse and manage cases")
    cases_sub = cases_parser.add_subparsers(dest="cases_action")

    # cases list
    list_p = cases_sub.add_parser("list", help="List cases with optional filters")
    list_p.add_argument("--organ-system", dest="organ_system", default=None)
    list_p.add_argument("--complaint", default=None)
    list_p.add_argument("--tag", default=None)
    list_p.add_argument("--origin", default=None)
    list_p.add_argument("--difficulty", default=None)
    list_p.add_argument("--icd", default=None)
    list_p.add_argument("--icpc", default=None)
    list_p.add_argument("--snomed", default=None)
    list_p.set_defaults(func=cmd_cases_list)

    # cases show
    show_p = cases_sub.add_parser("show", help="Show case details")
    show_p.add_argument("case_ref", help="Case ID or path to YAML file")
    show_p.set_defaults(func=cmd_cases_show)

    # cases validate
    val_p = cases_sub.add_parser("validate", help="Validate a case")
    val_p.add_argument("case_ref", help="Case ID or path to YAML file")
    val_p.set_defaults(func=cmd_cases_validate)

    # cases create
    create_p = cases_sub.add_parser("create", help="Print YAML case template")
    create_p.set_defaults(func=cmd_cases_create)

    # cases slice
    slice_p = cases_sub.add_parser("slice", help="Slice evaluation results by metadata field")
    slice_p.add_argument(
        "--by", required=True,
        choices=["organ_system", "presenting_complaint", "difficulty", "origin", "tag"],
        help="Metadata field to group by",
    )
    slice_p.set_defaults(func=cmd_cases_slice)

    # ── run ──
    run_p = subparsers.add_parser("run", help="Run a case through the pipeline")
    run_p.add_argument("case_ref", help="Case ID or path to YAML file")
    run_p.set_defaults(func=cmd_run)

    # ── compare ──
    cmp_p = subparsers.add_parser("compare", help="Text vs TTS comparison")
    cmp_p.add_argument("case_ref", help="Case ID or path to YAML file")
    cmp_p.add_argument("--provider", default="edge", help="TTS provider")
    cmp_p.add_argument("--output-dir", dest="output_dir", default="output/tts",
                        help="Directory for TTS audio output")
    cmp_p.set_defaults(func=cmd_compare)

    # ── dashboard ──
    dash_p = subparsers.add_parser("dashboard", help="Run evaluation dashboard")
    dash_p.add_argument("--case-dir", dest="case_dir", default=None)
    dash_p.add_argument("--synthea-path", dest="synthea_path", default=None)
    dash_p.set_defaults(func=cmd_dashboard)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)

    try:
        args.func(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
