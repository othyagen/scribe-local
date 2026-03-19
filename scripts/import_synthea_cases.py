#!/usr/bin/env python3
"""Synthea case import preview.

Loads sample Synthea patients, converts to SCRIBE cases, and prints
a preview of each generated case.

Run from the project root::

    python -m scripts.import_synthea_cases
"""

from __future__ import annotations

import sys
from pathlib import Path

from app.synthea_import import load_synthea_patients, convert_patients_to_cases
from app.case_system import validate_case


_SAMPLE_PATH = Path(__file__).resolve().parent.parent / "resources" / "synthea_sample.json"


def run_import() -> None:
    """Load, convert, and preview Synthea cases."""
    if not _SAMPLE_PATH.exists():
        print(f"Sample file not found: {_SAMPLE_PATH}")
        sys.exit(1)

    patients = load_synthea_patients(_SAMPLE_PATH)
    cases = convert_patients_to_cases(patients)

    print(f"Loaded {len(patients)} patient(s) from {_SAMPLE_PATH.name}")
    print(f"Generated {len(cases)} case(s)")
    print("=" * 60)
    print()

    for case in cases:
        case_id = case.get("case_id", "?")
        title = case.get("title", "?")
        validation = validate_case(case)
        status = "valid" if validation["valid"] else "INVALID"

        print(f"  {case_id:<25} {status}")
        print(f"    title: {title}")

        segments = case.get("segments", [])
        print(f"    segments: {len(segments)}")
        for seg in segments:
            text = seg.get("normalized_text", "")
            print(f"      - {text}")

        gt = case.get("ground_truth", {})
        hyps = gt.get("expected_hypotheses", [])
        findings = gt.get("key_findings", [])
        flags = gt.get("red_flags", [])

        print(f"    hypotheses: {', '.join(hyps) if hyps else '(none)'}")
        print(f"    findings:   {', '.join(findings) if findings else '(none)'}")
        print(f"    red flags:  {', '.join(flags) if flags else '(none)'}")

        meta = case.get("meta", {})
        tags = meta.get("tags", [])
        print(f"    tags:       {', '.join(tags) if tags else '(none)'}")
        print()

    # Summary.
    valid = sum(1 for c in cases if validate_case(c)["valid"])
    print("=" * 60)
    print(f"SUMMARY: {valid}/{len(cases)} valid cases generated")
    print()


if __name__ == "__main__":
    run_import()
