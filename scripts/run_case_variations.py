#!/usr/bin/env python3
"""Case variation generator report.

Loads all YAML cases from ``resources/cases/``, generates all supported
variations for each, and prints a concise report.

Run from the project root::

    python -m scripts.run_case_variations
"""

from __future__ import annotations

import sys
from pathlib import Path

from app.case_system import load_all_cases, validate_case
from app.case_variations import (
    generate_case_variations,
    list_supported_variations,
    summarize_case_variation,
)


_CASE_DIR = Path(__file__).resolve().parent.parent / "resources" / "cases"


def run_report() -> None:
    """Generate variations and print a report."""
    cases = load_all_cases(_CASE_DIR)

    if not cases:
        print(f"No cases found in {_CASE_DIR}")
        sys.exit(1)

    supported = list_supported_variations()
    print(f"Supported variations: {', '.join(supported)}")
    print(f"Base cases: {len(cases)}")
    print("=" * 60)
    print()

    total_variants = 0

    for case in cases:
        base_id = case.get("case_id", "(no id)")
        variants = generate_case_variations(case)
        total_variants += len(variants)

        print(f"  {base_id}")
        print(f"    variants: {len(variants)}")

        for var in variants:
            summary = summarize_case_variation(var)
            validation = validate_case(var)
            status = "valid" if validation["valid"] else "INVALID"
            warning = var.get("meta", {}).get("variation_warning", "")
            warn_str = f"  [{warning}]" if warning else ""

            print(
                f"      {summary['applied_variation']:<35} "
                f"{status}  segs={summary['segment_count']}"
                f"{warn_str}"
            )

        print()

    print("=" * 60)
    print("SUMMARY")
    print(f"  base cases:       {len(cases)}")
    print(f"  variations each:  {len(supported)}")
    print(f"  total variants:   {total_variants}")
    print()


if __name__ == "__main__":
    run_report()
