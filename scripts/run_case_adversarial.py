#!/usr/bin/env python3
"""Adversarial case generation report.

Loads all YAML cases from ``resources/cases/``, generates adversarial
variants for each, and prints a concise report.

Run from the project root::

    python -m scripts.run_case_adversarial
"""

from __future__ import annotations

import sys
from pathlib import Path

from app.case_system import load_all_cases, validate_case
from app.case_adversarial import (
    generate_adversarial_cases,
    list_adversarial_strategies,
)


_CASE_DIR = Path(__file__).resolve().parent.parent / "resources" / "cases"


def run_report() -> None:
    """Generate adversarial variants and print a report."""
    cases = load_all_cases(_CASE_DIR)

    if not cases:
        print(f"No cases found in {_CASE_DIR}")
        sys.exit(1)

    strategies = list_adversarial_strategies()
    print(f"Adversarial strategies: {', '.join(strategies)}")
    print(f"Base cases: {len(cases)}")
    print("=" * 70)
    print()

    total_variants = 0

    for case in cases:
        base_id = case.get("case_id", "(no id)")
        variants = generate_adversarial_cases(case)
        total_variants += len(variants)

        print(f"  {base_id}")
        print(f"    variants: {len(variants)}")

        for var in variants:
            adv = var.get("adversarial", {})
            strategy = adv.get("strategy", "?")
            validation = validate_case(var)
            status = "valid" if validation["valid"] else "INVALID"
            seg_count = len(var.get("segments", []))
            warning = var.get("meta", {}).get("variation_warning", "")
            warn_str = f"  [{warning}]" if warning else ""

            # First segment preview.
            segs = var.get("segments", [])
            preview = ""
            if segs:
                last_text = segs[-1].get("normalized_text", "")
                preview = last_text[:50] + ("..." if len(last_text) > 50 else "")

            print(
                f"      {strategy:<25} {status}  segs={seg_count}"
                f"{warn_str}"
            )
            if preview:
                print(f"        last: \"{preview}\"")

        print()

    print("=" * 70)
    print("SUMMARY")
    print(f"  base cases:          {len(cases)}")
    print(f"  strategies:          {len(strategies)}")
    print(f"  total variants:      {total_variants}")
    print()


if __name__ == "__main__":
    run_report()
