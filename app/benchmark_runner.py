"""Deterministic benchmark runner — evaluates extraction quality against ground truth.

Runs the full clinical pipeline on synthetic scenario transcripts and compares
structured output against known-good ground truth definitions.

Read-only on pipeline: only reads build_clinical_state() output, never modifies it.
Deterministic: same scenarios always produce same scores.
Self-contained: no external dependencies, no audio — works from dialogue text only.
"""

from __future__ import annotations

import json
import os
from datetime import datetime


def build_segments_from_scenario(scenario: dict) -> list[dict]:
    """Convert scenario dialogue into normalized segment dicts.

    Each dialogue turn becomes a segment with sequential timing (1.0s per turn).

    Args:
        scenario: scenario dict with ``dialogue`` list of ``{speaker_id, text}``.

    Returns:
        list of segment dicts with ``seg_id``, ``t0``, ``t1``,
        ``speaker_id``, ``normalized_text``.
    """
    segments: list[dict] = []
    for i, turn in enumerate(scenario.get("dialogue", [])):
        segments.append({
            "seg_id": i,
            "t0": float(i),
            "t1": float(i + 1),
            "speaker_id": turn["speaker_id"],
            "normalized_text": turn["text"],
        })
    return segments


# ── scoring helpers ──────────────────────────────────────────────


def _set_recall_precision_f1(
    expected: list[str],
    actual: list[str],
    *,
    substring: bool = False,
) -> dict:
    """Compute recall, precision, F1 for two string lists.

    Args:
        expected: ground truth items.
        actual: pipeline output items.
        substring: if True, a ground truth item counts as matched when it
            appears as a case-insensitive substring of any actual item.

    Returns:
        dict with expected, actual, true_positives, false_negatives,
        false_positives, recall, precision, f1.
    """
    if not expected:
        return {
            "expected": expected,
            "actual": actual,
            "true_positives": [],
            "false_negatives": [],
            "false_positives": list(actual),
            "recall": 1.0,
            "precision": 1.0 if not actual else 0.0,
            "f1": 1.0 if not actual else 0.0,
        }

    expected_lower = [e.lower() for e in expected]
    actual_lower = [a.lower() for a in actual]

    tp: list[str] = []
    fn: list[str] = []
    matched_actual: set[int] = set()

    for e in expected:
        e_low = e.lower()
        found = False
        if substring:
            for j, a_low in enumerate(actual_lower):
                if e_low in a_low and j not in matched_actual:
                    matched_actual.add(j)
                    found = True
                    break
        else:
            for j, a_low in enumerate(actual_lower):
                if e_low == a_low and j not in matched_actual:
                    matched_actual.add(j)
                    found = True
                    break
        if found:
            tp.append(e)
        else:
            fn.append(e)

    fp = [actual[j] for j in range(len(actual)) if j not in matched_actual]

    recall = len(tp) / (len(tp) + len(fn)) if (len(tp) + len(fn)) > 0 else 1.0
    precision = len(tp) / (len(tp) + len(fp)) if (len(tp) + len(fp)) > 0 else 1.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {
        "expected": expected,
        "actual": actual,
        "true_positives": tp,
        "false_negatives": fn,
        "false_positives": fp,
        "recall": recall,
        "precision": precision,
        "f1": f1,
    }


# ── public scoring functions ─────────────────────────────────────


def score_symptoms(expected: list[str], actual: list[str]) -> dict:
    """Recall, precision, F1 for symptom extraction."""
    return _set_recall_precision_f1(expected, actual)


def score_negations(expected: list[str], actual: list[str]) -> dict:
    """Substring-based recall for negation accuracy.

    Ground truth has clean terms like ``"sweating"``.  Pipeline produces
    phrased negations like ``"no sweating"``.  Match if the ground truth
    term appears as a substring (case-insensitive) in any pipeline negation.
    """
    return _set_recall_precision_f1(expected, actual, substring=True)


def score_medications(expected: list[str], actual: list[str]) -> dict:
    """Recall, precision, F1 for medication extraction."""
    return _set_recall_precision_f1(expected, actual)


def score_durations(expected: list[str], actual: list[str]) -> dict:
    """Recall, precision, F1 for duration extraction."""
    return _set_recall_precision_f1(expected, actual)


def score_patterns(expected: list[str], actual_patterns: list[dict]) -> dict:
    """Recall, precision, F1 for clinical pattern detection."""
    actual = [p.get("pattern", "") for p in actual_patterns]
    return _set_recall_precision_f1(expected, actual)


def score_red_flags(expected: list[str], actual_flags: list[dict]) -> dict:
    """Recall, precision, F1 for red flag detection."""
    actual = [f.get("flag", "") for f in actual_flags]
    return _set_recall_precision_f1(expected, actual)


def score_qualifiers(expected: list[dict], actual: list[dict]) -> dict:
    """Per-qualifier-field accuracy across symptoms.

    Args:
        expected: ground truth qualifier dicts with ``symptom`` and qualifier fields.
        actual: pipeline qualifier dicts with ``symptom`` and ``qualifiers`` sub-dict.

    Returns:
        dict with ``per_symptom`` breakdown and ``aggregate`` per-field accuracy.
    """
    if not expected:
        return {
            "per_symptom": [],
            "aggregate": {},
            "overall_accuracy": 1.0,
        }

    # Build actual lookup: symptom (lower) → qualifiers dict
    actual_index: dict[str, dict] = {}
    for entry in actual:
        key = entry.get("symptom", "").lower()
        if key and key not in actual_index:
            actual_index[key] = entry.get("qualifiers", {})

    scalar_fields = [
        "severity", "onset", "character", "pattern",
        "progression", "laterality", "radiation",
    ]
    list_fields = ["aggravating_factors", "relieving_factors"]

    per_symptom: list[dict] = []
    field_totals: dict[str, dict] = {}

    for exp_entry in expected:
        symptom = exp_entry.get("symptom", "")
        symptom_key = symptom.lower()
        act_quals = actual_index.get(symptom_key, {})

        symptom_result: dict = {"symptom": symptom, "fields": {}}

        for field in scalar_fields:
            exp_val = exp_entry.get(field)
            if exp_val is None:
                continue
            act_val = act_quals.get(field)
            match = (
                act_val is not None
                and act_val.lower() == exp_val.lower()
            )
            symptom_result["fields"][field] = {
                "expected": exp_val,
                "actual": act_val,
                "match": match,
            }
            field_totals.setdefault(field, {"correct": 0, "total": 0})
            field_totals[field]["total"] += 1
            if match:
                field_totals[field]["correct"] += 1

        for field in list_fields:
            exp_list = exp_entry.get(field, [])
            if not exp_list:
                continue
            act_list = act_quals.get(field, [])
            result = _set_recall_precision_f1(exp_list, act_list)
            symptom_result["fields"][field] = result
            field_totals.setdefault(field, {"correct": 0, "total": 0})
            field_totals[field]["total"] += len(exp_list)
            field_totals[field]["correct"] += len(result["true_positives"])

        per_symptom.append(symptom_result)

    # Aggregate per-field accuracy
    aggregate: dict[str, float] = {}
    total_correct = 0
    total_items = 0
    for field, counts in field_totals.items():
        if counts["total"] > 0:
            aggregate[field] = counts["correct"] / counts["total"]
            total_correct += counts["correct"]
            total_items += counts["total"]

    overall = total_correct / total_items if total_items > 0 else 1.0

    return {
        "per_symptom": per_symptom,
        "aggregate": aggregate,
        "overall_accuracy": overall,
    }


def score_sites(actual_sites: list[dict]) -> dict:
    """Informational count only (no ground truth yet)."""
    return {"count": len(actual_sites), "items": actual_sites}


def score_ice(actual_ice: dict) -> dict:
    """Informational count only (no ground truth yet)."""
    count = 0
    for key in ("ideas", "concerns", "expectations"):
        items = actual_ice.get(key, [])
        if isinstance(items, list):
            count += len(items)
    return {"count": count, "items": actual_ice}


# ── benchmark runner ─────────────────────────────────────────────


def run_benchmark(scenarios: dict[str, dict]) -> dict:
    """Run all scenarios through the clinical pipeline and score results.

    Args:
        scenarios: ``{case_id: scenario_dict}`` from SCENARIOS registry.

    Returns:
        dict with ``timestamp``, ``cases``, and ``aggregate`` keys.
    """
    from app.clinical_state import build_clinical_state

    timestamp = datetime.now().isoformat()
    cases: dict[str, dict] = {}

    for case_id, scenario in scenarios.items():
        gt = scenario.get("ground_truth", {})
        segments = build_segments_from_scenario(scenario)
        state = build_clinical_state(segments)

        derived = state.get("derived", {})

        case_result = {
            "symptom_extraction": score_symptoms(
                gt.get("symptoms", []),
                state.get("symptoms", []),
            ),
            "negation_accuracy": score_negations(
                gt.get("negations", []),
                state.get("negations", []),
            ),
            "medication_accuracy": score_medications(
                gt.get("medications", []),
                state.get("medications", []),
            ),
            "duration_accuracy": score_durations(
                gt.get("durations", []),
                state.get("durations", []),
            ),
            "qualifier_accuracy": score_qualifiers(
                gt.get("qualifiers", []),
                state.get("qualifiers", []),
            ),
            "pattern_detection": score_patterns(
                gt.get("expected_patterns", []),
                derived.get("clinical_patterns", []),
            ),
            "red_flag_detection": score_red_flags(
                gt.get("expected_red_flags", []),
                derived.get("red_flags", []),
            ),
            "site_extraction": score_sites(
                state.get("sites", []),
            ),
            "ice_extraction": score_ice(
                state.get("ice", {}),
            ),
        }
        cases[case_id] = case_result

    # Aggregate across cases
    aggregate = _compute_aggregate(cases)

    return {
        "timestamp": timestamp,
        "cases": cases,
        "aggregate": aggregate,
    }


def _compute_aggregate(cases: dict[str, dict]) -> dict:
    """Compute mean metrics across all cases."""
    metric_keys = [
        ("symptom_extraction", "recall"),
        ("negation_accuracy", "recall"),
        ("medication_accuracy", "recall"),
        ("duration_accuracy", "recall"),
        ("qualifier_accuracy", "overall_accuracy"),
        ("pattern_detection", "recall"),
        ("red_flag_detection", "recall"),
    ]

    agg: dict[str, float] = {}
    values_for_overall: list[float] = []

    for metric_name, score_key in metric_keys:
        values = []
        for case_result in cases.values():
            section = case_result.get(metric_name, {})
            val = section.get(score_key, 0.0)
            values.append(val)
        mean = sum(values) / len(values) if values else 0.0
        agg_key = f"{metric_name}_{score_key}"
        agg[agg_key] = round(mean, 4)
        values_for_overall.append(mean)

    agg["overall_score"] = round(
        sum(values_for_overall) / len(values_for_overall)
        if values_for_overall else 0.0,
        4,
    )

    return agg


# ── output generation ────────────────────────────────────────────


def generate_summary_markdown(results: dict) -> str:
    """Render benchmark results as a Markdown summary."""
    lines: list[str] = []
    lines.append("# Benchmark Summary")
    lines.append("")
    lines.append(f"**Timestamp:** {results.get('timestamp', 'N/A')}")
    lines.append("")

    # Per-case table
    lines.append("## Per-Case Results")
    lines.append("")
    headers = [
        "Case", "Symptoms", "Negations", "Medications",
        "Durations", "Qualifiers", "Patterns", "Red Flags",
    ]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")

    for case_id, case_result in results.get("cases", {}).items():
        row = [case_id]
        for metric in [
            "symptom_extraction", "negation_accuracy", "medication_accuracy",
            "duration_accuracy", "qualifier_accuracy", "pattern_detection",
            "red_flag_detection",
        ]:
            section = case_result.get(metric, {})
            if "recall" in section:
                row.append(f"{section['recall']:.0%}")
            elif "overall_accuracy" in section:
                row.append(f"{section['overall_accuracy']:.0%}")
            else:
                row.append("N/A")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")

    # Aggregate
    lines.append("## Aggregate Scores")
    lines.append("")
    agg = results.get("aggregate", {})
    for key, value in agg.items():
        lines.append(f"- **{key}:** {value:.4f}")
    lines.append("")

    # False negatives / false positives summary
    lines.append("## Diagnosis Details")
    lines.append("")
    for case_id, case_result in results.get("cases", {}).items():
        lines.append(f"### {case_id}")
        lines.append("")
        for metric in [
            "symptom_extraction", "negation_accuracy", "medication_accuracy",
            "duration_accuracy", "pattern_detection", "red_flag_detection",
        ]:
            section = case_result.get(metric, {})
            fn = section.get("false_negatives", [])
            fp = section.get("false_positives", [])
            if fn or fp:
                lines.append(f"**{metric}:**")
                if fn:
                    lines.append(f"  - False negatives: {', '.join(fn)}")
                if fp:
                    lines.append(f"  - False positives: {', '.join(fp)}")
                lines.append("")

    return "\n".join(lines)


def write_benchmark_outputs(results: dict, output_dir: str) -> None:
    """Write benchmark_results.json and benchmark_summary.md.

    Args:
        results: dict from :func:`run_benchmark`.
        output_dir: directory to write outputs into (created if missing).
    """
    os.makedirs(output_dir, exist_ok=True)

    results_path = os.path.join(output_dir, "benchmark_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    summary_path = os.path.join(output_dir, "benchmark_summary.md")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(generate_summary_markdown(results))
