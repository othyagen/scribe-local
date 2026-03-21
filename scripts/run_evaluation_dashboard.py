#!/usr/bin/env python3
"""Unified evaluation dashboard — single-entrypoint consolidated report.

Orchestrates existing case replay, variation, adversarial, Synthea
evaluation, scoring, and analysis into one combined report.

Run from the project root::

    python -m scripts.run_evaluation_dashboard
"""

from __future__ import annotations

from collections import Counter
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
from app.mismatch_explainer import (
    apply_suggestions,
    explain_mismatches,
    suggest_improvements,
    summarize_mismatches,
)
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


def build_mismatch_report(groups: list[dict]) -> dict:
    """Collect mismatches across all groups, summarize, and suggest fixes.

    Returns:
        Dict with ``mismatch_summary``, ``suggestions``, and
        ``apply_preview`` (dry-run only).
    """
    all_mismatches: list[list[dict]] = []
    for group in groups:
        for entry in group.get("scored_results", []):
            result_bundle = entry.get("result_bundle", {})
            score = entry.get("score", {})
            case_mismatches = explain_mismatches(result_bundle, score)
            all_mismatches.append(case_mismatches)

    summary = summarize_mismatches(all_mismatches)
    suggestions = suggest_improvements(summary)
    preview = apply_suggestions(suggestions, dry_run=True)

    return {
        "mismatch_summary": summary,
        "suggestions": suggestions,
        "apply_preview": preview,
    }


# ── encounter data collection ──────────────────────────────────────


_PRIORITY_ORDER = {"must_not_miss": 0, "most_likely": 1, "less_likely": 2}


def _collect_encounter_data(groups: list[dict]) -> dict:
    """Aggregate encounter_output data across all scored results."""
    case_count = 0
    key_findings: Counter[str] = Counter()
    red_flags: Counter[str] = Counter()
    red_flag_severities: dict[str, str] = {}
    hypotheses: dict[str, dict] = {}
    findings_by_hypothesis: dict[str, dict[str, dict]] = {}
    seen_questions: dict[str, dict] = {}

    for group in groups:
        for entry in group.get("scored_results", []):
            state = (
                entry.get("result_bundle", {})
                .get("session", {})
                .get("clinical_state", {})
            )
            eo = state.get("encounter_output")
            if not eo:
                continue

            case_count += 1

            for kf in eo.get("key_findings", []):
                key_findings[kf] += 1

            for rf in eo.get("red_flags", []):
                label = rf.get("label", "")
                if label:
                    red_flags[label] += 1
                    red_flag_severities.setdefault(label, rf.get("severity", ""))

            for hyp in eo.get("hypotheses", []):
                title = hyp.get("title", "")
                if not title:
                    continue
                pc = hyp.get("priority_class", "less_likely")
                rank = hyp.get("rank", 0)

                if title not in hypotheses:
                    hypotheses[title] = {
                        "count": 0,
                        "total_rank": 0,
                        "priority_classes": Counter(),
                    }
                hypotheses[title]["count"] += 1
                hypotheses[title]["total_rank"] += rank
                hypotheses[title]["priority_classes"][pc] += 1

                for f in hyp.get("findings", []):
                    name = f.get("name", "")
                    status = f.get("status", "absent")
                    reason = f.get("reason", "")
                    if not name:
                        continue
                    if title not in findings_by_hypothesis:
                        findings_by_hypothesis[title] = {}
                    if name not in findings_by_hypothesis[title]:
                        findings_by_hypothesis[title][name] = {
                            "absent": 0, "present": 0, "negated": 0,
                            "reason": reason,
                        }
                    bucket = status if status in ("absent", "present", "negated") else "absent"
                    findings_by_hypothesis[title][name][bucket] += 1

            # Suggested questions from hypothesis_evidence_gaps.
            for sq in (
                state.get("hypothesis_evidence_gaps", {})
                .get("suggested_questions", [])
            ):
                q_text = sq.get("question", "")
                if q_text and q_text not in seen_questions:
                    seen_questions[q_text] = {
                        "question": q_text,
                        "target_hypothesis": sq.get("target_hypothesis", ""),
                        "reason": sq.get("reason", ""),
                        "priority_class": sq.get("priority_class", "less_likely"),
                    }

    suggested_questions = sorted(
        seen_questions.values(),
        key=lambda q: _PRIORITY_ORDER.get(q["priority_class"], 2),
    )

    return {
        "case_count": case_count,
        "key_findings": key_findings,
        "red_flags": red_flags,
        "red_flag_severities": red_flag_severities,
        "hypotheses": hypotheses,
        "findings_by_hypothesis": findings_by_hypothesis,
        "suggested_questions": suggested_questions,
    }


# ── rendering ───────────────────────────────────────────────────────


def render_dashboard_report(
    groups: list[dict],
    global_analysis: dict,
    mismatch_report: dict | None = None,
    encounter_data: dict | None = None,
) -> str:
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

    if mismatch_report:
        lines.extend(_render_mismatch_section(mismatch_report))

    if encounter_data and encounter_data.get("case_count", 0) > 0:
        lines.extend(_render_encounter_preview(encounter_data))
        lines.extend(_render_combined_hypotheses(encounter_data))
        lines.extend(_render_evidence_gaps(encounter_data))
        lines.extend(_render_suggested_questions(encounter_data))

    return "\n".join(lines)


# ── encounter rendering ────────────────────────────────────────────


def _render_encounter_preview(data: dict) -> list[str]:
    """Render the encounter output preview section."""
    lines: list[str] = []
    kf = data.get("key_findings", Counter())
    rf = data.get("red_flags", Counter())
    hyps = data.get("hypotheses", {})

    if not kf and not rf and not hyps:
        return lines

    lines.append("--- ENCOUNTER OUTPUT PREVIEW ---")
    lines.append(f"  cases_with_encounter_output: {data.get('case_count', 0)}")
    lines.append("")

    if kf:
        lines.append("  top_key_findings:")
        for finding, count in kf.most_common(10):
            lines.append(f"    {finding:<30} ({count}x)")
        lines.append("")

    if rf:
        severities = data.get("red_flag_severities", {})
        lines.append("  red_flags:")
        for label, count in rf.most_common():
            sev = severities.get(label, "")
            lines.append(f"    {label:<30} [{sev}] ({count}x)")
        lines.append("")

    if hyps:
        lines.append("  top_hypotheses:")
        top = sorted(hyps.items(), key=lambda x: -x[1]["count"])[:5]
        for title, info in top:
            avg_rank = info["total_rank"] / info["count"] if info["count"] else 0
            lines.append(f"    {title:<30} ({info['count']}x, avg_rank={avg_rank:.1f})")
        lines.append("")

    return lines


def _render_combined_hypotheses(data: dict) -> list[str]:
    """Render hypotheses grouped by priority class."""
    hyps = data.get("hypotheses", {})
    if not hyps:
        return []

    lines: list[str] = ["--- COMBINED HYPOTHESIS VIEW ---"]

    for pc_label in ("must_not_miss", "most_likely", "less_likely"):
        members = [
            (title, info)
            for title, info in hyps.items()
            if info["priority_classes"].get(pc_label, 0) > 0
        ]
        if not members:
            continue

        members.sort(key=lambda x: x[1]["total_rank"] / max(x[1]["count"], 1))
        lines.append(f"  [{pc_label}]")
        for title, info in members:
            avg_rank = info["total_rank"] / info["count"] if info["count"] else 0
            lines.append(f"    #{avg_rank:.0f} {title} ({info['count']}x)")
        lines.append("")

    return lines


def _render_evidence_gaps(data: dict) -> list[str]:
    """Render absent findings for important hypotheses."""
    findings = data.get("findings_by_hypothesis", {})
    hyps = data.get("hypotheses", {})
    if not findings:
        return []

    # Only show must_not_miss and most_likely hypotheses.
    important = {
        title for title, info in hyps.items()
        if (info["priority_classes"].get("must_not_miss", 0) > 0
            or info["priority_classes"].get("most_likely", 0) > 0)
    }
    relevant = {t: f for t, f in findings.items() if t in important}
    if not relevant:
        return []

    lines: list[str] = ["--- EVIDENCE GAPS ---"]
    for title in sorted(relevant):
        absent = [
            (name, detail)
            for name, detail in relevant[title].items()
            if detail["absent"] > 0
        ]
        if not absent:
            continue
        lines.append(f"  {title}:")
        for name, detail in sorted(absent, key=lambda x: -x[1]["absent"]):
            reason = f" — {detail['reason']}" if detail["reason"] else ""
            lines.append(f"    {name:<25} absent {detail['absent']}x{reason}")
    lines.append("")
    return lines


def _render_suggested_questions(data: dict) -> list[str]:
    """Render suggested next questions ordered by priority."""
    questions = data.get("suggested_questions", [])
    if not questions:
        return []

    lines: list[str] = ["--- SUGGESTED NEXT QUESTIONS ---"]
    for sq in questions[:10]:
        lines.append(f"  [{sq['priority_class']}] {sq['question']}")
        lines.append(f"    hypothesis: {sq['target_hypothesis']}")
        if sq.get("reason"):
            lines.append(f"    reason:     {sq['reason']}")
    lines.append("")
    return lines


# ── mismatch rendering ─────────────────────────────────────────────


def _render_mismatch_section(report: dict) -> list[str]:
    """Render the mismatch analysis section of the dashboard."""
    lines: list[str] = []
    summary = report.get("mismatch_summary", {})
    suggestions = report.get("suggestions", [])
    preview = report.get("apply_preview", {})

    total = summary.get("total_mismatches", 0)
    cases_with = summary.get("cases_with_mismatches", 0)
    cases_total = summary.get("cases_total", 0)

    lines.append("--- MISMATCH ANALYSIS ---")
    lines.append(f"  total_mismatches:     {total}")
    lines.append(f"  cases_with_mismatches: {cases_with} / {cases_total}")

    by_reason = summary.get("by_reason", {})
    if by_reason:
        lines.append("  by_reason:")
        for reason in sorted(by_reason, key=lambda r: -by_reason[r]):
            lines.append(f"    {reason:<30} {by_reason[reason]}")

    top_missed = summary.get("top_missed_labels", [])
    if top_missed:
        lines.append("  top_missed_labels:")
        for entry in top_missed:
            lines.append(f"    {entry['label']:<30} ({entry['count']}x)")

    lines.append("")

    if suggestions:
        lines.append("--- IMPROVEMENT SUGGESTIONS ---")
        for s in suggestions:
            lines.append(f"  [{s['issue']}]")
            lines.append(f"    fix: {s['suggested_fix']}")
            labels = ", ".join(s["affected_labels"][:5])
            if len(s["affected_labels"]) > 5:
                labels += f" (+{len(s['affected_labels']) - 5} more)"
            lines.append(f"    labels: {labels}")
        lines.append("")

    proposed = preview.get("proposed_changes", [])
    skipped = preview.get("skipped_changes", [])

    if proposed or skipped:
        lines.append("--- AUTO-FIX PREVIEW (dry run) ---")
        if proposed:
            lines.append("  proposed (safe to apply):")
            for p in proposed:
                lines.append(
                    f"    + {p['label']:<30} -> {p.get('canonical_target', '?')}"
                )
        if skipped:
            lines.append(f"  skipped: {len(skipped)}")
            reason_counts: dict[str, int] = {}
            for sk in skipped:
                r = sk.get("reason", "unknown")
                reason_counts[r] = reason_counts.get(r, 0) + 1
            for r in sorted(reason_counts, key=lambda x: -reason_counts[x]):
                lines.append(f"    {r:<30} {reason_counts[r]}")
        lines.append("")

    return lines


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
    mismatch_report = build_mismatch_report(groups)
    encounter_data = _collect_encounter_data(groups)
    return {
        "groups": groups,
        "global_analysis": global_analysis,
        "mismatch_report": mismatch_report,
        "encounter_data": encounter_data,
    }


def main() -> None:
    dashboard = run_dashboard()
    report = render_dashboard_report(
        dashboard["groups"],
        dashboard["global_analysis"],
        dashboard.get("mismatch_report"),
        dashboard.get("encounter_data"),
    )
    print(report)


if __name__ == "__main__":
    main()
