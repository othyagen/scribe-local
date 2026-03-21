"""Case system — load, validate, and replay clinical cases.

Provides pure functions for working with stored clinical case files.
Cases are YAML files containing transcript segments, optional config,
ground truth expectations, and scripted answer sequences.

Execution delegates to :mod:`app.clinical_session` and
:mod:`app.clinical_metrics` — no new clinical reasoning logic.

Pure functions — no I/O mutation, no ML, deterministic.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from app.clinical_session import (
    initialize_session,
    get_app_view,
    submit_answers,
    apply_manual_update,
)
from app.clinical_metrics import derive_clinical_metrics


# ── constants ───────────────────────────────────────────────────────


_REQUIRED_FIELDS = frozenset({"case_id", "segments"})

_OPTIONAL_FIELDS = frozenset({
    "title", "description", "config", "ground_truth",
    "answer_script", "meta", "provenance", "safety",
})

_KNOWN_FIELDS = _REQUIRED_FIELDS | _OPTIONAL_FIELDS

_SEGMENT_REQUIRED_KEYS = frozenset({
    "seg_id", "t0", "t1", "speaker_id", "normalized_text",
})

_DEFAULT_CONFIG: dict = {
    "mode": "assist",
    "update_strategy": "manual",
    "show_summary_views": True,
    "show_insights": True,
    "show_questions": True,
}

_ANSWER_TYPE_FIELD = "question_type"


# ── loading ─────────────────────────────────────────────────────────


def load_case(path: str | Path) -> dict:
    """Load a single case from a YAML file.

    Args:
        path: file path to a ``.yaml`` or ``.yml`` case file.

    Returns:
        Parsed case dict.

    Raises:
        FileNotFoundError: if the file does not exist.
        yaml.YAMLError: if the file contains invalid YAML.
    """
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Case file must contain a YAML mapping, got {type(data).__name__}")
    return data


def load_all_cases(case_dir: str | Path) -> list[dict]:
    """Load all YAML case files from a directory.

    Scans for ``*.yaml`` and ``*.yml`` files, sorted by filename.

    Args:
        case_dir: directory containing case files.

    Returns:
        List of parsed case dicts.
    """
    case_dir = Path(case_dir)
    cases: list[dict] = []
    if not case_dir.is_dir():
        return cases
    for p in sorted(case_dir.iterdir()):
        if p.suffix in (".yaml", ".yml") and p.is_file():
            cases.append(load_case(p))
    return cases


# ── validation ──────────────────────────────────────────────────────


def validate_case(case: dict) -> dict:
    """Validate a case dict against the expected schema.

    Returns:
        ``{"valid": bool, "errors": [...], "warnings": [...]}``
    """
    errors: list[str] = []
    warnings: list[str] = []

    # Required fields.
    for field in _REQUIRED_FIELDS:
        if field not in case:
            errors.append(f"missing required field: {field}")

    # Segments structure.
    segments = case.get("segments")
    if segments is not None:
        if not isinstance(segments, list):
            errors.append("segments must be a list")
        elif len(segments) == 0:
            errors.append("segments must not be empty")
        else:
            for i, seg in enumerate(segments):
                if not isinstance(seg, dict):
                    errors.append(f"segment[{i}] must be a dict")
                    continue
                missing = _SEGMENT_REQUIRED_KEYS - set(seg.keys())
                if missing:
                    errors.append(
                        f"segment[{i}] missing keys: {', '.join(sorted(missing))}"
                    )

    # Config (optional).
    config = case.get("config")
    if config is not None and not isinstance(config, dict):
        errors.append("config must be a dict")

    # Ground truth (optional).
    gt = case.get("ground_truth")
    if gt is not None and not isinstance(gt, dict):
        errors.append("ground_truth must be a dict")

    # Answer script (optional).
    script = case.get("answer_script")
    if script is not None:
        if not isinstance(script, list):
            errors.append("answer_script must be a list")
        else:
            for i, entry in enumerate(script):
                if not isinstance(entry, dict):
                    errors.append(f"answer_script[{i}] must be a dict")
                elif _ANSWER_TYPE_FIELD not in entry:
                    warnings.append(
                        f"answer_script[{i}] missing '{_ANSWER_TYPE_FIELD}'"
                    )
                elif "value" not in entry:
                    warnings.append(f"answer_script[{i}] missing 'value'")

    # Unknown fields.
    unknown = set(case.keys()) - _KNOWN_FIELDS
    if unknown:
        warnings.append(f"unknown fields: {', '.join(sorted(unknown))}")

    # Provenance validation.
    from app.case_provenance import validate_provenance

    prov_result = validate_provenance(case)
    errors.extend(prov_result["errors"])
    warnings.extend(prov_result["warnings"])

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
    }


# ── execution ───────────────────────────────────────────────────────


def _merge_config(case: dict) -> dict:
    """Merge case config with defaults."""
    base = dict(_DEFAULT_CONFIG)
    case_config = case.get("config")
    if isinstance(case_config, dict):
        base.update(case_config)
    return base


def _text_input_metadata() -> dict:
    """Build input_metadata for text-mode execution."""
    return {"mode": "text", "synthetic": False, "tts": None}


def _build_result(
    case: dict,
    session: dict,
    app_view: dict,
    metrics: dict,
    validation: dict,
) -> dict:
    """Build a structured result bundle."""
    return {
        "case_id": case.get("case_id", ""),
        "session": session,
        "app_view": app_view,
        "metrics": metrics,
        "ground_truth": case.get("ground_truth") or {},
        "validation": validation,
        "input_metadata": _text_input_metadata(),
    }


def run_case(case: dict) -> dict:
    """Run a case through the session pipeline (no answer script).

    Initializes a session, builds the app view, derives metrics, and
    returns a structured result bundle.

    Args:
        case: parsed case dict.

    Returns:
        Result bundle dict.
    """
    validation = validate_case(case)
    if not validation["valid"]:
        return {
            "case_id": case.get("case_id", ""),
            "session": {},
            "app_view": {},
            "metrics": {},
            "ground_truth": case.get("ground_truth") or {},
            "validation": validation,
            "input_metadata": _text_input_metadata(),
        }

    config = _merge_config(case)
    session = initialize_session(case["segments"], config=config)
    app_view = get_app_view(session)
    metrics = derive_clinical_metrics(session["clinical_state"])

    return _build_result(case, session, app_view, metrics, validation)


def run_case_script(case: dict) -> dict:
    """Run a case with its answer script applied.

    Initializes a session, applies each scripted answer through
    :func:`submit_answers`, drains pending if in manual mode,
    and returns the final result bundle.

    Args:
        case: parsed case dict (should have ``answer_script``).

    Returns:
        Result bundle dict.
    """
    validation = validate_case(case)
    if not validation["valid"]:
        return {
            "case_id": case.get("case_id", ""),
            "session": {},
            "app_view": {},
            "metrics": {},
            "ground_truth": case.get("ground_truth") or {},
            "validation": validation,
            "input_metadata": _text_input_metadata(),
        }

    config = _merge_config(case)
    session = initialize_session(case["segments"], config=config)

    # Apply answer script.
    script = case.get("answer_script") or []
    for entry in script:
        answer = {
            "type": entry.get(_ANSWER_TYPE_FIELD, ""),
            "value": entry.get("value", ""),
        }
        if "related" in entry:
            answer["related"] = entry["related"]
        session = submit_answers(session, [answer])

    # Drain pending observations if any remain.
    if session.get("pending_observations"):
        session = apply_manual_update(session)

    app_view = get_app_view(session)
    metrics = derive_clinical_metrics(session["clinical_state"])

    return _build_result(case, session, app_view, metrics, validation)


# ── result helpers ──────────────────────────────────────────────────


def extract_top_hypotheses(result: dict, n: int = 3) -> list[dict]:
    """Extract the top N hypotheses from a result bundle.

    Args:
        result: result bundle from :func:`run_case` or :func:`run_case_script`.
        n: maximum number of hypotheses to return.

    Returns:
        List of hypothesis dicts (may be empty).
    """
    state = result.get("session", {}).get("clinical_state", {})
    hypotheses = state.get("hypotheses", [])
    return list(hypotheses[:n])


def compare_result_to_ground_truth(result: dict) -> dict:
    """Compare a result bundle against its ground truth.

    Checks whether expected hypotheses and red flags appear in the
    actual outputs.

    Args:
        result: result bundle dict.

    Returns:
        Comparison dict with ``hypothesis_matches`` and
        ``red_flag_matches`` lists.
    """
    gt = result.get("ground_truth", {})
    state = result.get("session", {}).get("clinical_state", {})

    # Hypothesis comparison.
    expected_hyps = gt.get("expected_hypotheses", [])
    actual_hyps = {
        h.get("title", "").lower()
        for h in state.get("hypotheses", [])
    }
    hyp_matches: list[dict] = []
    for expected in expected_hyps:
        expected_lower = expected.lower() if isinstance(expected, str) else ""
        found = any(expected_lower in a for a in actual_hyps)
        hyp_matches.append({"expected": expected, "found": found})

    # Red flag comparison.
    expected_flags = gt.get("red_flags", [])
    derived = state.get("derived", {})
    actual_flags = {
        rf.get("label", "").lower()
        for rf in derived.get("red_flags", [])
    }
    flag_matches: list[dict] = []
    for expected in expected_flags:
        expected_lower = expected.lower() if isinstance(expected, str) else ""
        found = any(expected_lower in a for a in actual_flags)
        flag_matches.append({"expected": expected, "found": found})

    # Key findings comparison.
    expected_findings = gt.get("key_findings", [])
    actual_symptoms = {s.lower() for s in state.get("symptoms", [])}
    finding_matches: list[dict] = []
    for expected in expected_findings:
        expected_lower = expected.lower() if isinstance(expected, str) else ""
        found = expected_lower in actual_symptoms
        finding_matches.append({"expected": expected, "found": found})

    return {
        "hypothesis_matches": hyp_matches,
        "red_flag_matches": flag_matches,
        "finding_matches": finding_matches,
    }
