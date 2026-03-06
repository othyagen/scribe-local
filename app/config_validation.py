"""Configuration validation — check config, resources, and templates without running the pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import yaml

from app.config import AppConfig


# ── validation result ────────────────────────────────────────────────

def validate_config(config: AppConfig) -> list[dict]:
    """Validate configuration and referenced resource files.

    Returns a list of issue dicts, each with:
      - ``level``: "error" or "warning"
      - ``message``: human-readable description

    An empty list means the configuration is valid.
    """
    issues: list[dict] = []

    # 1. Output directory
    _check_output_dir(config.output_dir, issues)

    # 2. Lexicon files
    _check_lexicons(config.normalization.lexicon_dir, config.language, issues)

    # 3. Extractor vocabularies
    _check_extractor_vocabs(issues)

    # 4. Templates
    _check_templates(issues)

    return issues


def _check_output_dir(output_dir: str, issues: list[dict]) -> None:
    """Check that the output directory exists or can be created."""
    out = Path(output_dir)
    if out.exists():
        if not out.is_dir():
            issues.append({
                "level": "error",
                "message": f"output path exists but is not a directory: {output_dir}",
            })
    else:
        # Check that the parent exists (so mkdir would succeed)
        parent = out.parent
        if not parent.exists():
            issues.append({
                "level": "warning",
                "message": f"output directory does not exist and parent is missing: {output_dir}",
            })


def _check_lexicons(lexicon_dir: str, language: str, issues: list[dict]) -> None:
    """Check that lexicon directory and files exist and are valid JSON."""
    base = Path(lexicon_dir) / language
    if not base.exists():
        issues.append({
            "level": "warning",
            "message": f"lexicon directory not found: {base}",
        })
        return

    for domain in ("custom", "medical", "general"):
        path = base / f"{domain}.json"
        if not path.exists():
            continue  # missing lexicon files are optional
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict) or "replacements" not in data:
                issues.append({
                    "level": "warning",
                    "message": f"lexicon file missing 'replacements' key: {path}",
                })
        except json.JSONDecodeError as e:
            issues.append({
                "level": "error",
                "message": f"invalid JSON in lexicon file {path}: {e}",
            })


def _check_extractor_vocabs(issues: list[dict]) -> None:
    """Check that extractor vocabulary files are valid JSON arrays."""
    vocab_dir = Path("resources/extractors")
    for name in ("symptoms", "medications"):
        path = vocab_dir / f"{name}.json"
        if not path.exists():
            issues.append({
                "level": "warning",
                "message": f"extractor vocabulary not found: {path}",
            })
            continue
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                issues.append({
                    "level": "error",
                    "message": f"extractor vocabulary is not a JSON array: {path}",
                })
        except json.JSONDecodeError as e:
            issues.append({
                "level": "error",
                "message": f"invalid JSON in extractor vocabulary {path}: {e}",
            })


def _check_templates(
    issues: list[dict],
    template_dir: Optional[Path] = None,
) -> None:
    """Check that all YAML templates in the template directory are valid."""
    base = template_dir or Path("templates")
    if not base.exists():
        issues.append({
            "level": "warning",
            "message": f"template directory not found: {base}",
        })
        return

    yaml_files = sorted(base.glob("*.yaml"))
    if not yaml_files:
        issues.append({
            "level": "warning",
            "message": f"no template files found in: {base}",
        })
        return

    required_keys = {"name", "format", "sections"}
    valid_formats = {"markdown", "text"}

    for path in yaml_files:
        try:
            with open(path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            issues.append({
                "level": "error",
                "message": f"invalid YAML in template {path.name}: {e}",
            })
            continue

        if not isinstance(data, dict):
            issues.append({
                "level": "error",
                "message": f"template {path.name} is not a YAML mapping",
            })
            continue

        missing = required_keys - set(data.keys())
        if missing:
            issues.append({
                "level": "error",
                "message": f"template {path.name} missing required fields: {', '.join(sorted(missing))}",
            })
            continue

        fmt = data.get("format", "markdown")
        if fmt not in valid_formats:
            issues.append({
                "level": "error",
                "message": f"template {path.name} has invalid format '{fmt}'",
            })

        sections = data.get("sections", [])
        if not isinstance(sections, list):
            issues.append({
                "level": "error",
                "message": f"template {path.name}: 'sections' must be a list",
            })


# ── report formatting ────────────────────────────────────────────────

def format_validation_report(
    issues: list[dict],
    config: AppConfig,
    template_dir: Optional[Path] = None,
) -> str:
    """Format validation results as a human-readable report.

    Returns multi-line string with pass/fail indicators.
    """
    lines: list[str] = []
    lines.append("Configuration validation")
    lines.append("")

    errors = [i for i in issues if i["level"] == "error"]
    warnings = [i for i in issues if i["level"] == "warning"]

    # Config loaded
    lines.append("  [PASS] config loaded")

    # Templates
    base = template_dir or Path("templates")
    template_issues = [
        i for i in issues
        if "template" in i["message"].lower()
    ]
    if base.exists():
        yaml_count = len(list(base.glob("*.yaml")))
        if template_issues:
            lines.append(f"  [FAIL] templates ({yaml_count} files, {len(template_issues)} issues)")
        else:
            lines.append(f"  [PASS] templates parsed ({yaml_count})")
    else:
        lines.append("  [FAIL] template directory not found")

    # Extractor vocabularies
    vocab_dir = Path("resources/extractors")
    vocab_issues = [
        i for i in issues
        if "extractor vocabulary" in i["message"].lower()
    ]
    if vocab_dir.exists():
        vocab_count = len(list(vocab_dir.glob("*.json")))
        if vocab_issues:
            lines.append(f"  [FAIL] extractor vocabularies ({len(vocab_issues)} issues)")
        else:
            lines.append(f"  [PASS] extractor vocabularies loaded ({vocab_count})")
    else:
        lines.append("  [FAIL] extractor vocabulary directory not found")

    # Lexicons
    lexicon_issues = [
        i for i in issues
        if "lexicon" in i["message"].lower()
    ]
    if lexicon_issues:
        lines.append(f"  [FAIL] lexicons ({len(lexicon_issues)} issues)")
    else:
        lines.append(f"  [PASS] lexicons loaded ({config.language})")

    # Output directory
    output_issues = [
        i for i in issues
        if "output" in i["message"].lower()
    ]
    if output_issues:
        lines.append(f"  [FAIL] output directory")
    else:
        lines.append(f"  [PASS] output directory accessible")

    # Detail section for errors/warnings
    if errors or warnings:
        lines.append("")
        for issue in errors:
            lines.append(f"  ERROR: {issue['message']}")
        for issue in warnings:
            lines.append(f"  WARNING: {issue['message']}")

    return "\n".join(lines)
