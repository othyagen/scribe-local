"""Case registry — in-memory metadata index for fast case browsing.

Scans case directories, extracts metadata from each YAML file via
:func:`app.case_schema.extract_case_metadata`, and builds an in-memory
index for filtering and resolution.

Filtering semantics (deterministic, documented):
  - ICD / ICPC / SNOMED: exact normalized match (strip + casefold)
  - organ_system / complaint / tag / origin / difficulty:
    case-insensitive exact match (casefold)
  - All filters combine with AND

Case resolution (deterministic, no guessing):
  1. Path-like ref (contains / or \\ or ends .yaml/.yml) → load directly
  2. Otherwise → exact case_id match in registry
  3. Zero matches → ValueError
  4. Multiple matches → ValueError listing paths
  5. Exactly one → return path

Pure functions — no persistent state, no cache files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from app.case_schema import extract_case_metadata
from app.case_system import load_case


# ── data model ────────────────────────────────────────────────────


@dataclass
class CaseEntry:
    """Lightweight metadata for a single case file."""

    case_id: str
    path: Path
    title: str = ""
    origin: str = "unknown"
    difficulty: str = "unspecified"
    organ_systems: list[str] = field(default_factory=list)
    presenting_complaints: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    patient_age: int | None = None
    patient_sex: str | None = None
    icd10_codes: list[str] = field(default_factory=list)
    icpc_codes: list[str] = field(default_factory=list)
    snomed_codes: list[str] = field(default_factory=list)
    has_ground_truth: bool = False
    segment_count: int = 0


# ── registry building ─────────────────────────────────────────────


def build_registry(case_dirs: list[Path]) -> list[CaseEntry]:
    """Scan directories for YAML case files and build metadata index.

    Args:
        case_dirs: directories to scan for ``.yaml`` / ``.yml`` files.

    Returns:
        List of :class:`CaseEntry` sorted by case_id.
    """
    entries: list[CaseEntry] = []
    seen_paths: set[Path] = set()

    for d in case_dirs:
        d = Path(d)
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix not in (".yaml", ".yml") or not p.is_file():
                continue
            resolved = p.resolve()
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)

            try:
                case = load_case(p)
            except Exception:
                continue

            meta = extract_case_metadata(case)
            entries.append(CaseEntry(
                case_id=meta["case_id"],
                path=p,
                title=meta["title"],
                origin=meta["origin"],
                difficulty=meta["difficulty"],
                organ_systems=meta["organ_systems"],
                presenting_complaints=meta["presenting_complaints"],
                tags=meta["tags"],
                patient_age=meta["patient_age"],
                patient_sex=meta["patient_sex"],
                icd10_codes=meta["icd10_codes"],
                icpc_codes=meta["icpc_codes"],
                snomed_codes=meta["snomed_codes"],
                has_ground_truth=meta["has_ground_truth"],
                segment_count=meta["segment_count"],
            ))

    entries.sort(key=lambda e: e.case_id)
    return entries


# ── filtering ──────────────────────────────────────────────────────


def _exact_match(value: str, candidates: list[str]) -> bool:
    """Case-insensitive exact match against a list."""
    needle = value.casefold()
    return any(c.casefold() == needle for c in candidates)


def _exact_code_match(code: str, candidates: list[str]) -> bool:
    """Exact normalized code match (strip + casefold)."""
    needle = code.strip().casefold()
    return any(c.strip().casefold() == needle for c in candidates)


def filter_registry(
    entries: list[CaseEntry],
    *,
    organ_system: str | None = None,
    complaint: str | None = None,
    tag: str | None = None,
    origin: str | None = None,
    difficulty: str | None = None,
    icd: str | None = None,
    icpc: str | None = None,
    snomed: str | None = None,
) -> list[CaseEntry]:
    """Filter entries by any combination of criteria (AND semantics).

    All string filters use case-insensitive exact matching.
    Code filters (icd, icpc, snomed) use strip + casefold normalization.

    Args:
        entries: list from :func:`build_registry`.
        organ_system: match against organ_systems list.
        complaint: match against presenting_complaints list.
        tag: match against tags list.
        origin: match against origin field.
        difficulty: match against difficulty field.
        icd: exact ICD-10 code match.
        icpc: exact ICPC code match.
        snomed: exact SNOMED code match.

    Returns:
        Filtered list (may be empty).
    """
    result = list(entries)

    if organ_system is not None:
        result = [e for e in result if _exact_match(organ_system, e.organ_systems)]
    if complaint is not None:
        result = [e for e in result if _exact_match(complaint, e.presenting_complaints)]
    if tag is not None:
        result = [e for e in result if _exact_match(tag, e.tags)]
    if origin is not None:
        needle = origin.casefold()
        result = [e for e in result if e.origin.casefold() == needle]
    if difficulty is not None:
        needle = difficulty.casefold()
        result = [e for e in result if e.difficulty.casefold() == needle]
    if icd is not None:
        result = [e for e in result if _exact_code_match(icd, e.icd10_codes)]
    if icpc is not None:
        result = [e for e in result if _exact_code_match(icpc, e.icpc_codes)]
    if snomed is not None:
        result = [e for e in result if _exact_code_match(snomed, e.snomed_codes)]

    return result


# ── case resolution ────────────────────────────────────────────────


def _is_path_ref(case_ref: str) -> bool:
    """Check if case_ref looks like a file path."""
    return (
        "/" in case_ref
        or "\\" in case_ref
        or case_ref.endswith(".yaml")
        or case_ref.endswith(".yml")
    )


def resolve_case(case_ref: str, case_dirs: list[Path]) -> Path:
    """Resolve a case reference to a single YAML file path.

    Args:
        case_ref: either a case_id or a file path.
        case_dirs: directories to scan if resolving by case_id.

    Returns:
        Path to the resolved YAML file.

    Raises:
        FileNotFoundError: if the path does not exist (path mode).
        ValueError: if case_id not found or matches multiple files.
    """
    if _is_path_ref(case_ref):
        p = Path(case_ref)
        if not p.exists():
            raise FileNotFoundError(f"case file not found: {case_ref}")
        return p

    # Scan for exact case_id match.
    registry = build_registry(case_dirs)
    matches = [e for e in registry if e.case_id == case_ref]

    if len(matches) == 0:
        raise ValueError(f"no case found with case_id={case_ref!r}")
    if len(matches) > 1:
        paths = "\n  ".join(str(e.path) for e in matches)
        raise ValueError(
            f"multiple cases match case_id={case_ref!r}:\n  {paths}"
        )

    return matches[0].path
