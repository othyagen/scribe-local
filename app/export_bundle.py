"""Export all session artifacts into a single directory or ZIP archive.

Discovers files by glob patterns, copies them with simplified names
into a ``session_<ts>/`` directory, and optionally creates a ZIP.

Never modifies original files.
"""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path


# Glob pattern → simplified base name (without extension).
# Extension is preserved from the matched file.
# Patterns that can match multiple files use None as simplified name
# and are handled specially.
_SINGLE_FILE_PATTERNS: list[tuple[str, str]] = [
    ("raw_{ts}_*.json", "raw_transcript"),
    ("raw_{ts}_*.txt", "raw_transcript"),
    ("normalized_{ts}_*.json", "normalized_transcript"),
    ("normalized_{ts}_*.txt", "normalized_transcript"),
    ("confidence_report_{ts}.json", "confidence_report"),
    ("session_report_{ts}.json", "session_report"),
    ("diarized_segments_{ts}.json", "diarized_segments"),
    ("diarized_{ts}.txt", "diarized_transcript"),
    ("speaker_tags_{ts}.json", "speaker_tags"),
    ("speaker_merge_{ts}.json", "speaker_merge"),
    ("tag_labeled_{ts}.txt", "tag_labeled"),
    ("audio_{ts}.wav", "audio"),
    ("session_summary_{ts}.md", "session_summary"),
    ("subtitles_{ts}.srt", "subtitles"),
    ("subtitles_{ts}.vtt", "subtitles"),
]

# Patterns that can match multiple files — keep original stem as name.
_MULTI_FILE_PATTERNS: list[str] = [
    "clinical_note_{ts}_*.md",
    "clinical_note_{ts}_*.txt",
]


def _discover_session_files(
    output_dir: Path,
    session_ts: str,
) -> list[tuple[Path, str]]:
    """Find all files for a session and compute simplified names.

    Returns list of (source_path, simplified_filename) pairs.
    """
    results: list[tuple[Path, str]] = []
    seen_sources: set[Path] = set()

    # Single-file patterns
    for pattern_tmpl, simple_base in _SINGLE_FILE_PATTERNS:
        pattern = pattern_tmpl.format(ts=session_ts)
        matches = sorted(output_dir.glob(pattern))
        for match in matches:
            if match in seen_sources:
                continue
            seen_sources.add(match)
            simple_name = f"{simple_base}{match.suffix}"
            results.append((match, simple_name))

    # Multi-file patterns (clinical notes — keep template id)
    for pattern_tmpl in _MULTI_FILE_PATTERNS:
        pattern = pattern_tmpl.format(ts=session_ts)
        matches = sorted(output_dir.glob(pattern))
        for match in matches:
            if match in seen_sources:
                continue
            seen_sources.add(match)
            # clinical_note_2026-01-01_12-00-00_soap.md → clinical_note_soap.md
            stem = match.stem
            prefix = f"clinical_note_{session_ts}_"
            if stem.startswith(prefix):
                template_id = stem[len(prefix):]
                simple_name = f"clinical_note_{template_id}{match.suffix}"
            else:
                simple_name = match.name
            results.append((match, simple_name))

    return results


def export_session_bundle(
    session_ts: str,
    output_dir: Path,
    zip_output: bool = False,
) -> Path:
    """Export all session artifacts into a bundle directory or ZIP.

    Args:
        session_ts: session timestamp (e.g. "2026-01-01_12-00-00")
        output_dir: directory containing session output files
        zip_output: if True, create a ZIP archive instead of directory

    Returns:
        Path to the created bundle directory or ZIP file.

    Raises:
        FileNotFoundError: if no files found for the session timestamp.
    """
    output_dir = Path(output_dir)
    files = _discover_session_files(output_dir, session_ts)

    if not files:
        raise FileNotFoundError(
            f"no session files found for timestamp {session_ts}"
        )

    bundle_dir = output_dir / f"session_{session_ts}"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    for src, simple_name in files:
        dst = bundle_dir / simple_name
        shutil.copy2(src, dst)

    if zip_output:
        zip_path = output_dir / f"session_{session_ts}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for item in sorted(bundle_dir.iterdir()):
                zf.write(item, arcname=item.name)
        # Clean up the temporary directory
        shutil.rmtree(bundle_dir)
        return zip_path

    return bundle_dir
