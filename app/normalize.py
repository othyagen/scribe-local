"""Strict, lexicon-based normalization with full audit logging.

Rules:
  - No paraphrasing or free rewriting.
  - Only allow changes for exact or high-confidence fuzzy lexicon matches.
  - Prefer "no change" when uncertain.
  - Every change is logged with full metadata.

Lexicon priority (per language):  custom > medical > general.
"""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from dataclasses import dataclass, asdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Tuple

from app.commit import RawSegment
from app.config import AppConfig


# ── data types ────────────────────────────────────────────────────────

@dataclass(frozen=True)
class NormalizedSegment:
    """A normalized transcript segment (derived from a RawSegment)."""
    seg_id: str
    t0: float
    t1: float
    speaker_id: str
    raw_text: str
    normalized_text: str
    model_name: str
    language: str
    paragraph_id: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_txt_line(self) -> str:
        from app.commit import _fmt_ts
        ts0 = _fmt_ts(self.t0)
        ts1 = _fmt_ts(self.t1)
        return f"[{ts0} - {ts1}] [{self.speaker_id}] {self.normalized_text}"


@dataclass(frozen=True)
class NormalizationChange:
    """Audit record for a single normalization change."""
    seg_id: str
    speaker_id: str
    language: str
    domain: str       # custom / medical / general
    from_text: str
    to_text: str
    confidence: float
    method: str       # exact / fuzzy

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── normalizer ────────────────────────────────────────────────────────

class Normalizer:
    """Applies lexicon-based corrections to raw segments."""

    # Domains in priority order
    DOMAINS: Tuple[str, ...] = ("custom", "medical", "general")

    def __init__(self, config: AppConfig) -> None:
        self.language: str = config.language
        self.fuzzy_threshold: float = config.normalization.fuzzy_threshold
        self.enabled: bool = config.normalization.enabled
        self._lexicons: OrderedDict[str, Dict[str, str]] = self._load_lexicons(
            config.normalization.lexicon_dir, config.language
        )

    # ------------------------------------------------------------------
    # Lexicon loading
    # ------------------------------------------------------------------

    @classmethod
    def _load_lexicons(
        cls, lexicon_dir: str, language: str
    ) -> OrderedDict[str, Dict[str, str]]:
        """Load lexicons for *language* in priority order."""
        base = Path(lexicon_dir) / language
        lexicons: OrderedDict[str, Dict[str, str]] = OrderedDict()
        for domain in cls.DOMAINS:
            path = base / f"{domain}.json"
            if path.exists():
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                lexicons[domain] = data.get("replacements", {})
            else:
                lexicons[domain] = {}
        return lexicons

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def normalize(
        self, segment: RawSegment
    ) -> Tuple[NormalizedSegment, List[NormalizationChange]]:
        """Normalize a raw segment.  Returns (normalized_segment, changes)."""
        if not self.enabled:
            return self._pass_through(segment), []

        text = segment.raw_text
        all_changes: List[NormalizationChange] = []

        # Pass 1 — multi-word exact replacements (longest key first)
        text, changes = self._exact_phrase_pass(text, segment)
        all_changes.extend(changes)

        # Pass 2 — single-word exact + fuzzy replacements
        text, changes = self._word_pass(text, segment)
        all_changes.extend(changes)

        norm = NormalizedSegment(
            seg_id=segment.seg_id,
            t0=segment.t0,
            t1=segment.t1,
            speaker_id=segment.speaker_id,
            raw_text=segment.raw_text,
            normalized_text=text,
            model_name=segment.model_name,
            language=segment.language,
            paragraph_id=segment.paragraph_id,
        )
        return norm, all_changes

    # ------------------------------------------------------------------
    # Internal — phrase-level exact match
    # ------------------------------------------------------------------

    def _exact_phrase_pass(
        self, text: str, segment: RawSegment
    ) -> Tuple[str, List[NormalizationChange]]:
        changes: List[NormalizationChange] = []
        for domain, replacements in self._lexicons.items():
            # Only consider multi-word keys, longest first
            multi = {k: v for k, v in replacements.items() if " " in k}
            for key in sorted(multi, key=len, reverse=True):
                value = multi[key]
                pattern = re.compile(r"\b" + re.escape(key) + r"\b", re.IGNORECASE)
                for match in pattern.finditer(text):
                    changes.append(NormalizationChange(
                        seg_id=segment.seg_id,
                        speaker_id=segment.speaker_id,
                        language=segment.language,
                        domain=domain,
                        from_text=match.group(),
                        to_text=value,
                        confidence=1.0,
                        method="exact",
                    ))
                text = pattern.sub(value, text)
        return text, changes

    # ------------------------------------------------------------------
    # Internal — word-level exact + fuzzy
    # ------------------------------------------------------------------

    def _word_pass(
        self, text: str, segment: RawSegment
    ) -> Tuple[str, List[NormalizationChange]]:
        changes: List[NormalizationChange] = []
        tokens = text.split()
        out_tokens: List[str] = []

        for token in tokens:
            leading, core, trailing = _split_punct(token)
            if not core:
                out_tokens.append(token)
                continue

            replacement, change = self._match_word(core, segment)
            if change is not None:
                changes.append(change)
                out_tokens.append(leading + replacement + trailing)
            else:
                out_tokens.append(token)

        return " ".join(out_tokens), changes

    def _match_word(
        self, word: str, segment: RawSegment
    ) -> Tuple[str, NormalizationChange | None]:
        """Try to match a single word against lexicons (exact then fuzzy)."""
        word_lower = word.lower()

        # --- exact ---
        for domain, replacements in self._lexicons.items():
            # Skip multi-word keys (handled in phrase pass)
            for key, value in replacements.items():
                if " " in key:
                    continue
                if word_lower == key.lower():
                    return value, NormalizationChange(
                        seg_id=segment.seg_id,
                        speaker_id=segment.speaker_id,
                        language=segment.language,
                        domain=domain,
                        from_text=word,
                        to_text=value,
                        confidence=1.0,
                        method="exact",
                    )

        # --- fuzzy ---
        best_value: str | None = None
        best_ratio: float = 0.0
        best_domain: str = ""
        best_key: str = ""

        for domain, replacements in self._lexicons.items():
            for key, value in replacements.items():
                if " " in key:
                    continue
                ratio = SequenceMatcher(None, word_lower, key.lower()).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_value = value
                    best_domain = domain
                    best_key = key

        if best_value is not None and best_ratio >= self.fuzzy_threshold:
            return best_value, NormalizationChange(
                seg_id=segment.seg_id,
                speaker_id=segment.speaker_id,
                language=segment.language,
                domain=best_domain,
                from_text=word,
                to_text=best_value,
                confidence=round(best_ratio, 4),
                method="fuzzy",
            )

        return word, None

    # ------------------------------------------------------------------
    @staticmethod
    def _pass_through(segment: RawSegment) -> NormalizedSegment:
        return NormalizedSegment(
            seg_id=segment.seg_id,
            t0=segment.t0,
            t1=segment.t1,
            speaker_id=segment.speaker_id,
            raw_text=segment.raw_text,
            normalized_text=segment.raw_text,
            model_name=segment.model_name,
            language=segment.language,
            paragraph_id=segment.paragraph_id,
        )


# ── helpers ───────────────────────────────────────────────────────────

def _split_punct(token: str) -> Tuple[str, str, str]:
    """Split a token into (leading_punct, core, trailing_punct)."""
    i = 0
    while i < len(token) and not token[i].isalnum():
        i += 1
    j = len(token)
    while j > i and not token[j - 1].isalnum():
        j -= 1
    return token[:i], token[i:j], token[j:]
