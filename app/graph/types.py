"""Node and edge type constants for the clinical graph."""

from __future__ import annotations


class NodeType:
    """Node type constants."""

    SYMPTOM = "symptom"
    SITE = "site"
    CHARACTER = "character"
    ONSET = "onset"
    PATTERN = "pattern"
    PROGRESSION = "progression"
    MODIFIER = "modifier"
    ASSOCIATED_SYMPTOM = "associated_symptom"
    NEGATED_SYMPTOM = "negated_symptom"
    ICE_ITEM = "ice_item"


class EdgeType:
    """Edge type constants."""

    HAS_SITE = "HAS_SITE"
    HAS_CHARACTER = "HAS_CHARACTER"
    HAS_ONSET = "HAS_ONSET"
    HAS_PATTERN = "HAS_PATTERN"
    HAS_PROGRESSION = "HAS_PROGRESSION"
    AGGRAVATED_BY = "AGGRAVATED_BY"
    RELIEVED_BY = "RELIEVED_BY"
    ASSOCIATED_WITH = "ASSOCIATED_WITH"
    NEGATED_BY = "NEGATED_BY"
    EXPRESSED_AS = "EXPRESSED_AS"
