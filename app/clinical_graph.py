"""Clinical graph orchestration — builds the full graph from clinical state.

Creates an empty graph and delegates to domain-specific builders.
Currently only the symptom builder is active; future builders
(history, exposure, vitals, labs) will be added here.

Pure function — no I/O, no side effects beyond graph construction.
"""

from __future__ import annotations

from app.graph.graph import ClinicalGraph
from app.graph.symptom_builder import build_symptom_graph
from app.graph.encounter_builder import build_encounter_graph
from app.graph.problem_builder import build_problem_graph


def build_clinical_graph(clinical_state: dict) -> ClinicalGraph:
    """Build a clinical graph from assembled clinical state.

    Args:
        clinical_state: dict produced by :func:`build_clinical_state`.

    Returns:
        A populated :class:`ClinicalGraph`.
    """
    graph = ClinicalGraph()

    # Symptom domain
    build_symptom_graph(graph, clinical_state)

    # Encounter domain
    build_encounter_graph(graph, clinical_state)

    # Problem domain
    build_problem_graph(graph, clinical_state)

    # Future domain builders:
    # build_history_graph(graph, clinical_state)
    # build_exposure_graph(graph, clinical_state)
    # build_vitals_graph(graph, clinical_state)
    # build_labs_graph(graph, clinical_state)

    return graph
