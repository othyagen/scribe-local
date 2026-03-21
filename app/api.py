"""SCRIBE public API — primary entrypoints in one place.

See docs/GUIDE.md for usage guidance.
"""

from app.case_system import run_case, run_case_script, load_case, load_all_cases
from app.case_tts import run_case_tts
from app.case_compare import compare_case_modes
from app.clinical_state import build_clinical_state
from app.clinical_session import initialize_session, get_app_view, submit_answers
from app.encounter_output import build_encounter_output
from app.case_scoring import score_result_against_ground_truth

__all__ = [
    "run_case",
    "run_case_script",
    "load_case",
    "load_all_cases",
    "run_case_tts",
    "compare_case_modes",
    "build_clinical_state",
    "build_encounter_output",
    "initialize_session",
    "get_app_view",
    "submit_answers",
    "score_result_against_ground_truth",
]
