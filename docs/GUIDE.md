# SCRIBE Navigation Guide

Quick orientation for the SCRIBE codebase. For the detailed pipeline specification, see [ARCHITECTURE.md](ARCHITECTURE.md). For the public API surface, see [`app/api.py`](../app/api.py).

## System Layers

### Input (S1–S3): Audio capture → ASR → normalization

Captures audio, runs VAD and diarization, transcribes with faster-whisper, applies lexicon-based normalization. Output: normalized segments ready for clinical analysis.

Modules: `audio`, `audio_quality`, `vad`, `diarization`, `calibration`, `asr`, `commit`, `normalize`, `io`, `confidence`, `streaming_buffer`

### Clinical Core (S4–S8): Extraction → structuring → reasoning

Deterministic feature extraction and clinical reasoning. Orchestrated by `build_clinical_state()` in `clinical_state.py`. Takes normalized segments, returns the full clinical state dict.

Modules: `extractors`, `extractor_vocab`, `observation_layer`, `observation_taxonomy`, `observation_normalization`, `clinical_graph`, `graph/*`, `structured_symptom_model`, `problem_representation`, `problem_summary`, `problem_model`, `problem_evidence`, `diagnostic_hints`, `diagnostic_hypotheses`, `hypothesis_ranking`, `hypothesis_prioritization`, `hypothesis_evidence_gaps`, `hypothesis_explanations`, `evidence_strength`, `pattern_matcher`, `red_flag_detector`, `ontology_mapper`, `symptom_groups`, `symptom_timeline`, `role_detection`, `review_flags`, `qualifier_extraction`, `history_extraction`, `ice_extraction`, `intensity_extraction`, `site_extraction`, `temporal_normalizer`, `temporal_reasoner`, `live_summary`, `classification_router`, `icpc_mapper`, `icd_mapper`

### Evaluation (case system): Case loading → execution → scoring → comparison

Load YAML case definitions, run them through the clinical pipeline (text or TTS), score against ground truth, analyze results across suites.

Modules: `case_system`, `case_tts`, `case_compare`, `case_scoring`, `case_analysis`, `case_variations`, `case_adversarial`, `case_validation`, `benchmark_runner`, `clinical_session`, `clinical_metrics`, `synthea_import`, `mismatch_explainer`, `tts_provider`

### Presentation (S9 + scripts): Export formats + dashboard

Render clinical state into output formats. Read-only projections — never modify state.

Modules: `export_clinical_note`, `export_subtitles`, `export_summary`, `export_bundle`, `fhir_exporter`, `output_selector`, `reporting`, `encounter_output`, `clinical_summary`, `clinical_summary_views`, `clinical_insights`

Scripts: `run_evaluation_dashboard`, `run_case_suite`, `run_case_scoring`, `run_case_variations`, `run_case_adversarial`, `run_case_analysis`, `run_case_simulation`, `run_synthea_suite`, `import_synthea_cases`

## Primary Entrypoints

| Function | Module | Purpose |
|----------|--------|---------|
| `run(config, args)` | `app/main.py` | Live recording pipeline |
| `build_clinical_state(segments, ...)` | `app/clinical_state.py` | Full S4–S8 clinical pipeline |
| `build_encounter_output(state)` | `app/encounter_output.py` | Aggregated clinician-facing reasoning view |
| `run_case(case)` | `app/case_system.py` | Execute case from text segments |
| `run_case_script(case)` | `app/case_system.py` | Execute case with scripted answer sequence |
| `run_case_tts(case, output_dir, ...)` | `app/case_tts.py` | Execute case via TTS → ASR → clinical |
| `compare_case_modes(case, output_dir, ...)` | `app/case_compare.py` | Text vs TTS structured diff |
| `initialize_session(segments, config)` | `app/clinical_session.py` | Interactive session facade |
| `score_result_against_ground_truth(result)` | `app/case_scoring.py` | Score case result against ground truth |
| `load_case(path)` / `load_all_cases(dir)` | `app/case_system.py` | Load YAML case definitions |
| `run_dashboard(...)` | `scripts/run_evaluation_dashboard.py` | Evaluation dashboard |

## What to Use When

**Text mode** (`run_case`): Default for evaluation. Deterministic, fast, no audio dependencies. Use this for regression testing and pipeline development.

**TTS mode** (`run_case_tts`): Tests the audio pipeline's effect on clinical output. Requires a TTS provider and ASR engine. Use this to measure how audio processing degrades extraction.

**Compare mode** (`compare_case_modes`): Produces a structured diff between text-mode and TTS-mode results for the same case. Use this to quantify audio-path degradation.

**Dashboard** (`run_evaluation_dashboard`): Batch scoring across all cases with aggregated metrics. Use this for overall system health assessment.

**Direct state inspection** (`build_clinical_state`): Returns the raw clinical state dict. Use this when debugging extractors, adding new features, or writing tests.

**Interactive session** (`initialize_session` + `submit_answers`): Simulates the clinician interaction loop with follow-up questions. Use this when developing the orchestration or interaction layers.

## Module Classification

### Core pipeline (S1–S8)
`audio`, `audio_quality`, `vad`, `diarization`, `calibration`, `asr`, `commit`, `normalize`, `io`, `confidence`, `extractors`, `extractor_vocab`, `observation_layer`, `observation_taxonomy`, `observation_normalization`, `clinical_graph`, `graph/*`, `structured_symptom_model`, `problem_representation`, `problem_summary`, `problem_model`, `problem_evidence`, `diagnostic_hints`, `diagnostic_hypotheses`, `hypothesis_ranking`, `hypothesis_prioritization`, `hypothesis_evidence_gaps`, `hypothesis_explanations`, `evidence_strength`, `pattern_matcher`, `red_flag_detector`, `ontology_mapper`, `symptom_groups`, `symptom_timeline`, `role_detection`, `review_flags`, `qualifier_extraction`, `history_extraction`, `ice_extraction`, `intensity_extraction`, `site_extraction`, `temporal_normalizer`, `temporal_reasoner`, `live_summary`, `classification_router`, `icpc_mapper`, `icd_mapper`, `llm_overlay`, `clinical_state`

### Evaluation & testing
`case_system`, `case_tts`, `case_compare`, `case_scoring`, `case_analysis`, `case_variations`, `case_adversarial`, `case_validation`, `benchmark_runner`, `clinical_session`, `clinical_metrics`, `synthea_import`, `mismatch_explainer`, `tts_provider`

### Presentation & export
`export_clinical_note`, `export_subtitles`, `export_summary`, `export_bundle`, `fhir_exporter`, `output_selector`, `reporting`, `encounter_output`, `clinical_summary`, `clinical_summary_views`, `clinical_insights`, `clinical_interaction`, `clinical_orchestration`

### Utilities
`config`, `config_validation`, `session_browser`, `lexicon_manager`, `tagging`, `live_findings`, `live_preview`, `streaming_buffer`, `clinical_input`, `clinical_update`, `clinical_flow`, `canonicalization`, `clinical_terminology`

## Quick Reference

```bash
# Run all tests
pytest tests/ -q

# Run a single case (text mode)
python -c "from app.case_system import load_case, run_case; print(run_case(load_case('resources/cases/chest_pain.yaml'))['state'].keys())"

# Compare text vs TTS for a case
python -c "from app.case_compare import compare_case_modes; from app.case_system import load_case; compare_case_modes(load_case('resources/cases/chest_pain.yaml'), 'output/compare')"

# Launch evaluation dashboard
python scripts/run_evaluation_dashboard.py

# Process a WAV file through full pipeline
python -m app.main --input-file path/to/file.wav

# Export clinical note from existing session
python -m app.main --session <ts> --export-clinical-note

# List available sessions
python -m app.main --list-sessions

# Run case suite with scoring
python scripts/run_case_suite.py
python scripts/run_case_scoring.py
```
