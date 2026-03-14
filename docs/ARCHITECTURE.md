# Pipeline Architecture

SCRIBE is a staged, forward-only pipeline. Each stage reads from earlier stages and writes to its own output namespace. Later stages never mutate earlier-stage outputs.

## Core Principles

1. **Forward-only flow.** Data flows from source to render. A stage may read any earlier stage's output but must never write back to it.
2. **Structured clinical state is the source of truth.** The `dict` returned by `build_clinical_state()` is the canonical representation of everything the pipeline knows. Clinical notes, FHIR bundles, subtitles, and summaries are projections of this state — they do not add clinical truth.
3. **Immutability of committed outputs.** RAW segments are append-only. Normalized text is derived from RAW and never modified after creation. Extraction results are read-only once written to state.
4. **No fabrication.** If evidence is absent, the field is `null` or empty. No inference across segment boundaries without explicit local evidence.
5. **Deterministic.** Same input produces same output (except wall-clock timestamps in `normalized_timeline`). No randomness, no LLM in the core pipeline.

## Pipeline Stages

```
Source → Transcription → Normalized → Observation → Feature → Graph → Structured → Reasoning → Render/Export
  S1         S2              S3           S4           S5       S6        S7           S8           S9
```

### S1: Source / Input

Audio capture, VAD, and diarization. Produces raw audio buffers and speaker turn boundaries.

| Module | Role |
|--------|------|
| `audio.py` | Microphone capture via sounddevice |
| `audio_quality.py` | Pre-session audio quality check |
| `vad.py` | RMS-based voice activity detection |
| `diarization.py` | Speaker identification + smoothing + merge + relabeling |
| `calibration.py` | Speaker profile embedding extraction and matching |

**Owned outputs:** WAV audio, diarization turns, calibration report.

### S2: Transcription

ASR inference on speech buffers. Produces raw text with timestamps and confidence metrics.

| Module | Role |
|--------|------|
| `asr.py` | faster-whisper speech-to-text |
| `commit.py` | Create immutable RAW segments (seg_id, timestamps, text) |
| `confidence.py` | Per-segment ASR quality metrics |

**Owned outputs:** RAW JSONL (append-only), confidence entries.

### S3: Normalized

Lexicon-based text correction applied to RAW text. Produces the normalized transcript that all downstream stages consume.

| Module | Role |
|--------|------|
| `normalize.py` | Lexicon matching (exact multi-word, exact word, fuzzy) |
| `io.py` | Write normalized JSON, change log, transcript files |
| `lexicon_manager.py` | CLI management of custom lexicon entries |

**Owned outputs:** Normalized JSON array, change log, transcript text files.

**Contract:** Normalized segments are the input boundary for the clinical pipeline. Everything from S4 onward reads `normalized_text` — never raw ASR text.

### S4: Observation

Source-level evidence records. Every segment where a finding appears produces a separate observation, preserving the full evidence trail.

| Module | Role |
|--------|------|
| `observation_layer.py` | Build observation records from segments + extracted findings |

**Owned outputs:** `state["observations"]` — list of `{observation_id, finding_type, value, seg_id, speaker_id, t_start, t_end, source_text}`.

**Reads from:** S3 (normalized segments), S5 (extracted finding lists for matching).

**Note:** S4 runs after S5 extraction but before S5 outputs are consumed by later stages. The observation layer localizes already-extracted findings back to their source segments.

### S5: Feature Extraction

Deterministic extraction of clinical features from the full normalized text and per-segment analysis.

| Module | Role |
|--------|------|
| `extractors.py` | Symptom, negation, duration, medication extraction |
| `extractor_vocab.py` | External vocabulary loading |
| `symptom_timeline.py` | Symptom-time expression pairing per segment |
| `qualifier_extraction.py` | Per-symptom qualifiers (severity, onset, pattern, etc.) |
| `history_extraction.py` | Patient history / context extraction |
| `ice_extraction.py` | Ideas, Concerns, Expectations (conservative) |
| `intensity_extraction.py` | Numeric pain intensity (X/10) |
| `site_extraction.py` | Anatomical site mentions per segment |
| `role_detection.py` | Speaker role classification (clinician/patient) |
| `review_flags.py` | Quality and safety flags |

**Owned outputs:** `state["symptoms"]`, `state["durations"]`, `state["negations"]`, `state["medications"]`, `state["timeline"]`, `state["qualifiers"]`, `state["history"]`, `state["ice"]`, `state["intensities"]`, `state["sites"]`, `state["review_flags"]`, `state["speaker_roles"]`.

### S6: Graph

Evidence-aware graph representation linking findings via conservative same-segment rules.

| Module | Role |
|--------|------|
| `clinical_graph.py` | Graph orchestration |
| `graph/graph.py` | ClinicalGraph container |
| `graph/models.py` | Node / Edge dataclasses |
| `graph/types.py` | NodeType / EdgeType constants |
| `graph/symptom_builder.py` | Symptom-domain graph population |

**Owned outputs:** `state["clinical_graph"]` — `{nodes: [...], edges: [...]}`.

**Reads from:** S4 (observations for evidence_obs_ids), S5 (symptoms, qualifiers, sites, ice, negations), S7 (structured_symptoms for qualifier/modifier nodes).

**Note:** S6 runs after S7 in the current implementation because it reads `derived.structured_symptoms`. This is an acceptable forward reference within the same `build_clinical_state()` call.

### S7: Structured

Domain-grouped per-symptom model with conservative same-segment linking. Consolidates extraction outputs into clinically meaningful structures.

| Module | Role |
|--------|------|
| `structured_symptom_model.py` | Per-symptom spatial/qualitative/temporal/modifier/context/safety/perspective model |

**Owned outputs:** `state["derived"]["structured_symptoms"]`.

**Reads from:** S4 (observations for co-occurrence), S5 (all feature keys), S8 (red_flags for safety linking).

**Note:** S7 runs after `red_flags` is populated (S8) so it can link safety flags. This is an acceptable dependency within `build_clinical_state()`.

### S8: Reasoning

Higher-order clinical reasoning derived from structured features. Pattern matching, problem representation, ontology mapping, temporal reasoning, red flag detection.

| Module | Role |
|--------|------|
| `problem_representation.py` | Core symptom identification, problem narrative |
| `problem_summary.py` | Human-readable problem summary |
| `diagnostic_hints.py` | Rule-based diagnostic suggestions |
| `ontology_mapper.py` | Symptom-to-SNOMED concept mapping |
| `pattern_matcher.py` | Clinical pattern detection (angina-like, migraine-like, etc.) |
| `red_flag_detector.py` | High-risk clinical constellation detection |
| `temporal_normalizer.py` | ISO date/duration normalization |
| `temporal_reasoner.py` | Onset ordering, progression tracking |
| `live_summary.py` | Running clinical summary for live display |
| `classification_router.py` | ICPC / ICD-10 classification selection |
| `icpc_mapper.py` | ICPC-2 code suggestion |
| `icd_mapper.py` | ICD-10 code suggestion |
| `llm_overlay.py` | Optional AI-generated text overlay (provider-abstracted) |

**Owned outputs:** `state["derived"]` keys: `problem_representation`, `problem_focus`, `symptom_representations`, `problem_summary`, `ontology_concepts`, `clinical_patterns`, `running_summary`, `normalized_timeline`, `temporal_context`, `red_flags`, `problem_narrative`. Also `state["diagnostic_hints"]`.

### S9: Render / Export

Projections of clinical state into output formats. These stages read from state — they never modify it.

| Module | Role |
|--------|------|
| `export_clinical_note.py` | Template-driven clinical note (.md / .txt) |
| `export_subtitles.py` | SRT / WebVTT subtitle export |
| `export_summary.py` | Session summary Markdown |
| `export_bundle.py` | Session artifact bundling (directory / ZIP) |
| `fhir_exporter.py` | FHIR Bundle JSON (Encounter, Observation, Condition, Composition) |
| `output_selector.py` | Centralized optional output control |
| `reporting.py` | Session report JSON |

**Owned outputs:** Clinical notes, subtitles, FHIR bundles, session reports, export bundles. All are derived views — deleting them loses no clinical truth.

## Support Modules (no stage)

| Module | Role |
|--------|------|
| `config.py` | Configuration loading and CLI parsing |
| `config_validation.py` | Config schema validation |
| `session_browser.py` | Read-only session listing and inspection |
| `tagging.py` | Speaker tag/label assignment |
| `main.py` | Pipeline orchestration, CLI entry point |
| `live_findings.py` | Live finding extraction for preview |
| `live_preview.py` | Terminal preview rendering |
| `streaming_buffer.py` | Streaming buffer management |

## Execution Order in `build_clinical_state()`

The clinical pipeline (S4-S8) executes in a fixed order within `build_clinical_state()`. The graph (S6) and structured model (S7) run after reasoning outputs they depend on are populated.

```
1.  Extract features (S5): symptoms, negations, durations, medications
2.  Timeline (S5): symptom_timeline
3.  Review flags (S5): review_flags
4.  Diagnostic hints (S8): diagnostic_hints
5.  History (S5): history
6.  Qualifiers (S5): qualifiers
7.  Observations (S4): observation_layer
8.  ICE, intensities, sites (S5)
9.  Problem representation (S8): problem_representation, symptom_representations
10. Problem summary (S8)
11. Ontology, patterns, live summary, timeline normalization, temporal context (S8)
12. Red flags (S8)
13. Structured symptoms (S7) — reads red_flags
14. Problem narrative (S8) — reads structured_symptoms
15. Clinical graph (S6) — reads structured_symptoms, observations
16. Optional outputs (S9): output_selector
```

## Immutability Rules

| Rule | Meaning |
|------|---------|
| RAW is append-only | Once a RAW segment is committed, it is never modified. Resume appends new segments. |
| Normalized is derived | Normalized text is recomputed from RAW at finalization. It is never edited directly. |
| Extractors do not write back | Feature extractors read normalized text and write to state keys. They never modify segments. |
| Reasoning is read-only on features | Problem representation, pattern matcher, etc. read feature keys. They never modify `symptoms`, `qualifiers`, etc. |
| Graph is additive | The clinical graph adds nodes and edges. It never removes or modifies feature-level data. |
| Render is a projection | Export modules read state. They never write back to state (except `output_selector` which adds optional derived keys). |

## Extending the Pipeline

To add a new extractor:
1. Create `app/<extractor>.py` with a pure function taking segments or full_text.
2. Add its output as a new key in `build_clinical_state()` at the appropriate stage.
3. Update `_EXPECTED_KEYS` in `tests/test_clinical_state.py`.
4. The structured symptom model and graph builder can optionally consume the new key.

To add a new graph domain builder:
1. Create `app/graph/<domain>_builder.py`.
2. Call it from `build_clinical_graph()` after `build_symptom_graph()`.
3. Define new `NodeType` / `EdgeType` constants as needed.

To add a new export format:
1. Create `app/export_<format>.py` that reads from clinical state.
2. Wire it in `main.py` with a CLI flag.
3. It must not modify the state dict.
