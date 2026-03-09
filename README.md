# scribe-local

Privacy-first, local AI scribe for near-real-time speech transcription. All processing runs on your machine — no audio or text leaves your device.

Designed for clinical use: deterministic, auditable, and safe.

## Supported Languages

| Code | Language |
|------|----------|
| `da` | Danish   |
| `sv` | Swedish  |
| `en` | English  |

Language is set once at session start. No automatic detection or mid-session switching.

---

## Setup

### 1. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Verify GPU / CUDA (optional but recommended)

faster-whisper uses CTranslate2 under the hood. To check CUDA availability:

```bash
python -c "import ctranslate2; print('CUDA devices:', ctranslate2.get_cuda_device_count())"
```

If the count is 0, the system will fall back to CPU automatically. CPU mode uses `int8` quantization and is slower but fully functional.

For CUDA support you need:
- An NVIDIA GPU with CUDA capability
- CUDA toolkit installed (version compatible with your ctranslate2 build)
- cuDNN installed

### 4. Check your microphone

```bash
python -m app.main --list-audio-devices
```

This prints all audio devices. Find your microphone's index number and set it in `config.yaml` under `audio.device`, or leave it as `null` to use the system default.

---

## Usage

### Basic run (uses config.yaml defaults)

```bash
python -m app.main --config config.yaml
```

### Override settings via CLI

```bash
python -m app.main --language da --model large-v3 --device cuda
python -m app.main --vad-speech-threshold 0.015 --vad-short-silence 0.8
python -m app.main --output-dir my_session
```

### CLI flags

| Flag | Description |
|------|-------------|
| `--config PATH` | Path to YAML config file (default: `config.yaml`) |
| `--language {da,sv,en}` | Session language |
| `--model NAME` | ASR model (e.g. `large-v3`, `medium`, `small`) |
| `--device {cuda,cpu}` | Compute device |
| `--output-dir PATH` | Output directory |
| `--vad-speech-threshold FLOAT` | RMS threshold for speech detection |
| `--vad-short-silence FLOAT` | Seconds of silence to commit a sentence |
| `--vad-long-silence FLOAT` | Seconds of silence to start a new paragraph |
| `--list-audio-devices` | Print audio devices and exit |
| `--auto-tags {none,alphabetical,index}` | Auto-assign speaker tags (default: `none`) |
| `--set-tag SPK=TAG` | Set speaker tag, repeatable (e.g. `--set-tag spk_0=Host`) |
| `--set-label SPK=LABEL` | Set speaker label, repeatable (e.g. `--set-label spk_0=Alice`) |
| `--merge SPK=TARGET` | Merge speaker into target, repeatable (e.g. `--merge spk_2=spk_0`) |
| `--session TIMESTAMP` | Session timestamp for standalone tag/merge/export operations |
| `--resume TIMESTAMP` | Resume an interrupted session by appending to its files |
| `--list-sessions` | List all sessions in the output directory and exit |
| `--show-session TIMESTAMP` | Show detailed info for a session and exit |
| `--audio-precheck` | Enable audio quality pre-check (overrides config) |
| `--no-audio-precheck` | Disable audio quality pre-check (overrides config) |
| `--audio-precheck-seconds FLOAT` | Pre-check recording duration in seconds (overrides config) |
| `--export-srt` | Export diarized transcript as SRT subtitle file |
| `--export-vtt` | Export diarized transcript as WebVTT subtitle file |
| `--export-summary` | Export session summary as Markdown file |
| `--export-clinical-note` | Export clinical note from normalized transcript |
| `--note-template ID` | Clinical note template ID (default: `soap`) |
| `--export-notes IDS` | Export multiple clinical notes (comma-separated template IDs) |
| `--export-session TIMESTAMP` | Export all session artifacts into a bundle directory |
| `--zip` | Create ZIP archive instead of directory (use with `--export-session`) |
| `--reprocess TIMESTAMP` | Reprocess a single session (re-normalize, re-export) and exit |
| `--reprocess-all` | Reprocess all sessions (re-normalize, re-export) and exit |
| `--add-term FROM=TO` | Add or update a custom lexicon term (e.g. `--add-term pt=patient`) |
| `--remove-term FROM` | Remove a custom lexicon term (e.g. `--remove-term pt`) |
| `--list-terms` | List all custom lexicon terms and exit |

### Stopping

Press **Ctrl+C** to stop recording. The pipeline will flush any remaining audio, write final output files, and print a summary.

---

## Output Files

All outputs are written to the configured `output_dir` (default: `outputs/`).

A session produces these files:

| File | Format | Purpose |
|------|--------|---------|
| `raw_<timestamp>_<model>.json` | JSON Lines | Immutable RAW segments (one JSON object per line) |
| `raw_<timestamp>_<model>.txt` | Plain text | Human-readable RAW transcript |
| `normalized_<timestamp>_<model>.json` | JSON array | Normalized segments |
| `normalized_<timestamp>_<model>.txt` | Plain text | Human-readable normalized transcript |
| `changes_<timestamp>_<model>.json` | JSON array | Full audit log of every normalization change |
| `audio_<timestamp>.wav` | WAV | Full session audio (16kHz, mono, 16-bit PCM) |
| `diarization_<timestamp>.json` | JSON | Speaker turns (only when `diarization.backend: pyannote`) |
| `diarized_segments_<timestamp>.json` | JSON array | Segment relabeling map (old → new speaker_id) |
| `diarized_<timestamp>.txt` | Plain text | Transcript with diarized speaker_ids |
| `speaker_merge_<timestamp>.json` | JSON | Speaker merge map (only when merges are applied) |
| `speaker_tags_<timestamp>.json` | JSON | Speaker tag/label mapping |
| `tag_labeled_<timestamp>.txt` | Plain text | Transcript with human-readable speaker tags |
| `confidence_report_<timestamp>.json` | JSON | Per-segment ASR quality flags |
| `session_report_<timestamp>.json` | JSON | Consolidated session summary (config, flags, stats) |
| `subtitles_<timestamp>.srt` | SRT | Subtitle file with speaker-prefixed cues (when `--export-srt`) |
| `subtitles_<timestamp>.vtt` | WebVTT | Subtitle file with speaker-prefixed cues (when `--export-vtt`) |
| `clinical_note_<timestamp>_<template>.md` | Markdown | Clinical note generated from template (when `--export-clinical-note`) |

### RAW vs. NORMALIZED

**RAW** is the audit truth. It contains exactly what the ASR model produced, with no modifications. RAW files are append-only and must never be altered after writing.

**NORMALIZED** is the display-ready version. It applies strict, lexicon-based corrections (spelling fixes, abbreviation expansion). Every change from RAW to NORMALIZED is logged in the changes file with:
- Which segment was changed (`seg_id`)
- Who was speaking (`speaker_id`)
- What language
- Which lexicon domain matched (`custom` / `medical` / `general`)
- The original text (`from`) and replacement (`to`)
- Match confidence score
- Match method (`exact` or `fuzzy`)

No free rewriting. No invented facts. No paraphrasing.

---

## Architecture

```
Microphone → Audio Capture → VAD → Diarization → ASR → Commit → Normalize → Output
               (sounddevice)  (RMS)  (speaker_id)  (whisper)  (immutable)  (lexicon)
```

Each layer has a single responsibility:

| Module | Responsibility |
|--------|---------------|
| `audio.py` | Capture microphone audio via sounddevice |
| `vad.py` | RMS-based voice activity detection |
| `diarization.py` | Speaker identification (metadata only, never modifies text) |
| `asr.py` | Speech-to-text using faster-whisper |
| `commit.py` | Create immutable RAW segments with IDs and timestamps |
| `normalize.py` | Apply lexicon corrections with audit logging |
| `io.py` | Write all output files |
| `config.py` | Configuration loading and CLI argument parsing |
| `tagging.py` | Speaker tag/label assignment and tagged transcript generation |
| `lexicon_manager.py` | CLI lexicon management (add, remove, list custom terms) |
| `extractors.py` | Deterministic symptom/medication/negation/duration extraction |
| `extractor_vocab.py` | Load extractor vocabularies from external JSON files |
| `export_clinical_note.py` | Template-driven clinical note generation |
| `symptom_timeline.py` | Symptom–time expression pairing per segment |
| `diagnostic_hints.py` | Rule-based diagnostic suggestions with SNOMED codes |
| `clinical_state.py` | Structured clinical state assembly from all extraction modules |
| `problem_representation.py` | Deterministic structured problem representation from clinical state |
| `problem_summary.py` | Deterministic human-readable problem summary from core symptom |
| `ontology_mapper.py` | Lightweight file-based symptom-to-SNOMED concept mapping (143 SNOMED CT entries) |
| `pattern_matcher.py` | Deterministic rule-based clinical pattern detection |
| `live_summary.py` | Running clinical summary aggregation for live display |
| `qualifier_extraction.py` | Per-symptom qualifier extraction (severity, onset, pattern, etc.) |
| `history_extraction.py` | Patient history/context extraction from transcript |
| `temporal_normalizer.py` | ISO date/duration normalization of time expressions |
| `temporal_reasoner.py` | Clinical onset ordering, progression tracking, temporal uncertainty |
| `red_flag_detector.py` | High-risk clinical constellation detection |
| `classification_router.py` | Configurable classification system selector (ICPC / ICD-10) |
| `icpc_mapper.py` | ICPC-2 code suggestion from symptoms (105 curated mappings) |
| `icd_mapper.py` | ICD-10 code suggestion with symptom + pattern level mapping |
| `fhir_exporter.py` | Deterministic FHIR Bundle export (Encounter, Observation, Condition, Composition) |
| `output_selector.py` | Centralized control of optional outputs (classification, FHIR, AI overlay) |
| `llm_overlay.py` | Optional AI-generated clinical text overlay (provider-abstracted) |
| `export_bundle.py` | Session artifact bundling (directory or ZIP export) |
| `role_detection.py` | Deterministic speaker role classification (clinician/patient) |
| `main.py` | Pipeline orchestration |

### Speaker Diarization

`speaker_id` is an acoustic-only label (e.g., `spk_0`, `spk_1`). It identifies *who* is speaking based on audio characteristics.

Speaker diarization:
- Runs BEFORE ASR
- NEVER modifies transcribed text
- NEVER assigns roles (patient, clinician, etc.)
- NEVER depends on language or content

The default backend assigns all audio to `spk_0`.

#### Pyannote backend

Set `diarization.backend: pyannote` in `config.yaml` to enable post-session speaker diarization using [pyannote.audio](https://github.com/pyannote/pyannote-audio). This runs on the saved session WAV after recording ends and writes a separate `diarization_<timestamp>.json` file with speaker turns.

Requirements:
- Set the `HF_TOKEN` environment variable with a valid Hugging Face token
- Accept the gated model licenses on Hugging Face:
  - [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
  - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
  - [pyannote/speaker-diarization-community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)

Diarization output format:

```json
{
  "turns": [
    {"start": 0.0, "end": 3.2, "speaker": "spk_0"},
    {"start": 3.2, "end": 5.1, "speaker": "spk_1"}
  ]
}
```

#### Diarization smoothing (reducing short speaker flips)

Pyannote often produces very short speaker turns from backchannels ("mm", "ja") that cause noisy speaker flips in the transcript. Smoothing cleans these up automatically.

Three settings control smoothing behavior:

| Setting | Default | What it does |
|---------|---------|--------------|
| `diarization.smoothing` | `true` | Enable/disable smoothing entirely |
| `diarization.min_turn_sec` | `0.7` | Turns shorter than this (in seconds) are merged into the nearest neighbor. Same-speaker neighbors are preferred; otherwise the longer neighbor absorbs the short turn. |
| `diarization.gap_merge_sec` | `0.3` | If two consecutive turns from the same speaker are separated by a gap this small (in seconds) or less, they are merged into one continuous turn. |

Smoothing only affects derived output files (`diarization_<timestamp>.json`, `diarized_segments_<timestamp>.json`, `diarized_<timestamp>.txt`, `tag_labeled_<timestamp>.txt`). RAW and normalized outputs are **never** modified.

##### Presets

**A) Clinic — typical 1:1 consult (balanced)**

Good default for two-speaker sessions with normal turn-taking.

```yaml
diarization:
  backend: pyannote
  smoothing: true
  min_turn_sec: 0.7
  gap_merge_sec: 0.3
```

**B) Interrupt-heavy (many "mm/ja", quick overlaps)**

More aggressive smoothing for sessions with frequent backchannels and overlapping speech.

```yaml
diarization:
  backend: pyannote
  smoothing: true
  min_turn_sec: 1.2
  gap_merge_sec: 0.5
```

**C) Noisy room / TV in background**

Aggressive smoothing to reduce false speaker changes caused by background noise. Also consider lowering the VAD threshold and positioning the microphone closer to the speakers.

```yaml
diarization:
  backend: pyannote
  smoothing: true
  min_turn_sec: 1.5
  gap_merge_sec: 0.8
```

**D) Multi-speaker meeting (3+ people)**

Less aggressive smoothing to preserve genuine short turns from different speakers.

```yaml
diarization:
  backend: pyannote
  smoothing: true
  min_turn_sec: 0.4
  gap_merge_sec: 0.2
```

#### Speaker merge

Pyannote sometimes splits the same person into multiple speaker IDs (e.g., `spk_0` and `spk_2` are actually the same person). Use `--merge` to combine them:

```bash
# Inspect the diarized output to identify split speakers
# Then merge spk_2 into spk_0:
python -m app.main --session 2026-02-22_14-30-00 --merge spk_2=spk_0

# Multiple merges at once:
python -m app.main --session 2026-02-22_14-30-00 \
  --merge spk_2=spk_0 --merge spk_3=spk_1

# Chain merges are resolved automatically:
# spk_3→spk_2 and spk_2→spk_0 becomes spk_3→spk_0
python -m app.main --session 2026-02-22_14-30-00 \
  --merge spk_3=spk_2 --merge spk_2=spk_0
```

Merges are saved to `speaker_merge_<timestamp>.json` and applied to the diarization turns. After merging, adjacent turns from the now-same speaker are combined. Cycles are detected and rejected with a friendly error.

Only derived outputs are affected. RAW and normalized outputs are **never** modified.

#### Segment relabeling

After diarization (and optional smoothing/merging), each ASR segment is relabeled with the speaker who has the largest time overlap. This produces `diarized_segments_<timestamp>.json` (relabeling map) and `diarized_<timestamp>.txt` (transcript with new speaker_ids). RAW and normalized outputs are never modified.

### Speaker Tagging

Speaker tags let you assign human-readable names to diarized speakers (`spk_0`, `spk_1`, ...). Tags are generic — use them for any context (clinic, podcast, meeting, etc.).

Tag mapping file format (`speaker_tags_<timestamp>.json`):

```json
{
  "spk_0": {"tag": "Host", "label": "Alice"},
  "spk_1": {"tag": "Guest", "label": null}
}
```

The `tag` field is the display name. The optional `label` is a personal name or identifier. In the tagged transcript, speakers appear as `[Host]` or `[Host: Alice]`.

#### Standalone tag mode

Use `--session` to tag an existing session without recording:

```bash
# Set tags manually
python -m app.main --session 2026-02-21_16-30-10 \
  --set-tag spk_0="Host" --set-tag spk_1="Guest" \
  --set-label spk_0="Alice"

# Auto-tag with alphabetical names (Speaker A, Speaker B, ...)
python -m app.main --session 2026-02-21_16-30-10 --auto-tags alphabetical

# Auto-tag with index names (Speaker 1, Speaker 2, ...)
python -m app.main --session 2026-02-21_16-30-10 --auto-tags index
```

Auto-tags only fill missing entries — they never overwrite existing manual tags. Manual `--set-tag` / `--set-label` overrides are applied first.

#### During a session

Pass `--auto-tags` during recording to auto-tag speakers after diarization completes:

```bash
python -m app.main --config config.yaml --auto-tags alphabetical
```

### Calibration mode (Phase 1 — infrastructure)

Calibration profiles let the pipeline assign consistent `spk_N` IDs across sessions by matching speaker embeddings. A profile maps human-readable speaker names to embedding vectors and internal `spk_N` IDs.

Profile format (`profiles/<name>.json`):

```json
{
  "speakers": {
    "Speaker A": {"embedding": [0.12, -0.34, ...]},
    "Speaker B": {"embedding": [0.56, 0.78, ...]}
  },
  "speaker_id_map": {
    "Speaker A": "spk_0",
    "Speaker B": "spk_1"
  }
}
```

The calibration step runs after smoothing and before merge. For each diarization turn that has an `"embedding"` field, it computes cosine similarity against all profile speakers and overrides `turn["speaker"]` with the mapped `spk_N` ID if the best match exceeds the threshold.

#### Creating calibration profiles

Record voice samples and save embeddings:

```bash
python -m app.main --create-profile my_clinic --profile-speakers 2 --profile-duration 12
```

This records each speaker for 12 seconds, extracts embeddings, and saves to `profiles/my_clinic.json`.

Requires `HF_TOKEN` environment variable and pyannote model access.

Calibration only influences internal `speaker` IDs (`spk_0`, `spk_1`, ...). Human-readable labels are handled separately by the tagging layer.

Configuration:

```yaml
diarization:
  calibration_profile: null   # profile name from profiles/ (without .json)
  calibration_similarity_threshold: 0.72
  calibration_similarity_margin: 0.05
  calibration_debug: false
```

> **Note:** Phase 1 provides the matching infrastructure only. Real audio embedding extraction is not yet implemented — turns must already contain an `"embedding"` field for matching to occur.

### Calibration diagnostics (Phase 2D)

When calibration runs, a JSON report is always written next to the calibrated diarization file:

```
outputs/diarization_<ts>.calibration_report.json
```

The report contains the full similarity matrix (each cluster vs each profile speaker), per-cluster assignment decisions (best/second-best scores, margin, pass/fail reason), and the final mapping.

To also print a human-readable summary to the console, use `--calibration-debug`:

```bash
python -m app.main --calibration-debug
```

Example console output:

```
[CAL] Profile: my_clinic threshold=0.72 margin=0.05
[CAL] spk_0: best Speaker A=0.81 second Speaker B=0.65 margin=0.16 -> ASSIGNED spk_0->spk_0
[CAL] spk_1: best Speaker B=0.84 second Speaker A=0.71 margin=0.13 -> ASSIGNED spk_1->spk_1
[CAL] mapping: {"spk_0":"spk_0","spk_1":"spk_1"}
```

This can also be enabled via config YAML (`diarization.calibration_debug: true`).

### Calibration performance tuning

To reduce embedding compute time, very short diarization turns can be skipped during calibration. Skipped turns receive no embedding and are excluded from cluster means, but still appear in the output with their original speaker ID.

```yaml
diarization:
  calibration_min_turn_duration_sec: 0.5  # skip turns < 0.5s (default: 0.0 = embed all)
```

The pyannote embedding model is cached as a module-level singleton, so it is loaded only once per process even if calibration runs multiple times.

### Calibration robustness guards

Three opt-in guards filter out unreliable clusters before calibration assignment. All are disabled by default (no behavior change unless explicitly configured).

| Setting | Default | What it does |
|---------|---------|--------------|
| `diarization.calibration_min_cluster_turns` | `0` | Clusters with fewer embedded turns than this are excluded from assignment |
| `diarization.calibration_min_cluster_voiced_sec` | `0.0` | Clusters with less total embedded duration (seconds) than this are excluded |
| `diarization.calibration_allow_partial_assignment` | `true` | When `false`, no overrides are applied unless *all* eligible clusters get assigned |

Ineligible clusters keep their original `spk_N` IDs. When `allow_partial_assignment` is `false` and assignment is incomplete, all clusters keep their original IDs — this prevents half-mapped results in ambiguous sessions.

The calibration report (`calibration_report.json`) includes per-cluster stats (`turn_count`, `voiced_sec`), ineligibility reasons, and whether partial assignment was applied.

```yaml
diarization:
  calibration_min_cluster_turns: 3        # require at least 3 embedded turns
  calibration_min_cluster_voiced_sec: 2.0  # require at least 2s of voiced audio
  calibration_allow_partial_assignment: false  # all-or-nothing assignment
```

### Feature Flags (A/B Testing)

Individual calibration pipeline components can be toggled for A/B testing. All default to `true` (current behavior preserved).

```yaml
diarization:
  calibration_enabled: true           # master switch for calibration
  overlap_stabilizer_enabled: true    # detect and freeze overlapping turns
  prototype_matching_enabled: true    # cluster-level prototype assignment
  min_duration_filter_enabled: true   # 1.2s minimum for embedding computation
```

| Flag | When `false` |
|------|-------------|
| `calibration_enabled` | Skip calibration entirely — keep diarization speaker ids |
| `overlap_stabilizer_enabled` | No overlap detection — all turns eligible for embedding |
| `prototype_matching_enabled` | Skip speaker remapping — keep original spk_N ids |
| `min_duration_filter_enabled` | No 1.2s floor — embed all turns regardless of duration |

### Session Report

Each session produces a consolidated `session_report_<timestamp>.json` summarising config, feature flags, output file paths, and pipeline statistics.

When audio pre-check is enabled, the report includes an `audio_precheck` block with the computed metrics and any warnings. When pre-check is disabled or not run, this block is omitted.

```yaml
reporting:
  session_report_enabled: true   # default: true
```

### Audio Quality Pre-check

At session start, a short audio sample is recorded and evaluated before the main recording begins. This catches bad mic setup, excessive noise, or clipping before a long session is wasted.

Metrics computed:
- **Peak** and **RMS** level (dBFS)
- **Clipping rate** (fraction of samples at or above clip level)
- **SNR estimate** (95th vs 10th percentile frame RMS, clamped to 0–60 dB)

Warnings are printed if SNR, RMS, or clipping exceed configured thresholds. Results are persisted in the session report JSON.

```yaml
audio:
  precheck_enabled: true
  precheck_seconds: 4.0
```

```bash
# Run with pre-check (default)
python -m app.main --config config.yaml

# Disable pre-check for this session
python -m app.main --no-audio-precheck

# Override pre-check duration
python -m app.main --audio-precheck-seconds 8.0
```

See [`docs/CONFIG_REFERENCE.md`](docs/CONFIG_REFERENCE.md) for all `audio.precheck_*` fields.

### Subtitle Export (SRT / VTT)

Export diarized transcripts as subtitle files. Each cue is prefixed with the speaker label. Requires diarization to be enabled (`diarization.backend: pyannote`).

```bash
# Export SRT
python -m app.main --config config.yaml --export-srt

# Export VTT
python -m app.main --config config.yaml --export-vtt

# Both at once
python -m app.main --config config.yaml --export-srt --export-vtt
```

Output files: `subtitles_<timestamp>.srt` and/or `subtitles_<timestamp>.vtt` in the output directory. Paths are included in the session report.

Edge cases handled automatically:
- Empty or whitespace-only text segments are skipped
- Segments where end <= start are fixed (end = start + 0.01)
- Negative timestamps are clamped to 0

### Session Summary

Generate a human-readable Markdown summary from the session report. Includes configuration, feature flags, audio pre-check results (if present), diarization statistics, and output file paths.

```bash
# During a live session
python -m app.main --config config.yaml --export-summary

# From an existing session
python -m app.main --session 2026-03-01_10-00-00 --export-summary
```

Output file: `session_summary_<timestamp>.md` in the output directory. Uses the in-memory report dict during live sessions, or loads `session_report_<timestamp>.json` in standalone mode.

### Standalone Session Export

Export subtitles and summaries from existing sessions without re-recording. Uses `--session` with export flags. No audio device or ASR initialization occurs.

```bash
# Export SRT from an existing session
python -m app.main --session 2026-03-01_10-00-00 --export-srt

# Export VTT and summary together
python -m app.main --session 2026-03-01_10-00-00 --export-vtt --export-summary

# All exports at once
python -m app.main --session 2026-03-01_10-00-00 --export-srt --export-vtt --export-summary
```

Subtitle export reads `diarized_segments_<timestamp>.json` and `normalized_<timestamp>_<model>.json`, joining segments by `seg_id` for text lookup. Summary export reads `session_report_<timestamp>.json`.

If required files are missing, a clear error is printed and the process exits with a non-zero code.

### Lexicons

Lexicons live in `resources/lexicons/<language>/`:

```
resources/lexicons/
  da/
    custom.json    # Highest priority — site-specific replacements
    medical.json   # Medical terminology
    general.json   # General language corrections
  sv/
    ...
  en/
    ...
```

Each file uses this format:

```json
{
  "replacements": {
    "from_text": "to_text"
  }
}
```

Priority order: `custom` > `medical` > `general`. Exact matches are applied first, then fuzzy matches above the configured threshold (default: 0.92).

#### Lexicon Management CLI

Add, update, remove, and list custom lexicon terms from the command line. All operations target the `custom.json` file (highest priority). No audio or ASR initialization occurs.

```bash
# Add a new term
python -m app.main --language en --add-term "pt=patient"

# Update an existing term
python -m app.main --language en --add-term "pt=Patient (updated)"

# Remove a term
python -m app.main --language en --remove-term pt

# List all custom terms (sorted alphabetically)
python -m app.main --language en --list-terms

# Combine add and list
python -m app.main --language da --add-term "afd=afdeling" --list-terms
```

Terms are validated: both FROM and TO must be non-empty and must not have leading or trailing whitespace. Multi-word terms are supported (e.g. `--add-term "blood presure=blood pressure"`).

### Extractor Vocabularies

The clinical note extractors use keyword lists for symptom and medication detection. These vocabularies are loaded from external JSON files, making them easy to extend without modifying code.

Vocabulary files live in `resources/extractors/`:

```
resources/extractors/
  symptoms.json      # Symptom keywords (e.g. "headache", "chest pain")
  medications.json   # Medication keywords (e.g. "ibuprofen", "metformin")
```

Each file is a JSON array of lowercase strings:

```json
[
    "headache",
    "nausea",
    "shortness of breath"
]
```

To add custom terms, edit the JSON files directly. The extractors pick up changes on next import (process restart). If a file is missing, built-in defaults are used automatically. Invalid JSON triggers a warning to stderr and falls back to defaults.

### Clinical Note Export

Generate structured clinical notes from normalized transcripts using YAML templates. No ML or LLM — extraction is purely deterministic (regex + keyword matching).

```bash
# Export SOAP note (default template)
python -m app.main --config config.yaml --export-clinical-note

# Use a specific template
python -m app.main --config config.yaml --export-clinical-note --note-template soap
python -m app.main --config config.yaml --export-clinical-note --note-template isbar
python -m app.main --config config.yaml --export-clinical-note --note-template referral
python -m app.main --config config.yaml --export-clinical-note --note-template summary

# From an existing session
python -m app.main --session 2026-03-01_10-00-00 --export-clinical-note

# Reprocess with clinical note
python -m app.main --reprocess 2026-03-01_10-00-00 --export-clinical-note

# Export multiple notes at once (comma-separated template IDs)
python -m app.main --config config.yaml --export-notes soap,isbar,summary

# From an existing session
python -m app.main --session 2026-03-01_10-00-00 --export-notes soap,referral

# Reprocess with multiple notes
python -m app.main --reprocess 2026-03-01_10-00-00 --export-notes soap,isbar
```

`--export-notes` generates multiple clinical notes in a single pass. Speaker role detection runs at most once, even when multiple templates need it. Duplicate template IDs are ignored. Can be combined with `--export-clinical-note` (they are independent).

Output file: `clinical_note_<timestamp>_<template>.md` (or `.txt` for text-format templates).

#### Templates

Templates are YAML files in the `templates/` directory. Each template defines named sections with optional extractors and speaker scope.

Included templates:

| Template | Description |
|----------|-------------|
| `soap` | SOAP note — Subjective, Objective, Assessment, Plan (default) |
| `isbar` | ISBAR clinical handover — Identification, Situation, Background, Assessment, Recommendation |
| `referral` | Specialist referral — Nordic clinical style with Danish/English section names |
| `summary` | Short consultation summary — Problem, Patient history, Assessment, Plan |

The `soap.yaml` template:

```yaml
name: SOAP Note
format: markdown
sections:
  - title: Subjective
    extractors: [symptoms, negations, durations]
    scope: patient_only
  - title: Objective
    extractors: []
    scope: all
  - title: Assessment
    extractors: [symptoms, negations]
    scope: all
  - title: Plan
    extractors: [medications, durations]
    scope: clinician_only
transcript_section: true
```

**Template fields:**

| Field | Description |
|-------|-------------|
| `name` | Display title for the note |
| `format` | Output format: `markdown` or `text` |
| `sections` | List of sections (each with `title`, optional `extractors`, optional `scope`) |
| `transcript_section` | If `true`, append a timestamped transcript at the end |

**Section scope** controls which speakers' text is used for extraction:

| Scope | Description |
|-------|-------------|
| `all` | Use all speakers (default) |
| `patient_only` | Only text from speakers detected as patients |
| `clinician_only` | Only text from speakers detected as clinicians |

Scoped sections use **soft scoping**: if scoped extraction yields too few items, it supplements with all-segment extraction to avoid empty sections.

**Available extractors:** `symptoms`, `negations`, `durations`, `medications`. Sections with no extractors include raw transcript lines instead.

#### Speaker Role Detection

When a template uses scoped sections or `transcript_section`, speaker roles are detected automatically from the transcript text. Detection is deterministic — no ML or LLM.

Roles are assigned based on linguistic signal density:
- **Clinician signals:** questions ("how long", "do you"), directives ("I recommend", "prescribe"), medical terms ("examination", "diagnosis")
- **Patient signals:** first-person narrative ("I have", "I feel", "it hurts"), symptom descriptions ("my pain", "woke up")

Supports English, Danish, and Swedish signal patterns. Speakers with ambiguous or insufficient signals are marked as `unknown` and their text is included in all scopes.

To create custom templates, add a new `.yaml` file to `templates/` following the format above.

#### Symptom Timeline

Templates can include an optional symptom timeline section that pairs detected symptoms with their time expressions from the same segment. Enable it with the `show_symptom_timeline` flag in a template:

```yaml
show_symptom_timeline: true
```

When enabled, the note includes a "Symptom Timeline" section after any review flags:

```markdown
## Symptom Timeline
- headache — 3 days
- dizziness — since yesterday
- nausea
```

Symptoms without a time expression are listed without a dash. Time expressions include both numeric durations ("3 days", "2 weeks") and relative phrases ("since yesterday", "started today", "since last night", "for two weeks").

Deduplication is case-insensitive — the first occurrence of each symptom wins. The timeline is computed once and shared across all templates when using `--export-notes`.

#### Diagnostic Hints

Templates can include an optional diagnostic hints section that provides deterministic, rule-based diagnostic suggestions with SNOMED CT codes. Enable it with the `show_diagnostic_hints` flag in a template:

```yaml
show_diagnostic_hints: true
```

When enabled, the note includes a "Diagnostic Hints" section:

```markdown
## Diagnostic Hints
- Pneumonia (SNOMED: 233604007)
  Evidence: cough, fever, shortness of breath
- Pharyngitis (SNOMED: 405737000)
  Evidence: fever, sore throat
```

Matching rules:
- All required symptoms for a condition must be present in the transcript
- If a symptom is negated (e.g. "No fever", "Denies chest pain"), it is excluded — rules requiring that symptom will not trigger
- Results are sorted by number of evidence symptoms (most specific first), then alphabetically

Built-in conditions (10 rules): Pneumonia, Meningitis, Acute coronary syndrome, Migraine, Pharyngitis, Upper respiratory tract infection, Gastroenteritis, Arthritis, Vertigo, Generalised anxiety disorder.

No ML, no LLM, no external APIs — pure deterministic rule matching. Hints are computed once and shared across all templates when using `--export-notes`.

### Structured Clinical State

`build_clinical_state()` assembles all deterministic pipeline outputs into a single dictionary for programmatic access. This is an internal data layer — it does not change CLI behavior or note rendering.

```python
from app.clinical_state import build_clinical_state

state = build_clinical_state(segments, speaker_roles=roles, confidence_entries=conf)
```

Returned structure:

```python
{
    "symptoms": ["headache", "nausea"],
    "durations": ["3 days"],
    "negations": ["Denies fever", "No chest pain"],
    "medications": ["ibuprofen 400 mg"],
    "timeline": [
        {"symptom": "headache", "time_expression": "3 days",
         "seg_id": "seg_0001", "speaker_id": "spk_0", "t_start": 0.0}
    ],
    "review_flags": [
        {"type": "symptom_without_duration", "message": "...", "severity": "info"}
    ],
    "diagnostic_hints": [
        {"condition": "Migraine", "snomed_code": "37796009",
         "evidence": ["headache", "nausea", "vomiting"]}
    ],
    "speaker_roles": {"spk_0": {"role": "patient", ...}}
}
```

The module orchestrates existing extractors, review flags, symptom timeline, and diagnostic hints — no new extraction logic. The structure is designed to be extended with future fields (ICD mappings, objective findings, labs) without breaking consumers.

### Problem Representation

`build_problem_representation()` reads the structured clinical state and produces a formal problem representation that identifies the primary clinical problem and organises its attributes. This is a pure function — no I/O, no ML, no LLM.

```python
from app.problem_representation import build_problem_representation

pr = build_problem_representation(clinical_state)
```

The problem representation is automatically computed by `build_clinical_state()` and stored under `state["derived"]["problem_representation"]`, with `state["derived"]["problem_focus"]` set to the core symptom name (or `None`).

Returned structure:

```python
{
    "core_symptom": "headache",           # primary symptom (earliest by timeline)
    "severity": "severe",                 # from qualifiers for core symptom
    "duration": "3 days",                 # from timeline or durations list
    "onset": "sudden",                    # from qualifiers for core symptom
    "pattern": "constant",               # from qualifiers for core symptom
    "progression": "worsening",           # from qualifiers for core symptom
    "laterality": "left",                # from qualifiers for core symptom
    "radiation": "to left arm",           # from qualifiers for core symptom
    "associated_symptoms": ["nausea"],    # other symptoms (not core)
    "aggravating_factors": ["movement"],  # from qualifiers (core first, then all)
    "relieving_factors": ["rest"],        # from qualifiers (core first, then all)
    "pertinent_negatives": ["no fever"],  # from negations
    "timeline": [...],                    # full symptom timeline
    "diagnostic_hints": ["Migraine"],     # condition names from diagnostic hints
}
```

**Core symptom selection:** The symptom with the earliest `t_start` in the timeline is selected. If no timeline entries have a `t_start`, the first symptom from the symptoms list is used. If there are no symptoms, `core_symptom` is `None` and all other fields are empty.

**Qualifier fields** (severity, onset, pattern, progression, laterality, radiation) are populated from the qualifier entry matching the core symptom. If no match exists, all are `None`.

**Aggravating/relieving factors** prefer the core symptom's qualifier entry. If the core symptom has none for a given factor type, factors are collected from all qualifier entries (deduplicated, preserving order).

The `derived` key is compatible with the AI overlay — `apply_ai_overlay()` uses `setdefault("derived", {})` so it merges into the existing dict without overwriting the problem representation.

### Symptom Representations

`build_symptom_representations()` produces one representation per symptom with only the qualifiers that were actually detected for *that* symptom. Qualifiers are never inherited across symptoms — this prevents summaries that incorrectly imply associated symptoms share the core symptom's duration, severity, or other attributes.

```python
from app.problem_representation import build_symptom_representations

reps = build_symptom_representations(clinical_state)
```

Automatically computed by `build_clinical_state()` and stored under `state["derived"]["symptom_representations"]`.

Returned structure (one entry per symptom, preserving order):

```python
[
    {
        "symptom": "headache",
        "severity": "severe",           # from qualifiers for THIS symptom only
        "duration": "3 days",           # from timeline for THIS symptom only
        "onset": "sudden",
        "pattern": "constant",
        "progression": "worsening",
        "laterality": "left",
        "radiation": "to left arm",
        "aggravating_factors": ["movement"],
        "relieving_factors": ["rest"],
    },
    {
        "symptom": "nausea",
        "severity": "mild",            # nausea's own qualifier, not inherited from headache
        "duration": None,              # no timeline entry for nausea
        "onset": None,
        "pattern": "intermittent",
        "progression": None,
        "laterality": None,
        "radiation": None,
        "aggravating_factors": ["eating"],
        "relieving_factors": [],
    },
]
```

**Key design rule:** No qualifier fallback across symptoms. If a symptom has no detected qualifiers, its fields are `None` / empty lists — never populated from another symptom's data. This ensures the problem summary can safely describe the core symptom's attributes without leaking them to associated symptoms.

### Problem Summary

`summarize_problem()` produces a concise deterministic human-readable sentence describing the primary clinical problem. It uses only the core symptom's own attributes from `symptom_representations` — additional symptoms are mentioned by name only, never inheriting qualifiers from the core symptom.

```python
from app.problem_summary import summarize_problem

summary = summarize_problem(clinical_state)
```

Automatically computed by `build_clinical_state()` and stored under `state["derived"]["problem_summary"]`.

The sentence is built in this order (fields included only when present):

1. **severity** + **core symptom** (e.g. "Severe headache")
2. **duration** (e.g. "for 3 days")
3. **onset** (e.g. "sudden onset")
4. **pattern** (e.g. "constant")
5. **progression** (e.g. "worsening")
6. **laterality** (e.g. "left side")
7. **radiation** (e.g. "radiating to left arm")
8. **aggravating factors** (e.g. "worse with movement")
9. **relieving factors** (e.g. "relieved by rest")
10. **additional symptoms** (e.g. "with additional nausea, dizziness")

Example output:

```
"Severe headache, for 3 days, worsening, with additional nausea."
```

**Safety guarantee:** The summary never merges attributes across symptoms. A transcript with "severe headache for 3 days" and "nausea since this morning" will never produce "severe headache and nausea for 3 days" — nausea is listed by name only.

Returns an empty string if there is no core symptom. This is a read-only layer — it never replaces or modifies `symptoms`, `qualifiers`, `symptom_representations`, or `problem_representation`.

### Ontology Mapper

`map_symptoms_to_concepts()` maps extracted symptoms to standardised clinical concept entries using a curated file-based mapping. Currently maps to SNOMED CT codes. No ML, no LLM, no external API calls — exact case-insensitive lookup only.

```python
from app.ontology_mapper import load_ontology_map, map_symptoms_to_concepts

ontology = load_ontology_map()  # loads shipped symptoms_snomed.json
concepts = map_symptoms_to_concepts(clinical_state, ontology)
```

Automatically computed by `build_clinical_state()` and stored under `state["derived"]["ontology_concepts"]`.

Returned structure (one entry per mapped symptom, preserving order):

```python
[
    {
        "text": "headache",          # original symptom text
        "system": "SNOMED",          # coding system
        "code": "25064002",          # concept code
        "display": "Headache",       # standard display name
        "type": "symptom"            # concept type
    },
    {
        "text": "nausea",
        "system": "SNOMED",
        "code": "422587007",
        "display": "Nausea",
        "type": "symptom"
    }
]
```

The shipped mapping (`resources/ontology/symptoms_snomed.json`) covers 143 symptoms across 15 body-system categories (constitutional, neurologic, respiratory, cardiac, gastrointestinal, genitourinary, musculoskeletal, skin, psychiatric, endocrine/metabolic, ophthalmologic, ENT, gynecologic/obstetric, male reproductive, hematologic/lymphatic). Unknown symptoms are silently skipped (no fuzzy matching). Duplicate concepts (same code) are not repeated.

Currently maps symptoms only. Medications, procedures, and diagnoses are not yet mapped.

The mapping file can be extended by editing the JSON directly:

```json
{
    "headache": {
        "system": "SNOMED",
        "code": "25064002",
        "display": "Headache",
        "type": "symptom"
    }
}
```

If the mapping file is missing or contains invalid JSON, an empty map is used and a warning is printed to stderr.

### Clinical Pattern Matcher

`match_clinical_patterns()` detects common clinical patterns from structured symptom data using deterministic rule-based matching. It reads from `derived.symptom_representations` to keep qualifiers linked to the correct symptom — factors on one symptom never trigger patterns for another.

```python
from app.pattern_matcher import match_clinical_patterns

patterns = match_clinical_patterns(clinical_state)
```

Automatically computed by `build_clinical_state()` and stored under `state["derived"]["clinical_patterns"]`.

Returned structure:

```python
[
    {
        "pattern": "angina_like",
        "label": "Angina-like pattern",
        "evidence": ["chest pain", "aggravating factor: exertion", "relieving factor: rest"]
    }
]
```

**Initial pattern set (5 rules):**

| Pattern | Required | Optional |
|---------|----------|----------|
| `angina_like` | chest pain + aggravating: exertion/exercise/walking + relieving: rest | |
| `lower_respiratory_pattern` | cough + fever + shortness of breath/dyspnea | |
| `migraine_like` | headache + nausea | severity: severe |
| `urinary_irritative_pattern` | dysuria/painful urination + frequency/urinary frequency | |
| `gastroenteritis_like` | diarrhea + nausea/vomiting | fever |

**Factor isolation:** The angina pattern requires exertion and rest factors specifically on the chest pain symptom representation — exertion on a co-occurring headache will not trigger it. Similarly, migraine severity is checked on the headache representation only.

Evidence reflects the exact matched findings. No duplicate patterns are emitted. Multiple patterns can match simultaneously (e.g. migraine-like + gastroenteritis-like when headache, nausea, and diarrhea are all present).

No ML, no LLM, no external API calls. Additive only — does not modify any existing structured data.

### Temporal Normalizer

`normalize_timeline()` converts time expressions from the symptom timeline into standardised ISO formats. Pure function — no LLM, no external APIs.

```python
from app.temporal_normalizer import normalize_timeline
from datetime import datetime

normalized = normalize_timeline(timeline, reference_date=datetime(2026, 3, 8))
```

Automatically computed by `build_clinical_state()` and stored under `state["derived"]["normalized_timeline"]`.

Supported normalizations:
- Relative days: "today" → `2026-03-08`, "yesterday" → `2026-03-07`, "day before yesterday"
- Bare weekdays: "Monday", "Tuesday" → most recent past occurrence (never same day)
- "since <weekday>" expressions
- "last week" → 7 days back
- Durations: "3 days" → `P3D`, "2 weeks" → `P2W`, "5 hours" → `PT5H`, "3 months" → `P3M`

Unrecognised expressions return `None` — no guessing.

### Temporal Reasoner

`derive_temporal_context()` produces clinical onset ordering and progression tracking from the normalized timeline. Only explicit temporal evidence is used — mention order alone never establishes onset.

```python
from app.temporal_reasoner import derive_temporal_context

context = derive_temporal_context(clinical_state)
```

Automatically computed by `build_clinical_state()` and stored under `state["derived"]["temporal_context"]`.

Returned structure:

```python
{
    "clinical_onset_order": ["headache", "nausea"],   # ordered by ISO date evidence only
    "progression_events": [{"symptom": "headache", "progression": "worsening"}],
    "new_symptoms": ["nausea"],                        # strictly after earliest onset
    "temporal_uncertainty": ["nausea: duration P3D — no calendar date"],
}
```

**Key rule:** Do NOT infer clinical onset order from transcript mention order alone. Only ISO date strings establish ordering — durations produce uncertainty notes.

### Red Flag Detector

`detect_red_flags()` identifies high-risk clinical constellations from structured symptom data using deterministic rule-based matching.

```python
from app.red_flag_detector import detect_red_flags

flags = detect_red_flags(clinical_state)
```

Automatically computed by `build_clinical_state()` and stored under `state["derived"]["red_flags"]`.

**Built-in rules (5):**

| Flag | Evidence Required | Severity |
|------|------------------|----------|
| `sudden_severe_headache` | headache + severity:severe + onset:sudden | critical |
| `chest_pain_with_dyspnea` | chest pain + shortness of breath/dyspnea | critical |
| `hemoptysis_flag` | hemoptysis/coughing up blood | high |
| `suicidal_ideation_flag` | suicidal ideation/suicidal thoughts | critical |
| `systemic_malignancy_pattern` | weight loss + fatigue + night sweats | high |

Uses `derived.symptom_representations` for qualifier-linked matching. Additive only.

### Classification Router

Configurable classification system selector. Runs as an optional output controlled by `config.classification`.

```yaml
classification:
  enabled: false
  system: none   # none | icpc | icd10 | icd11
```

When enabled, stores result under `state["derived"]["classification"]`:

```python
{
    "system": "ICPC",   # or "ICD-10"
    "suggestions": [
        {"code": "N01", "label": "Headache", "kind": "symptom", "evidence": ["headache"]}
    ]
}
```

**ICPC-2:** 105 curated symptom mappings in `resources/classification/icpc_symptoms.json`.

**ICD-10:** ~95 symptom mappings + conservative pattern-level suggestions. Symptom-level codes are always emitted. Pattern-level codes (e.g. migraine G43.909) are only suggested when a matching clinical pattern has an explicit mapping in `resources/classification/icd10_patterns.json`.

**ICD-11:** Placeholder — returns empty suggestions.

Classification never modifies extraction results. It is an optional output layer only.

### FHIR Exporter

Deterministic FHIR Bundle export from structured clinical state. Produces interoperable healthcare resources without any external API calls.

```yaml
export:
  fhir_enabled: false
```

When enabled, stores result under `state["derived"]["fhir_bundle"]`:

```python
{
    "resourceType": "Bundle",
    "type": "collection",
    "entry": [
        {"resource": {"resourceType": "Encounter", ...}},
        {"resource": {"resourceType": "Observation", ...}},
        {"resource": {"resourceType": "Condition", ...}},
        {"resource": {"resourceType": "Composition", ...}},
    ]
}
```

**Resource types:**

| Resource | Source Data |
|----------|-----------|
| Encounter | Session metadata, problem_summary as reasonCode |
| Observation | One per symptom — SNOMED coding from ontology_concepts, qualifier components |
| Condition | Core symptom from problem_representation, classification suggestions as notes |
| Composition | Narrative sections: Chief Complaint, Clinical Summary, Red Flags, Classification |

All resource IDs are deterministic (SHA-256 hash of content). Verification status is always `provisional` — the system never overclaims diagnoses. Classification suggestions appear as notes on Condition, not as coding.

### Output Selector

Centralized control of optional output layers. Each is independently controllable via config:

```yaml
ai:
  enabled: false

classification:
  enabled: false
  system: none

export:
  fhir_enabled: false
```

The deterministic core pipeline always runs regardless of optional settings. Optional outputs run in order: classification → FHIR export → AI overlay. All outputs are additive — they never modify extraction results.

```python
from app.output_selector import apply_optional_outputs, should_run_ai_overlay

apply_optional_outputs(clinical_state, config)

if should_run_ai_overlay(config):
    # caller handles async LLM call
    ...
```

Supports both plain dict configs and AppConfig dataclass configs.

### AI Overlay (Optional)

Optional AI-generated clinical text from structured state. Fully optional — if disabled, the system behaves exactly as before. LLM output is an overlay only — it never modifies or replaces deterministic extraction results.

```yaml
ai:
  enabled: false
  provider: openai
  model: gpt-4.1-mini
  temperature: 0.2
```

```bash
python -m app.main --ai          # enable AI overlay
python -m app.main --no-ai       # disable AI overlay
```

Prompts are loaded from files in the `prompts/` directory — never hardcoded. Failures are logged and swallowed; the deterministic pipeline is never interrupted.

Overlay keys: `soap_draft`, `clinical_summary`, `follow_up_questions`, `problem_representation_refined`. Stored under `state["derived"]["ai_overlay"]` and `state["derived"]["ai_overlay_meta"]`.

### Session Export Bundle

Export all artifacts from a session into a single directory or ZIP archive for sharing, auditing, or archiving.

```bash
# Export as directory
python -m app.main --export-session 2026-03-01_10-00-00

# Export as ZIP archive
python -m app.main --export-session 2026-03-01_10-00-00 --zip
```

The bundle collects all files matching the session timestamp and copies them with simplified names:

| Original | Bundle name |
|----------|-------------|
| `raw_<ts>_<model>.json` | `raw_transcript.json` |
| `normalized_<ts>_<model>.json` | `normalized_transcript.json` |
| `confidence_report_<ts>.json` | `confidence_report.json` |
| `session_report_<ts>.json` | `session_report.json` |
| `clinical_note_<ts>_soap.md` | `clinical_note_soap.md` |
| `audio_<ts>.wav` | `audio.wav` |
| `diarized_segments_<ts>.json` | `diarized_segments.json` |
| `speaker_tags_<ts>.json` | `speaker_tags.json` |

Directory export creates `session_<ts>/` in the output directory. ZIP export creates `session_<ts>.zip`. Missing optional files are silently skipped. Original files are never modified.

### Confidence Report

Each session produces a `confidence_report_<timestamp>.json` that flags segments with low ASR quality. This is a diagnostic-only layer — RAW and normalized outputs are never modified.

Per-segment metrics from faster-whisper are checked against conservative thresholds:

| Metric | Threshold | Flag |
|--------|-----------|------|
| `no_speech_prob` | > 0.6 | `no_speech` — segment may be silence or noise |
| `avg_logprob` | < -1.0 | `low_confidence` — model uncertain about transcription |
| `compression_ratio` | > 2.4 | `repetitive` — possible hallucination or repeated text |

If all three metrics are missing (e.g., older faster-whisper version), the segment is flagged as `missing_metrics`. Segments with metrics within thresholds have an empty `flags` list.

The report includes all segments (not just flagged ones) so users can audit thresholds. The `flagged_count` and `total_count` fields provide a quick summary.

### Session Resume

Resume an interrupted session without losing data:

```bash
python -m app.main --resume 2026-02-28_14-30-00
```

Resume mode:
- Appends new RAW segments to the existing JSONL file (timestamps continue from where the session left off)
- Writes new audio as a part file (`audio_<ts>_part2.wav`), then concatenates all parts into a single WAV for pyannote
- Re-normalizes ALL segments (old + new) at session end to produce consistent derived outputs
- Re-runs diarization, smoothing, calibration, tagging, and confidence reporting on the full combined session

Safety checks before resume:
- RAW file must exist and contain valid JSONL with at least one segment
- ASR model must match the original session
- Timestamps must be monotonically non-decreasing with t0 <= t1 per segment
- If any check fails, resume is refused with a clear error message

Part files are never deleted — both parts and the concatenated WAV are preserved. RAW and normalized formats are unchanged.

`--resume` and `--session` are mutually exclusive.

### Reprocessing

Re-normalize and re-export existing sessions from their immutable RAW transcripts. Useful after updating lexicons, templates, or extractor vocabularies. Does **not** re-run ASR or diarization.

```bash
# Reprocess a single session
python -m app.main --reprocess 2026-03-01_10-00-00

# Reprocess with exports
python -m app.main --reprocess 2026-03-01_10-00-00 --export-srt --export-clinical-note
```

#### Batch reprocessing

Reprocess **all** sessions in the output directory in one command:

```bash
python -m app.main --reprocess-all

# With exports applied to every session
python -m app.main --reprocess-all --export-clinical-note --note-template soap
```

Sessions are processed oldest-first. If one session fails (corrupt RAW, no valid segments), processing continues with the next. A summary is printed at the end:

```
Found 12 sessions.
[1/12] 2026-03-01_10-00-00 ... OK
[2/12] 2026-03-01_14-30-00 ... FAILED: no valid segments in ...
...
Summary
  Found:     12
  Succeeded: 11
  Failed:    1
```

RAW transcripts are never modified — only derived outputs (normalized, changes, diarized segments, tags, reports, exports) are regenerated.

### Session Browser

Inspect past sessions without recording:

```bash
# List all sessions (newest first)
python -m app.main --list-sessions

# Show detailed info for a specific session
python -m app.main --show-session 2026-02-28_14-30-00
```

`--list-sessions` prints a table with timestamp, duration, segment count, model, language, and which companion files exist (audio, diarization, tags, normalization, confidence flags).

`--show-session` prints detailed info including file paths, speaker count (from diarization), confidence stats, and whether the session can be resumed.

Both are read-only — no files are created or modified. Corrupt RAW files are skipped with a warning to stderr.

`--list-sessions` and `--show-session` are mutually exclusive with each other and with `--session`.

---

## Configuration

All tunable parameters are in `config.yaml`. See [`docs/CONFIG_REFERENCE.md`](docs/CONFIG_REFERENCE.md) for complete configuration documentation with types, defaults, allowed values, and operational guidance.

```yaml
language: en

audio:
  device: null
  sample_rate: 16000
  channels: 1
  precheck_enabled: true
  precheck_seconds: 4.0

vad:
  speech_threshold_rms: 0.01
  short_silence_sec: 1.0
  long_silence_sec: 3.0
  chunk_duration_ms: 30
  min_speech_sec: 0.3

asr:
  model: large-v3
  device: auto
  compute_type: float16

diarization:
  enabled: true
  backend: default
  smoothing: true
  min_turn_sec: 0.7
  gap_merge_sec: 0.3
  calibration_profile: null
  calibration_similarity_threshold: 0.72
  calibration_similarity_margin: 0.05

normalization:
  enabled: true
  fuzzy_threshold: 0.92
  lexicon_dir: resources/lexicons

reporting:
  session_report_enabled: true

output_dir: outputs
```

---

## Synthetic Case Generator

A modular synthetic audio/scenario generator for end-to-end testing and benchmarking of the SCRIBE pipeline. Completely separate from the runtime transcription flow.

```bash
# Generate all cases (clean audio):
python -m tools.generate_synthetic_cases

# Generate a specific case:
python -m tools.generate_synthetic_cases --case chest_pain_consultation

# Telephone simulation:
python -m tools.generate_synthetic_cases --env telephone

# List available scenarios:
python -m tools.generate_synthetic_cases --list

# Generate and play back:
python -m tools.generate_synthetic_cases --case chest_pain_consultation --play
```

Each case generates:

| File | Content |
|------|---------|
| `audio.wav` | 16 kHz mono 16-bit PCM (SCRIBE-ready) |
| `transcript.txt` | Speaker-labeled reference transcript |
| `ground_truth.json` | Expected clinical facts for benchmarking |
| `meta.json` | Generation metadata (timestamps, config, segments) |

**Starter scenarios (3):**

| Case | Type | Theme | Turns |
|------|------|-------|-------|
| `chest_pain_consultation` | in_person | chest_pain | 15 |
| `cough_fever_telephone` | telephone_triage | cough_fever | 13 |
| `abdominal_pain_consultation` | in_person | abdominal_pain | 15 |

**Audio environment modes:** `clean`, `telephone` (bandpass 300–3400 Hz), `noisy` (additive Gaussian), `distance_near`, `distance_far` (attenuation + low-pass). Per-speaker environment overrides supported (e.g. patient far from mic).

Uses `pyttsx3` (Windows SAPI) for local offline TTS with male/female voices. Falls back to sine-tone placeholders if pyttsx3 is unavailable. All generation is deterministic given the same seed.

See [`tools/synthetic_cases/README.md`](tools/synthetic_cases/README.md) for full documentation on extending scenarios, voices, and running SCRIBE on generated files.

---

## Testing

```bash
python -m pytest tests/ -v
```

1488 tests covering WAV export, normalizer, diarization, segment cleaning, turn smoothing, speaker merge, segment relabeling, speaker tagging, calibration, confidence report, session resume, session browser, overlap stabilization, feature flags, audio quality pre-check, subtitle export, session summary, standalone export, lexicon management, extractor vocabularies, clinical note export, symptom timeline, diagnostic hints, clinical state, problem representation, symptom representations, problem summary, ontology mapper, pattern matcher, review flags, role detection, multi-note export, session export bundle, batch reprocessing, config validation, qualifier extraction, history extraction, live summary, temporal normalizer, temporal reasoner, red flag detector, classification router (ICPC/ICD-10), FHIR exporter, output selector, LLM overlay compatibility, synthetic case generator (scenarios, audio environments, TTS engine, case generation, CLI), and end-to-end integration.

---

## Troubleshooting

### Microphone not detected

1. Run `python -m app.main --list-audio-devices`
2. Find your microphone's index number
3. Set `audio.device` in `config.yaml` to that number
4. On Windows, check that the microphone is enabled in Sound settings

### No transcription output / very high latency

- Lower the ASR model size: `--model small` or `--model medium`
- Ensure CUDA is available if using a large model: `--device cuda`
- Adjust VAD thresholds — if `speech_threshold_rms` is too high, speech may not be detected. Try lowering it to `0.005`
- If `short_silence_sec` is too high, transcription appears delayed. Try `0.6`

### CUDA not available

- Verify NVIDIA drivers are installed
- Verify CUDA toolkit is installed and in your PATH
- Run: `python -c "import ctranslate2; print(ctranslate2.get_cuda_device_count())"`
- If 0: install the CUDA-enabled version of ctranslate2
- The system falls back to CPU automatically (`int8` quantization)

### Import errors

- Make sure you activated the virtual environment
- Run `pip install -r requirements.txt` again
- For sounddevice on Windows, you may need the PortAudio DLL (usually bundled)

### Audio clipping or distortion

- Move the microphone further from your mouth
- Lower the system microphone gain
- The VAD threshold may need adjustment for your setup
