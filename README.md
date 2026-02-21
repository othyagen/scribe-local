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
| `--session TIMESTAMP` | Session timestamp for standalone tag operations |

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
| `speaker_tags_<timestamp>.json` | JSON | Speaker tag/label mapping |
| `tag_labeled_<timestamp>.txt` | Plain text | Transcript with human-readable speaker tags |

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

#### Segment relabeling

After diarization, each ASR segment is relabeled with the speaker who has the largest time overlap. This produces `diarized_segments_<timestamp>.json` (relabeling map) and `diarized_<timestamp>.txt` (transcript with new speaker_ids). RAW and normalized outputs are never modified.

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

---

## Configuration

All tunable parameters are in `config.yaml`. No magic numbers in code.

```yaml
language: en          # da, sv, en

audio:
  device: null        # null = system default, or device index
  sample_rate: 16000
  channels: 1

vad:
  speech_threshold_rms: 0.01
  short_silence_sec: 1.0
  long_silence_sec: 3.0
  chunk_duration_ms: 30
  min_speech_sec: 0.3

asr:
  model: large-v3
  device: auto        # auto, cuda, cpu
  compute_type: float16

diarization:
  enabled: true
  backend: default    # default or pyannote

normalization:
  enabled: true
  fuzzy_threshold: 0.92
  lexicon_dir: resources/lexicons

output_dir: outputs
```

---

## Testing

```bash
python -m pytest tests/ -v
```

94 tests covering WAV export, normalizer (exact/fuzzy/phrase matching, domain priority, edge cases), diarization (DefaultDiarizer, factory, pyannote pipeline with mocks), segment relabeling (overlap assignment, output formats), speaker tagging (auto-tags, manual set-tag/set-label, CLI parsing, tagged transcript generation), and end-to-end integration (full pipeline without live microphone).

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
