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

### Stopping

Press **Ctrl+C** to stop recording. The pipeline will flush any remaining audio, write final output files, and print a summary.

---

## Output Files

All outputs are written to the configured `output_dir` (default: `outputs/`).

A session produces five files:

| File | Format | Purpose |
|------|--------|---------|
| `raw_<timestamp>_<model>.json` | JSON Lines | Immutable RAW segments (one JSON object per line) |
| `raw_<timestamp>_<model>.txt` | Plain text | Human-readable RAW transcript |
| `normalized_<timestamp>_<model>.json` | JSON array | Normalized segments |
| `normalized_<timestamp>_<model>.txt` | Plain text | Human-readable normalized transcript |
| `changes_<timestamp>_<model>.json` | JSON array | Full audit log of every normalization change |

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
| `main.py` | Pipeline orchestration |

### Speaker Diarization

`speaker_id` is an acoustic-only label (e.g., `spk_0`, `spk_1`). It identifies *who* is speaking based on audio characteristics.

Speaker diarization:
- Runs BEFORE ASR
- NEVER modifies transcribed text
- NEVER assigns roles (patient, clinician, etc.)
- NEVER depends on language or content

The default backend assigns all audio to `spk_0`. The architecture is pluggable — replace `DefaultDiarizer` with a real backend (e.g., pyannote.audio) by subclassing `Diarizer` and registering it in `create_diarizer()`.

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
  backend: default

normalization:
  enabled: true
  fuzzy_threshold: 0.92
  lexicon_dir: resources/lexicons

output_dir: outputs
```

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
