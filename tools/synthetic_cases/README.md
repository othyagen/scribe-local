# Synthetic Clinical Case Generator

Deterministic synthetic audio/scenario generator for testing and benchmarking the SCRIBE pipeline end-to-end.

## Quick Start

```bash
# Generate all cases (clean audio):
python -m tools.generate_synthetic_cases

# Generate a specific case:
python -m tools.generate_synthetic_cases --case chest_pain_consultation

# List available scenarios:
python -m tools.generate_synthetic_cases --list
```

## Quick Start: Testing SCRIBE with Synthetic Cases

**1. Generate a case and inspect the output:**

```bash
python -m tools.generate_synthetic_cases --case chest_pain_consultation
python -m tools.generate_synthetic_cases --case chest_pain_consultation --show
```

**2. Listen to the generated audio:**

```bash
python -m tools.generate_synthetic_cases --case chest_pain_consultation --play
```

**3. Run SCRIBE extraction on the reference transcript (no ASR needed):**

```python
import json
from app.clinical_state import build_clinical_state

meta = json.loads(open("test_data/synthetic/chest_pain_consultation/meta.json").read())
segments = [
    {"seg_id": f"seg_{i:04d}", "t0": s["t0"], "t1": s["t1"],
     "speaker_id": s["speaker_id"], "normalized_text": s["text"]}
    for i, s in enumerate(meta["segments"])
]
state = build_clinical_state(segments)
print("Symptoms:", state["symptoms"])
print("Negations:", state["negations"])
```

**4. Compare against ground truth:**

```bash
type test_data/synthetic/chest_pain_consultation/ground_truth.json
```

**5. Try different audio environments:**

```bash
python -m tools.generate_synthetic_cases --case chest_pain_consultation --env telephone
python -m tools.generate_synthetic_cases --case chest_pain_consultation --env noisy
python -m tools.generate_synthetic_cases --patient-distance far
```

**6. Learn more about the workflow:**

```bash
python -m tools.generate_synthetic_cases --explain
```

## Output Structure

Each case generates four files:

```
test_data/synthetic/<case_id>/
  audio.wav          # 16 kHz mono 16-bit PCM (SCRIBE-ready)
  transcript.txt     # Speaker-labeled reference transcript
  ground_truth.json  # Expected clinical facts for benchmarking
  meta.json          # Generation metadata (timestamps, config, segments)
```

## Audio Environment Modes

```bash
# Clean room (default):
python -m tools.generate_synthetic_cases --env clean

# Telephone simulation (300-3400 Hz bandpass):
python -m tools.generate_synthetic_cases --env telephone

# Noisy room (additive Gaussian noise):
python -m tools.generate_synthetic_cases --env noisy --noise-level 0.01

# Distance simulation (far mic placement):
python -m tools.generate_synthetic_cases --env distance_far

# Patient far from mic, doctor near:
python -m tools.generate_synthetic_cases --patient-distance far
```

## Playback

```bash
# Generate and play back:
python -m tools.generate_synthetic_cases --case chest_pain_consultation --play
```

## Running SCRIBE on Generated Files

The generated `audio.wav` files are standard 16 kHz mono PCM, compatible with SCRIBE's input format. To test the pipeline:

```python
# Example: run clinical_state on the reference transcript
import json
from app.clinical_state import build_clinical_state

# Load the ground truth transcript segments
meta = json.loads(open("test_data/synthetic/chest_pain_consultation/meta.json").read())
segments = [
    {
        "seg_id": f"seg_{i:04d}",
        "t0": s["t0"],
        "t1": s["t1"],
        "speaker_id": s["speaker_id"],
        "normalized_text": s["text"],
    }
    for i, s in enumerate(meta["segments"])
]

state = build_clinical_state(segments)
print("Symptoms:", state["symptoms"])
print("Negations:", state["negations"])
print("Medications:", state["medications"])
```

## Available Scenarios

| Case ID | Type | Theme | Turns |
|---------|------|-------|-------|
| `chest_pain_consultation` | in_person_consultation | chest_pain | 15 |
| `cough_fever_telephone` | telephone_triage | cough_fever | 13 |
| `abdominal_pain_consultation` | in_person_consultation | abdominal_pain | 15 |

## Configuration Options

| Flag | Default | Description |
|------|---------|-------------|
| `--case` | all | Specific case ID to generate |
| `--output-dir` | `test_data/synthetic` | Output directory |
| `--env` | `clean` | Audio environment mode |
| `--noise-level` | `0.005` | Noise RMS for noisy mode |
| `--patient-distance` | none | Patient mic distance (near/far) |
| `--rate` | `160` | TTS speaking rate (WPM) |
| `--pause` | `0.8` | Pause between turns (seconds) |
| `--seed` | `42` | Random seed for reproducibility |
| `--play` | off | Play back after generation |
| `--show` | off | Show case folder contents and ground truth summary |
| `--open` | off | Open case folder in system file browser |
| `--explain` | off | Print step-by-step workflow guide |

## Extending

### Adding a new scenario

Edit `tools/synthetic_cases/scenarios.py` and register a new scenario:

```python
_register("new_case_id", {
    "encounter_type": "in_person_consultation",
    "theme": "new_theme",
    "participants": [_doctor("spk_0"), _patient("spk_1")],
    "dialogue": [
        {"speaker_id": "spk_0", "text": "Doctor's line."},
        {"speaker_id": "spk_1", "text": "Patient's line."},
    ],
    "ground_truth": {
        "symptoms": ["symptom1"],
        "negations": [],
        "medications": [],
        "durations": [],
        "qualifiers": [],
        "speaker_roles": {"spk_0": "clinician", "spk_1": "patient"},
        "expected_patterns": [],
        "expected_red_flags": [],
    },
})
```

### Adding more speakers

Participants are a list — add a third speaker:

```python
"participants": [
    _doctor("spk_0"),
    _patient("spk_1"),
    {"speaker_id": "spk_2", "role": "consultant", "label": "Consultant", "voice_hint": "male"},
],
```

### Custom voice configuration

Override voices programmatically:

```python
from tools.synthetic_cases.tts_engine import VoiceConfig
from tools.synthetic_cases.generator import GeneratorConfig

config = GeneratorConfig(
    voice_overrides={
        "spk_0": VoiceConfig(rate=140, volume=0.9),  # slower doctor
        "spk_1": VoiceConfig(rate=180, volume=1.0),   # faster patient
    },
)
```

## Dependencies

- `pyttsx3` — local TTS via Windows SAPI (install: `pip install pyttsx3`)
- `scipy` — audio filtering (already in SCRIBE deps)
- `numpy` — audio processing (already in SCRIBE deps)

If `pyttsx3` is unavailable, the generator falls back to sine-tone placeholders so the framework still works for structural testing.

## Tests

```bash
python -m pytest tests/test_synthetic_cases.py -v
```

55 tests covering scenarios, audio environments, TTS engine, case generation, and CLI.
