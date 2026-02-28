# Configuration Reference

All configuration is defined in `config.yaml` at the project root. CLI flags override config values where applicable. This document provides complete documentation for every configuration field.

---

## `language`

| Property | Value |
|----------|-------|
| Type | `string` |
| Default | `en` |
| Allowed | `da`, `sv`, `en` |

The session language. Set once at session start. Controls which lexicon directory is used for normalization (`resources/lexicons/<language>/`) and is passed to the ASR model as a language hint.

**When to modify:** Set to match the primary spoken language of the session. There is no automatic language detection or mid-session switching.

---

## `audio`

### `audio.device`

| Property | Value |
|----------|-------|
| Type | `int` or `null` |
| Default | `null` |

The audio input device index. When `null`, the system default microphone is used. Use `--list-audio-devices` to find device indices.

**When to modify:** When the system default is not the intended microphone, or when multiple audio devices are connected.

### `audio.sample_rate`

| Property | Value |
|----------|-------|
| Type | `int` |
| Default | `16000` |

Sample rate in Hz. The ASR model (faster-whisper) expects 16 kHz mono audio. Changing this value will produce incorrect timestamps and degraded transcription quality.

**Safety note:** Do not change unless you have a specific reason and understand the downstream effects on VAD, ASR, and diarization.

### `audio.channels`

| Property | Value |
|----------|-------|
| Type | `int` |
| Default | `1` |

Number of audio channels. Must be `1` (mono). Stereo input is not supported.

### `audio.precheck_enabled`

| Property | Value |
|----------|-------|
| Type | `bool` |
| Default | `true` |

When `true`, a short audio quality pre-check runs at session start before recording begins. Evaluates peak level, RMS, clipping rate, and estimated SNR. Prints a summary and warnings if thresholds are exceeded.

### `audio.precheck_seconds`

| Property | Value |
|----------|-------|
| Type | `float` |
| Default | `4.0` |
| Unit | seconds |

Duration of audio captured for the pre-check evaluation.

**When to modify:** Increase for more stable SNR estimates in variable environments. Decrease if startup latency is a concern.

### `audio.precheck_frame_ms`

| Property | Value |
|----------|-------|
| Type | `int` |
| Default | `20` |
| Unit | milliseconds |

Frame duration used for per-frame RMS computation during SNR estimation. The 95th percentile frame RMS is treated as signal, the 10th percentile as noise.

### `audio.precheck_snr_warn_db`

| Property | Value |
|----------|-------|
| Type | `float` |
| Default | `15.0` |
| Unit | dB |

SNR threshold below which a warning is emitted. Sessions with SNR below this value may produce degraded transcription quality.

**When to modify:** Lower in consistently noisy environments where warnings are not actionable. Raise for clinical settings where audio quality is critical.

### `audio.precheck_rms_warn_dbfs`

| Property | Value |
|----------|-------|
| Type | `float` |
| Default | `-45.0` |
| Unit | dBFS |

RMS level threshold below which a "low signal level" warning is emitted. Indicates the microphone may be too far from the speaker or gain is too low.

### `audio.precheck_clip_warn_rate`

| Property | Value |
|----------|-------|
| Type | `float` |
| Default | `0.001` |

Fraction of samples at or above the clip level that triggers a clipping warning. A value of `0.001` means 0.1% of samples.

### `audio.precheck_clip_level`

| Property | Value |
|----------|-------|
| Type | `float` |
| Default | `0.99` |
| Range | `0.0` to `1.0` |

Amplitude threshold for clipping detection. Samples with `|x| >= clip_level` are counted as clipped.

---

## `vad`

Voice Activity Detection parameters. These control when speech is detected in the audio stream and how silence is interpreted.

### `vad.speech_threshold_rms`

| Property | Value |
|----------|-------|
| Type | `float` |
| Default | `0.01` |

RMS amplitude threshold for classifying an audio chunk as speech. Values below this are treated as silence.

**Operational impact:**
- **Increase:** Reduces sensitivity; quiet speech may not be detected. Useful in noisy environments.
- **Decrease:** Increases sensitivity; background noise may be classified as speech. Useful for soft-spoken speakers.

**When to modify:** If transcription is missing quiet passages (lower the threshold) or producing output from background noise (raise it). Typical range: `0.001` to `0.05`.

### `vad.short_silence_sec`

| Property | Value |
|----------|-------|
| Type | `float` |
| Default | `1.0` |

Duration of silence (in seconds) that triggers a sentence commit. When the speaker pauses for this long, the accumulated speech buffer is sent to ASR.

**Operational impact:**
- **Increase:** Longer pauses needed before transcription appears. Sentences may be longer.
- **Decrease:** Faster response time but may split mid-sentence on brief pauses.

**When to modify:** If transcription feels delayed (lower to `0.6`–`0.8`) or sentences are being split unnaturally (raise to `1.2`–`1.5`).

### `vad.long_silence_sec`

| Property | Value |
|----------|-------|
| Type | `float` |
| Default | `3.0` |

Duration of silence (in seconds) that triggers a paragraph break. The committer creates a new `paragraph_id` after this gap.

**Operational impact:**
- **Increase:** Fewer paragraph breaks; longer contiguous blocks.
- **Decrease:** More frequent paragraph breaks.

**When to modify:** Adjust based on conversation cadence. Clinical consultations with long pauses may benefit from a higher value (`4.0`–`5.0`).

### `vad.chunk_duration_ms`

| Property | Value |
|----------|-------|
| Type | `int` |
| Default | `30` |

Duration of each audio chunk in milliseconds, used for RMS computation. Smaller chunks provide finer-grained VAD at the cost of more processing overhead.

**When to modify:** Rarely. The default of 30 ms is standard for speech processing.

### `vad.min_speech_sec`

| Property | Value |
|----------|-------|
| Type | `float` |
| Default | `0.3` |

Minimum duration of speech (in seconds) required before a buffer is considered valid. Buffers shorter than this are discarded to avoid sending noise bursts to ASR.

**When to modify:** Raise if very short noise artifacts are being transcribed. Lower if short utterances ("yes", "no") are being dropped.

---

## `asr`

Automatic Speech Recognition parameters.

### `asr.model`

| Property | Value |
|----------|-------|
| Type | `string` |
| Default | `large-v3` |

The faster-whisper model name. Common options: `large-v3`, `medium`, `small`, `base`, `tiny`.

**Operational impact:**
- Larger models produce higher-quality transcriptions but require more VRAM and compute time.
- `large-v3` provides the best accuracy; `small` or `medium` are practical for CPU-only setups.

**When to modify:** If GPU memory is insufficient or latency is too high, downgrade to a smaller model.

### `asr.device`

| Property | Value |
|----------|-------|
| Type | `string` |
| Default | `auto` |
| Allowed | `auto`, `cuda`, `cpu` |

Compute device for ASR inference. When `auto`, the system checks `ctranslate2.get_cuda_device_count()` and uses CUDA if available, otherwise falls back to CPU.

**When to modify:** Force `cpu` if CUDA is unreliable or `cuda` to skip auto-detection.

### `asr.compute_type`

| Property | Value |
|----------|-------|
| Type | `string` |
| Default | `float16` |

Quantization type for the ASR model. `float16` is used on GPU; the system automatically falls back to `int8` on CPU.

**When to modify:** Rarely. The auto-fallback handles the common case. Use `int8` explicitly if you want to force CPU-optimised quantization on GPU.

---

## `diarization`

Speaker diarization parameters. Controls how speaker identity is determined for each audio segment.

### `diarization.enabled`

| Property | Value |
|----------|-------|
| Type | `bool` |
| Default | `true` |

Master switch for the diarization subsystem.

### `diarization.backend`

| Property | Value |
|----------|-------|
| Type | `string` |
| Default | `default` |
| Allowed | `default`, `pyannote` |

Diarization backend. `default` assigns all audio to `spk_0` (single speaker). `pyannote` runs post-session speaker diarization on the saved WAV file using pyannote.audio.

**Requirements for `pyannote`:**
- `HF_TOKEN` environment variable with a valid Hugging Face token.
- Accepted gated model licenses: `pyannote/speaker-diarization-3.1`, `pyannote/segmentation-3.0`.

### `diarization.smoothing`

| Property | Value |
|----------|-------|
| Type | `bool` |
| Default | `true` |

Enable post-diarization smoothing to reduce short speaker flips caused by backchannels.

### `diarization.min_turn_sec`

| Property | Value |
|----------|-------|
| Type | `float` |
| Default | `0.7` |
| Unit | seconds |

Turns shorter than this are merged into the nearest neighbor during smoothing. Same-speaker neighbors are preferred; otherwise the longer neighbor absorbs the short turn.

**Operational impact:**
- **Increase:** More aggressive smoothing; removes more short turns but may absorb genuine brief utterances.
- **Decrease:** Less aggressive; preserves short turns but may leave noisy speaker flips.

**When to modify:** For sessions with frequent backchannels ("mm", "ja"), increase to `1.0`–`1.5`. For multi-speaker meetings where short turns are genuine, decrease to `0.3`–`0.5`.

### `diarization.gap_merge_sec`

| Property | Value |
|----------|-------|
| Type | `float` |
| Default | `0.3` |
| Unit | seconds |

If two consecutive turns from the same speaker are separated by a gap this small or less, they are merged into one continuous turn.

**Operational impact:**
- **Increase:** More merging of same-speaker segments; may bridge across genuine speaker changes if set too high.
- **Decrease:** Less merging; preserves gaps.

### `diarization.calibration_profile`

| Property | Value |
|----------|-------|
| Type | `string` or `null` |
| Default | `null` |

Name of the calibration profile from `profiles/` (without `.json` extension). When set, the pipeline loads speaker embeddings from the profile and attempts to map diarization clusters to known speakers.

### `diarization.calibration_similarity_threshold`

| Property | Value |
|----------|-------|
| Type | `float` |
| Default | `0.72` |
| Range | `0.0` to `1.0` |

Minimum cosine similarity between a cluster prototype and a profile speaker embedding for assignment to occur.

**Safety note:** Setting too low (below `0.5`) may produce incorrect speaker assignments. Setting too high (above `0.85`) may leave most clusters unassigned as UNKNOWN.

**When to modify:** If calibration is producing too many UNKNOWN labels, lower cautiously. If wrong assignments appear, raise the threshold.

### `diarization.calibration_similarity_margin`

| Property | Value |
|----------|-------|
| Type | `float` |
| Default | `0.05` |

Minimum gap between the best and second-best cosine similarity scores for a cluster. If the margin is not met, the cluster is considered ambiguous and not assigned.

**Safety note:** Setting to `0.0` disables the margin check entirely and may cause incorrect assignments in sessions with similar-sounding speakers.

### `diarization.calibration_debug`

| Property | Value |
|----------|-------|
| Type | `bool` |
| Default | `false` |

When `true`, prints a human-readable calibration diagnostics summary to the console and writes a `calibration_report.json` file. Also available via `--calibration-debug` CLI flag.

### `diarization.calibration_min_turn_duration_sec`

| Property | Value |
|----------|-------|
| Type | `float` |
| Default | `0.0` |
| Unit | seconds |

Minimum diarization turn duration for embedding extraction. Turns shorter than this are skipped during `embed_turns()`. Note: when `min_duration_filter_enabled` is `true`, a floor of 1.2 seconds is enforced regardless of this value.

### `diarization.calibration_min_cluster_turns`

| Property | Value |
|----------|-------|
| Type | `int` |
| Default | `0` |

Minimum number of embedded turns required for a cluster to be eligible for calibration assignment. Clusters with fewer embedded turns are excluded.

**When to modify:** Set to `2`–`3` to prevent assignment based on a single noisy embedding.

### `diarization.calibration_min_cluster_voiced_sec`

| Property | Value |
|----------|-------|
| Type | `float` |
| Default | `0.0` |
| Unit | seconds |

Minimum total voiced duration (sum of embedded turn durations) required for a cluster to be eligible for calibration assignment.

**When to modify:** Set to `1.0`–`2.0` to prevent assignment for clusters with very little audio evidence.

### `diarization.calibration_allow_partial_assignment`

| Property | Value |
|----------|-------|
| Type | `bool` |
| Default | `true` |

When `true`, calibration applies assignments even if not all eligible clusters are matched. When `false`, no overrides are applied unless every eligible cluster gets assigned (all-or-nothing mode).

### `diarization.calibration_enabled`

| Property | Value |
|----------|-------|
| Type | `bool` |
| Default | `true` |

Master switch for calibration. When `false`, the entire calibration pipeline is skipped even if `calibration_profile` is set. Diarization speaker IDs are preserved as-is.

### `diarization.overlap_stabilizer_enabled`

| Property | Value |
|----------|-------|
| Type | `bool` |
| Default | `true` |

When `true`, overlapping diarization turns are detected and excluded from embedding computation and speaker remapping (freeze rule). When `false`, all turns are eligible for embedding and remapping (pre-overlap-stabilization behavior).

### `diarization.prototype_matching_enabled`

| Property | Value |
|----------|-------|
| Type | `bool` |
| Default | `true` |

When `true`, cluster-level prototype matching is used to assign speakers. When `false`, the assignment and override steps are skipped entirely — diarization speaker IDs (spk_0, spk_1, ...) are preserved unchanged. The calibration report is still written with an empty mapping and a note indicating prototype matching was disabled.

### `diarization.min_duration_filter_enabled`

| Property | Value |
|----------|-------|
| Type | `bool` |
| Default | `true` |

When `true`, a 1.2-second minimum duration floor is enforced for embedding computation and prototype building (via `MIN_PROTOTYPE_DURATION_SEC`). When `false`, the floor is not applied and the `calibration_min_turn_duration_sec` value is used directly.

---

## `normalization`

Lexicon-based text normalization parameters.

### `normalization.enabled`

| Property | Value |
|----------|-------|
| Type | `bool` |
| Default | `true` |

Master switch for normalization. When `false`, the normalized output is a copy of RAW text with no modifications.

### `normalization.fuzzy_threshold`

| Property | Value |
|----------|-------|
| Type | `float` |
| Default | `0.92` |
| Range | `0.0` to `1.0` |

Minimum similarity score (SequenceMatcher ratio) for fuzzy lexicon matching. Exact matches are always applied first; fuzzy matching is a fallback.

**Operational impact:**
- **Increase:** Fewer fuzzy matches; only very close misspellings are corrected.
- **Decrease:** More fuzzy matches; may produce false corrections.

**Safety note:** Values below `0.85` may produce incorrect replacements. The default of `0.92` is conservative and suitable for clinical use.

### `normalization.lexicon_dir`

| Property | Value |
|----------|-------|
| Type | `string` |
| Default | `resources/lexicons` |

Path to the lexicon directory. Expected structure: `<lexicon_dir>/<language>/{custom,medical,general}.json`. Priority order: `custom` > `medical` > `general`.

---

## `reporting`

Session reporting parameters.

### `reporting.session_report_enabled`

| Property | Value |
|----------|-------|
| Type | `bool` |
| Default | `true` |

When `true`, a consolidated session report (`session_report_<timestamp>.json`) is written at the end of each recording session. The report contains a config snapshot, feature flag states, output file paths, and pipeline statistics.

---

## Feature Flag Testing Examples

These examples show minimal config overrides for A/B testing individual calibration components. All flags default to `true`; only set the flags you want to disable.

### Disable calibration entirely

Skips the entire calibration pipeline. Diarization speaker IDs (spk_0, spk_1, ...) are preserved as-is.

```yaml
diarization:
  calibration_enabled: false
```

### Disable prototype matching only

Embeddings are still computed and overlap detection still runs, but cluster-to-profile assignment is skipped. Speaker IDs remain unchanged (spk_0, spk_1, ...). The calibration report is still written with an empty mapping.

```yaml
diarization:
  prototype_matching_enabled: false
```

### Disable overlap stabilization

All turns are eligible for embedding computation and speaker remapping, including overlapping turns. Reverts to pre-overlap-stabilization behavior.

```yaml
diarization:
  overlap_stabilizer_enabled: false
```

---

## `output_dir`

| Property | Value |
|----------|-------|
| Type | `string` |
| Default | `outputs` |

Directory where all session output files are written. Created automatically if it does not exist.
