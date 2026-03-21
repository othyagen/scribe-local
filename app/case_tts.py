"""TTS case execution — run text cases through the audio pipeline.

Provides two entry points:

  - :func:`case_to_audio` — synthesize case text to a WAV file (TTS only).
  - :func:`run_case_tts` — full TTS → ASR → clinical pipeline.

**Synthetic/test mode only.**  All outputs are tagged ``synthetic=True``
and must not be mixed with real patient recordings.

Separation of concerns:
  - TTS layer produces audio from text (no clinical logic).
  - ASR layer transcribes audio to segments (existing engine).
  - Clinical pipeline receives segments (identical to text mode).
  - No parallel reasoning path — same :func:`initialize_session`.

Pure orchestration — delegates to existing modules, adds no new
clinical logic.
"""

from __future__ import annotations

from pathlib import Path

from app.tts_provider import get_tts_provider, read_wav_float32
from app.clinical_session import initialize_session, get_app_view
from app.clinical_metrics import derive_clinical_metrics
from app.case_system import validate_case, _merge_config


def _tts_input_metadata(tts_result: dict | None) -> dict:
    """Build input_metadata for TTS-mode execution."""
    tts_info = None
    if tts_result and tts_result.get("success"):
        tts_info = {
            "provider": tts_result.get("provider", ""),
            "voice": tts_result.get("voice", ""),
        }
    return {"mode": "tts", "synthetic": True, "tts": tts_info}


# ── TTS synthesis ──────────────────────────────────────────────────


def case_to_audio(
    case: dict,
    output_dir: Path,
    *,
    provider: str = "edge",
    voice: str | None = None,
    lang: str | None = None,
) -> dict:
    """Synthesize case segment text into a WAV file.

    Joins all segment ``normalized_text`` fields into a single text
    block and delegates to the named TTS provider.  No ASR or clinical
    logic — pure audio generation.

    Args:
        case: parsed case dict with ``segments``.
        output_dir: directory for the output WAV.
        provider: TTS provider name (must be explicitly chosen).
        voice: optional provider-specific voice identifier.
        lang: optional language code hint.

    Returns:
        TTS result dict (see :func:`tts_provider._tts_result`).
    """
    segments = case.get("segments", [])
    text = " ".join(
        seg.get("normalized_text", "") for seg in segments
    ).strip()

    if not text:
        return {
            "audio_path": "",
            "provider": provider,
            "voice": voice or "",
            "text": "",
            "success": False,
            "error": "No text to synthesize",
            "synthetic": True,
        }

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    case_id = case.get("case_id", "unknown")
    wav_path = output_dir / f"tts_{case_id}.wav"

    tts = get_tts_provider(provider)
    return tts.synthesize(text, wav_path, voice=voice, lang=lang)


# ── full pipeline ──────────────────────────────────────────────────


def run_case_tts(
    case: dict,
    output_dir: Path,
    *,
    provider: str = "edge",
    voice: str | None = None,
    lang: str | None = None,
    asr_engine: object | None = None,
) -> dict:
    """Run a case through TTS → ASR → clinical pipeline.

    1. Synthesize case text to WAV via :func:`case_to_audio`.
    2. Load WAV as float32 numpy array.
    3. Transcribe via ASR engine.
    4. Build segments matching case_system format.
    5. Run through the same clinical session pipeline as text mode.
    6. Return result_bundle in the same shape as :func:`run_case`.

    Args:
        case: parsed case dict.
        output_dir: directory for TTS audio output.
        provider: TTS provider name (explicit — no silent fallback).
        voice: optional provider-specific voice identifier.
        lang: optional language code hint.
        asr_engine: optional pre-initialised ASR engine (avoids reload).

    Returns:
        Result bundle dict (same shape as ``run_case()``), with an
        additional ``tts_result`` key containing the TTS metadata.
    """
    validation = validate_case(case)
    if not validation["valid"]:
        return {
            "case_id": case.get("case_id", ""),
            "session": {},
            "app_view": {},
            "metrics": {},
            "ground_truth": case.get("ground_truth") or {},
            "validation": validation,
            "tts_result": None,
            "input_metadata": _tts_input_metadata(None),
        }

    # Step 1: TTS synthesis.
    tts_result = case_to_audio(
        case, output_dir, provider=provider, voice=voice, lang=lang,
    )

    if not tts_result.get("success"):
        return {
            "case_id": case.get("case_id", ""),
            "session": {},
            "app_view": {},
            "metrics": {},
            "ground_truth": case.get("ground_truth") or {},
            "validation": validation,
            "tts_result": tts_result,
            "input_metadata": _tts_input_metadata(tts_result),
        }

    # Step 2: Load audio.
    audio_path = Path(tts_result["audio_path"])
    audio, sample_rate = read_wav_float32(audio_path)

    # Resample to 16 kHz if needed.
    if sample_rate != 16000:
        import numpy as np

        ratio = 16000 / sample_rate
        new_length = int(len(audio) * ratio)
        indices = np.linspace(0, len(audio) - 1, new_length)
        audio = np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

    # Step 3: ASR transcription.
    engine = asr_engine
    if engine is None:
        from app.asr import ASREngine
        from app.config import load_config

        config = load_config()
        engine = ASREngine(config)

    asr_results = engine.transcribe(audio)

    # Step 4: Build segments.
    segments = []
    for i, result in enumerate(asr_results):
        segments.append({
            "seg_id": f"seg_{i + 1:04d}",
            "t0": result.start,
            "t1": result.end,
            "speaker_id": "spk_0",
            "normalized_text": result.text,
        })

    # Step 5: Clinical pipeline (identical path to text mode).
    config_dict = _merge_config(case)
    session = initialize_session(segments, config=config_dict)
    app_view = get_app_view(session)
    metrics = derive_clinical_metrics(session["clinical_state"])

    # Step 6: Result bundle (same shape as run_case + tts metadata).
    return {
        "case_id": case.get("case_id", ""),
        "session": session,
        "app_view": app_view,
        "metrics": metrics,
        "ground_truth": case.get("ground_truth") or {},
        "validation": validation,
        "tts_result": tts_result,
        "input_metadata": _tts_input_metadata(tts_result),
    }
