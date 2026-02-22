"""Configuration loading and dataclass definitions."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class AudioConfig:
    device: Optional[int] = None
    sample_rate: int = 16000
    channels: int = 1


@dataclass
class VadConfig:
    speech_threshold_rms: float = 0.01
    short_silence_sec: float = 1.0
    long_silence_sec: float = 3.0
    chunk_duration_ms: int = 30
    min_speech_sec: float = 0.3


@dataclass
class AsrConfig:
    model: str = "large-v3"
    device: str = "auto"
    compute_type: str = "float16"


@dataclass
class DiarizationConfig:
    enabled: bool = True
    backend: str = "default"
    smoothing: bool = True
    min_turn_sec: float = 0.7
    gap_merge_sec: float = 0.3
    calibration_profile: Optional[str] = None
    calibration_similarity_threshold: float = 0.72


@dataclass
class NormalizationConfig:
    enabled: bool = True
    fuzzy_threshold: float = 0.92
    lexicon_dir: str = "resources/lexicons"


@dataclass
class AppConfig:
    language: str = "en"
    output_dir: str = "outputs"
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VadConfig = field(default_factory=VadConfig)
    asr: AsrConfig = field(default_factory=AsrConfig)
    diarization: DiarizationConfig = field(default_factory=DiarizationConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)


def load_config(path: str) -> AppConfig:
    """Load configuration from a YAML file."""
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return _build_config(data)


def _build_config(data: dict) -> AppConfig:
    return AppConfig(
        language=data.get("language", "en"),
        output_dir=data.get("output_dir", "outputs"),
        audio=_build_audio(data.get("audio", {})),
        vad=_build_vad(data.get("vad", {})),
        asr=_build_asr(data.get("asr", {})),
        diarization=_build_diarization(data.get("diarization", {})),
        normalization=_build_normalization(data.get("normalization", {})),
    )


def _build_audio(d: dict) -> AudioConfig:
    return AudioConfig(
        device=d.get("device"),
        sample_rate=d.get("sample_rate", 16000),
        channels=d.get("channels", 1),
    )


def _build_vad(d: dict) -> VadConfig:
    return VadConfig(
        speech_threshold_rms=d.get("speech_threshold_rms", 0.01),
        short_silence_sec=d.get("short_silence_sec", 1.0),
        long_silence_sec=d.get("long_silence_sec", 3.0),
        chunk_duration_ms=d.get("chunk_duration_ms", 30),
        min_speech_sec=d.get("min_speech_sec", 0.3),
    )


def _build_asr(d: dict) -> AsrConfig:
    return AsrConfig(
        model=d.get("model", "large-v3"),
        device=d.get("device", "auto"),
        compute_type=d.get("compute_type", "float16"),
    )


def _build_diarization(d: dict) -> DiarizationConfig:
    return DiarizationConfig(
        enabled=d.get("enabled", True),
        backend=d.get("backend", "default"),
        smoothing=d.get("smoothing", True),
        min_turn_sec=d.get("min_turn_sec", 0.7),
        gap_merge_sec=d.get("gap_merge_sec", 0.3),
        calibration_profile=d.get("calibration_profile"),
        calibration_similarity_threshold=d.get(
            "calibration_similarity_threshold", 0.72
        ),
    )


def _build_normalization(d: dict) -> NormalizationConfig:
    return NormalizationConfig(
        enabled=d.get("enabled", True),
        fuzzy_threshold=d.get("fuzzy_threshold", 0.92),
        lexicon_dir=d.get("lexicon_dir", "resources/lexicons"),
    )


def apply_cli_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    """Merge CLI arguments into the loaded config (CLI wins)."""
    if args.language:
        config.language = args.language
    if args.model:
        config.asr.model = args.model
    if args.device:
        config.asr.device = args.device
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.vad_speech_threshold is not None:
        config.vad.speech_threshold_rms = args.vad_speech_threshold
    if args.vad_short_silence is not None:
        config.vad.short_silence_sec = args.vad_short_silence
    if args.vad_long_silence is not None:
        config.vad.long_silence_sec = args.vad_long_silence
    return config


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    p = argparse.ArgumentParser(
        prog="scribe-local",
        description="Privacy-first local AI scribe with streaming ASR.",
    )
    p.add_argument("--config", type=str, default="config.yaml",
                    help="Path to YAML config file (default: config.yaml)")
    p.add_argument("--language", choices=["da", "sv", "en"],
                    help="Session language (overrides config)")
    p.add_argument("--model", type=str,
                    help="ASR model name (overrides config)")
    p.add_argument("--device", choices=["cuda", "cpu"],
                    help="Compute device (overrides config)")
    p.add_argument("--output-dir", type=str,
                    help="Output directory (overrides config)")
    p.add_argument("--vad-speech-threshold", type=float,
                    help="VAD RMS speech threshold (overrides config)")
    p.add_argument("--vad-short-silence", type=float,
                    help="Short silence duration in seconds (overrides config)")
    p.add_argument("--vad-long-silence", type=float,
                    help="Long silence / paragraph break in seconds (overrides config)")
    p.add_argument("--list-audio-devices", action="store_true",
                    help="List available audio input devices and exit")

    # Speaker tagging
    p.add_argument("--auto-tags", choices=["none", "alphabetical", "index"],
                    default="none",
                    help="Auto-assign speaker tags (default: none)")
    p.add_argument("--set-tag", action="append", default=[],
                    metavar="SPK=TAG",
                    help="Set speaker tag, repeatable (e.g. --set-tag spk_0=Me)")
    p.add_argument("--set-label", action="append", default=[],
                    metavar="SPK=LABEL",
                    help="Set speaker label, repeatable (e.g. --set-label spk_0=Mette)")
    p.add_argument("--session", type=str, default=None,
                    metavar="TIMESTAMP",
                    help="Session timestamp for standalone tag/merge operations")

    # Speaker merge
    p.add_argument("--merge", action="append", default=[],
                    metavar="SPK=TARGET",
                    help="Merge speaker into target, repeatable (e.g. --merge spk_2=spk_0)")

    # Calibration profile creation
    p.add_argument("--create-profile", type=str, default=None,
                    metavar="NAME", help="Create a calibration profile and exit")
    p.add_argument("--profile-speakers", type=int, default=2,
                    help="Number of speakers to record (default: 2)")
    p.add_argument("--profile-duration", type=float, default=12.0,
                    help="Recording duration per speaker in seconds (default: 12)")
    return p
