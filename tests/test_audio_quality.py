"""Tests for audio quality pre-check metrics."""

from __future__ import annotations

import numpy as np
import pytest

from app.audio_quality import (
    DBFS_FLOOR,
    SNR_CAP,
    AudioPrecheckMetrics,
    compute_audio_precheck_metrics,
    detect_warnings,
    format_precheck_report,
)
from app.config import AppConfig, _build_config


# ── helpers ──────────────────────────────────────────────────────────

SR = 16000


class _Cfg:
    """Minimal config stub for precheck functions."""

    class audio:
        precheck_clip_level = 0.99
        precheck_frame_ms = 20
        precheck_snr_warn_db = 15.0
        precheck_rms_warn_dbfs = -45.0
        precheck_clip_warn_rate = 0.001


# ── compute metrics ──────────────────────────────────────────────────

class TestComputeMetrics:
    def test_silent_audio(self):
        audio = np.zeros(SR * 2, dtype=np.float32)
        m = compute_audio_precheck_metrics(audio, SR, _Cfg())
        assert m.duration_sec == pytest.approx(2.0)
        assert m.peak == 0.0
        assert m.peak_dbfs == DBFS_FLOOR
        assert m.rms == 0.0
        assert m.rms_dbfs == DBFS_FLOOR
        assert m.clipping_rate == 0.0
        assert m.snr_db == SNR_CAP  # noise <= 1e-8 → capped

    def test_clipped_audio(self):
        audio = np.ones(SR * 2, dtype=np.float32)
        m = compute_audio_precheck_metrics(audio, SR, _Cfg())
        assert m.peak == pytest.approx(1.0)
        assert m.clipping_rate > 0.0  # all samples >= 0.99
        assert m.rms == pytest.approx(1.0)

    def test_high_snr(self):
        """Alternating loud and quiet frames → high SNR."""
        frame_samples = SR * 20 // 1000  # 320 samples per frame
        n_frames = 200
        frames = []
        for i in range(n_frames):
            if i % 2 == 0:
                # Loud frame: amplitude 0.5
                frames.append(np.full(frame_samples, 0.5, dtype=np.float32))
            else:
                # Quiet frame: amplitude 0.001
                frames.append(np.full(frame_samples, 0.001, dtype=np.float32))
        audio = np.concatenate(frames)
        m = compute_audio_precheck_metrics(audio, SR, _Cfg())
        # P95 ≈ 0.5, P10 ≈ 0.001 → ~54 dB
        assert m.snr_db > 40.0
        assert m.snr_db <= SNR_CAP

    def test_low_snr(self):
        """Constant-level noise → SNR near 0."""
        rng = np.random.default_rng(42)
        audio = (rng.standard_normal(SR * 2) * 0.1).astype(np.float32)
        m = compute_audio_precheck_metrics(audio, SR, _Cfg())
        # All frames have similar RMS → P95 ≈ P10 → SNR ≈ 0
        assert m.snr_db < 5.0
        assert m.snr_db >= 0.0

    def test_short_audio_below_one_frame(self):
        """Audio shorter than one frame → snr_db = 0."""
        audio = np.array([0.5, 0.3, -0.2], dtype=np.float32)
        m = compute_audio_precheck_metrics(audio, SR, _Cfg())
        assert m.snr_db == 0.0
        assert m.peak == pytest.approx(0.5)


# ── detect warnings ─────────────────────────────────────────────────

class TestDetectWarnings:
    def test_no_warnings(self):
        m = AudioPrecheckMetrics(
            duration_sec=4.0, peak=0.5, peak_dbfs=-6.0,
            rms=0.1, rms_dbfs=-20.0, clipping_rate=0.0, snr_db=30.0,
        )
        assert detect_warnings(m, _Cfg()) == []

    def test_low_snr_warning(self):
        m = AudioPrecheckMetrics(
            duration_sec=4.0, peak=0.5, peak_dbfs=-6.0,
            rms=0.1, rms_dbfs=-20.0, clipping_rate=0.0, snr_db=10.0,
        )
        warnings = detect_warnings(m, _Cfg())
        assert len(warnings) == 1
        assert "SNR" in warnings[0]

    def test_low_rms_warning(self):
        m = AudioPrecheckMetrics(
            duration_sec=4.0, peak=0.01, peak_dbfs=-40.0,
            rms=0.001, rms_dbfs=-60.0, clipping_rate=0.0, snr_db=30.0,
        )
        warnings = detect_warnings(m, _Cfg())
        assert len(warnings) == 1
        assert "signal level" in warnings[0]

    def test_clipping_warning(self):
        m = AudioPrecheckMetrics(
            duration_sec=4.0, peak=1.0, peak_dbfs=0.0,
            rms=0.8, rms_dbfs=-1.9, clipping_rate=0.01, snr_db=30.0,
        )
        warnings = detect_warnings(m, _Cfg())
        assert len(warnings) == 1
        assert "Clipping" in warnings[0]

    def test_multiple_warnings(self):
        m = AudioPrecheckMetrics(
            duration_sec=4.0, peak=1.0, peak_dbfs=0.0,
            rms=0.001, rms_dbfs=-60.0, clipping_rate=0.05, snr_db=5.0,
        )
        warnings = detect_warnings(m, _Cfg())
        assert len(warnings) == 3


# ── format report ────────────────────────────────────────────────────

class TestFormatReport:
    def test_report_contains_metrics(self):
        m = AudioPrecheckMetrics(
            duration_sec=4.0, peak=0.5, peak_dbfs=-6.0,
            rms=0.1, rms_dbfs=-20.0, clipping_rate=0.0, snr_db=25.0,
        )
        report = format_precheck_report(m)
        assert "Peak" in report
        assert "RMS" in report
        assert "Clipping" in report
        assert "SNR" in report
        assert "4.0 s" in report


# ── config parsing ───────────────────────────────────────────────────

class TestConfig:
    def test_precheck_defaults(self):
        cfg = AppConfig()
        assert cfg.audio.precheck_enabled is True
        assert cfg.audio.precheck_seconds == 4.0
        assert cfg.audio.precheck_frame_ms == 20
        assert cfg.audio.precheck_snr_warn_db == 15.0
        assert cfg.audio.precheck_rms_warn_dbfs == -45.0
        assert cfg.audio.precheck_clip_warn_rate == 0.001
        assert cfg.audio.precheck_clip_level == 0.99

    def test_precheck_disabled_yaml(self):
        cfg = _build_config({"audio": {"precheck_enabled": False}})
        assert cfg.audio.precheck_enabled is False

    def test_precheck_seconds_override(self):
        cfg = _build_config({"audio": {"precheck_seconds": 8.0}})
        assert cfg.audio.precheck_seconds == 8.0
