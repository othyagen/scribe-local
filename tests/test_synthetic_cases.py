"""Tests for the synthetic clinical case generator.

Tests cover scenario definitions, audio environment simulation,
TTS engine fallback, case generation, and ground truth structure.

Tests that require pyttsx3 (actual TTS) are marked with
``@pytest.mark.tts`` and skipped if the engine is unavailable.
"""

from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from tools.synthetic_cases.scenarios import (
    SCENARIOS,
    get_scenario,
    list_scenarios,
)
from tools.synthetic_cases.audio_env import (
    AudioEnvConfig,
    TARGET_RATE,
    apply_environment,
)
from tools.synthetic_cases.tts_engine import (
    VoiceConfig,
    synthesize_utterance,
    voice_config_from_hint,
    _synthesize_placeholder,
)
from tools.synthetic_cases.generator import (
    GeneratorConfig,
    generate_case,
)


# ── helpers ─────────────────────────────────────────────────────


def _has_pyttsx3() -> bool:
    try:
        import pyttsx3
        engine = pyttsx3.init()
        return True
    except Exception:
        return False


_TTS_AVAILABLE = _has_pyttsx3()


# ══════════════════════════════════════════════════════════════════
# Scenario definitions
# ══════════════════════════════════════════════════════════════════


class TestScenarios:
    def test_three_scenarios_registered(self):
        assert len(SCENARIOS) >= 3

    def test_list_scenarios_sorted(self):
        ids = list_scenarios()
        assert ids == sorted(ids)

    def test_get_scenario_valid(self):
        s = get_scenario("chest_pain_consultation")
        assert s["theme"] == "chest_pain"
        assert s["encounter_type"] == "in_person_consultation"

    def test_get_scenario_invalid(self):
        with pytest.raises(KeyError):
            get_scenario("nonexistent_case")

    @pytest.mark.parametrize("case_id", list_scenarios())
    def test_scenario_has_required_keys(self, case_id):
        s = SCENARIOS[case_id]
        assert "case_id" in s
        assert "encounter_type" in s
        assert "theme" in s
        assert "participants" in s
        assert "dialogue" in s
        assert "ground_truth" in s

    @pytest.mark.parametrize("case_id", list_scenarios())
    def test_scenario_has_two_speakers(self, case_id):
        s = SCENARIOS[case_id]
        assert len(s["participants"]) >= 2

    @pytest.mark.parametrize("case_id", list_scenarios())
    def test_scenario_dialogue_not_empty(self, case_id):
        assert len(SCENARIOS[case_id]["dialogue"]) > 0

    @pytest.mark.parametrize("case_id", list_scenarios())
    def test_scenario_dialogue_speakers_valid(self, case_id):
        s = SCENARIOS[case_id]
        valid_ids = {p["speaker_id"] for p in s["participants"]}
        for turn in s["dialogue"]:
            assert turn["speaker_id"] in valid_ids

    @pytest.mark.parametrize("case_id", list_scenarios())
    def test_ground_truth_has_symptoms(self, case_id):
        gt = SCENARIOS[case_id]["ground_truth"]
        assert "symptoms" in gt
        assert isinstance(gt["symptoms"], list)

    @pytest.mark.parametrize("case_id", list_scenarios())
    def test_ground_truth_has_speaker_roles(self, case_id):
        gt = SCENARIOS[case_id]["ground_truth"]
        assert "speaker_roles" in gt

    def test_chest_pain_has_expected_red_flags(self):
        gt = SCENARIOS["chest_pain_consultation"]["ground_truth"]
        assert "chest_pain_with_dyspnea" in gt["expected_red_flags"]

    def test_cough_fever_is_telephone(self):
        assert SCENARIOS["cough_fever_telephone"]["encounter_type"] == "telephone_triage"


# ══════════════════════════════════════════════════════════════════
# Audio environment
# ══════════════════════════════════════════════════════════════════


class TestAudioEnv:
    def _tone(self, duration: float = 1.0, freq: float = 440.0) -> np.ndarray:
        t = np.linspace(0, duration, int(TARGET_RATE * duration), dtype=np.float32)
        return 0.5 * np.sin(2 * np.pi * freq * t)

    def test_clean_returns_copy(self):
        audio = self._tone()
        result = apply_environment(audio, AudioEnvConfig(mode="clean"))
        np.testing.assert_array_equal(result, audio)
        assert result is not audio  # must be a copy

    def test_telephone_reduces_bandwidth(self):
        # A 200 Hz tone should be attenuated by the telephone bandpass (300-3400)
        audio_low = self._tone(freq=200.0)
        result = apply_environment(audio_low, AudioEnvConfig(mode="telephone"))
        assert np.max(np.abs(result)) < np.max(np.abs(audio_low)) * 0.5

    def test_telephone_passes_midband(self):
        audio_mid = self._tone(freq=1000.0)
        result = apply_environment(audio_mid, AudioEnvConfig(mode="telephone"))
        # Midband should pass with reasonable level
        assert np.max(np.abs(result)) > np.max(np.abs(audio_mid)) * 0.3

    def test_noisy_adds_noise(self):
        audio = self._tone()
        rng = np.random.default_rng(42)
        result = apply_environment(audio, AudioEnvConfig(mode="noisy"), rng)
        diff = result - audio
        assert np.std(diff) > 0.001

    def test_noisy_deterministic(self):
        audio = self._tone()
        r1 = apply_environment(audio, AudioEnvConfig(mode="noisy"), np.random.default_rng(42))
        r2 = apply_environment(audio, AudioEnvConfig(mode="noisy"), np.random.default_rng(42))
        np.testing.assert_array_equal(r1, r2)

    def test_distance_near_attenuates(self):
        audio = self._tone()
        cfg = AudioEnvConfig(mode="distance_near", distance_near_gain=0.85)
        result = apply_environment(audio, cfg)
        assert np.max(np.abs(result)) < np.max(np.abs(audio))
        assert np.max(np.abs(result)) > np.max(np.abs(audio)) * 0.8

    def test_distance_far_attenuates_more(self):
        audio = self._tone()
        result = apply_environment(audio, AudioEnvConfig(mode="distance_far"))
        assert np.max(np.abs(result)) < np.max(np.abs(audio)) * 0.5

    def test_unknown_mode_returns_copy(self):
        audio = self._tone()
        result = apply_environment(audio, AudioEnvConfig(mode="unknown"))
        np.testing.assert_array_equal(result, audio)

    def test_output_same_length(self):
        audio = self._tone()
        for mode in ["clean", "telephone", "noisy", "distance_near", "distance_far"]:
            result = apply_environment(audio, AudioEnvConfig(mode=mode))
            assert len(result) == len(audio), f"mode={mode} changed length"


# ══════════════════════════════════════════════════════════════════
# TTS engine
# ══════════════════════════════════════════════════════════════════


class TestTTSEngine:
    def test_placeholder_returns_array(self):
        audio = _synthesize_placeholder("Hello world")
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert len(audio) > 0

    def test_placeholder_duration_proportional(self):
        short = _synthesize_placeholder("Hi")
        long = _synthesize_placeholder("This is a much longer sentence for testing")
        assert len(long) > len(short)

    def test_placeholder_minimum_duration(self):
        audio = _synthesize_placeholder("A")
        assert len(audio) >= TARGET_RATE * 0.5  # minimum 0.5s

    def test_voice_config_from_hint_male(self):
        vc = voice_config_from_hint("male")
        assert vc.voice_id is not None
        assert "DAVID" in vc.voice_id

    def test_voice_config_from_hint_female(self):
        vc = voice_config_from_hint("female")
        assert vc.voice_id is not None
        assert "ZIRA" in vc.voice_id

    def test_voice_config_from_hint_unknown(self):
        vc = voice_config_from_hint("unknown_voice")
        assert vc.voice_id is None  # no match, falls back to None

    def test_voice_config_custom_rate(self):
        vc = voice_config_from_hint("male", rate=200)
        assert vc.rate == 200

    @pytest.mark.skipif(not _TTS_AVAILABLE, reason="pyttsx3 not available")
    def test_synthesize_utterance_produces_audio(self):
        audio = synthesize_utterance("Hello world")
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert len(audio) > TARGET_RATE * 0.3  # at least 0.3s

    @pytest.mark.skipif(not _TTS_AVAILABLE, reason="pyttsx3 not available")
    def test_synthesize_different_voices(self):
        male = synthesize_utterance("Hello", voice_config_from_hint("male"))
        female = synthesize_utterance("Hello", voice_config_from_hint("female"))
        # Different voices should produce different audio
        assert not np.array_equal(male, female)


# ══════════════════════════════════════════════════════════════════
# Case generation
# ══════════════════════════════════════════════════════════════════


class TestCaseGeneration:
    @pytest.mark.skipif(not _TTS_AVAILABLE, reason="pyttsx3 not available")
    def test_generate_produces_all_files(self, tmp_path):
        config = GeneratorConfig(output_dir=str(tmp_path))
        scenario = SCENARIOS["chest_pain_consultation"]
        out_dir = generate_case(scenario, config)

        assert (out_dir / "audio.wav").exists()
        assert (out_dir / "transcript.txt").exists()
        assert (out_dir / "ground_truth.json").exists()
        assert (out_dir / "meta.json").exists()

    @pytest.mark.skipif(not _TTS_AVAILABLE, reason="pyttsx3 not available")
    def test_audio_wav_valid(self, tmp_path):
        config = GeneratorConfig(output_dir=str(tmp_path))
        out_dir = generate_case(SCENARIOS["chest_pain_consultation"], config)

        with wave.open(str(out_dir / "audio.wav"), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getframerate() == 16000
            assert wf.getsampwidth() == 2
            assert wf.getnframes() > 0

    @pytest.mark.skipif(not _TTS_AVAILABLE, reason="pyttsx3 not available")
    def test_transcript_has_all_turns(self, tmp_path):
        config = GeneratorConfig(output_dir=str(tmp_path))
        scenario = SCENARIOS["chest_pain_consultation"]
        out_dir = generate_case(scenario, config)

        text = (out_dir / "transcript.txt").read_text(encoding="utf-8")
        lines = [l for l in text.strip().split("\n") if l]
        assert len(lines) == len(scenario["dialogue"])

    @pytest.mark.skipif(not _TTS_AVAILABLE, reason="pyttsx3 not available")
    def test_transcript_has_speaker_labels(self, tmp_path):
        config = GeneratorConfig(output_dir=str(tmp_path))
        out_dir = generate_case(SCENARIOS["chest_pain_consultation"], config)

        text = (out_dir / "transcript.txt").read_text(encoding="utf-8")
        assert "[Doctor]" in text
        assert "[Patient]" in text

    @pytest.mark.skipif(not _TTS_AVAILABLE, reason="pyttsx3 not available")
    def test_ground_truth_structure(self, tmp_path):
        config = GeneratorConfig(output_dir=str(tmp_path))
        out_dir = generate_case(SCENARIOS["chest_pain_consultation"], config)

        gt = json.loads((out_dir / "ground_truth.json").read_text(encoding="utf-8"))
        assert gt["case_id"] == "chest_pain_consultation"
        assert gt["encounter_type"] == "in_person_consultation"
        assert gt["theme"] == "chest_pain"
        assert "symptoms" in gt
        assert "negations" in gt
        assert "medications" in gt
        assert "participants" in gt
        assert "dialogue" in gt

    @pytest.mark.skipif(not _TTS_AVAILABLE, reason="pyttsx3 not available")
    def test_meta_structure(self, tmp_path):
        config = GeneratorConfig(output_dir=str(tmp_path))
        out_dir = generate_case(SCENARIOS["chest_pain_consultation"], config)

        meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
        assert meta["case_id"] == "chest_pain_consultation"
        assert meta["sample_rate"] == 16000
        assert meta["num_turns"] == 15
        assert "segments" in meta
        assert meta["segments"][0]["t0"] == 0.0

    @pytest.mark.skipif(not _TTS_AVAILABLE, reason="pyttsx3 not available")
    def test_meta_segments_monotonic(self, tmp_path):
        config = GeneratorConfig(output_dir=str(tmp_path))
        out_dir = generate_case(SCENARIOS["chest_pain_consultation"], config)

        meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
        for i in range(1, len(meta["segments"])):
            assert meta["segments"][i]["t0"] >= meta["segments"][i - 1]["t1"]

    @pytest.mark.skipif(not _TTS_AVAILABLE, reason="pyttsx3 not available")
    def test_telephone_env(self, tmp_path):
        config = GeneratorConfig(
            output_dir=str(tmp_path),
            audio_env=AudioEnvConfig(mode="telephone"),
        )
        out_dir = generate_case(SCENARIOS["cough_fever_telephone"], config)
        assert (out_dir / "audio.wav").exists()

        meta = json.loads((out_dir / "meta.json").read_text(encoding="utf-8"))
        assert meta["audio_env_mode"] == "telephone"

    @pytest.mark.skipif(not _TTS_AVAILABLE, reason="pyttsx3 not available")
    def test_noisy_env(self, tmp_path):
        config = GeneratorConfig(
            output_dir=str(tmp_path),
            audio_env=AudioEnvConfig(mode="noisy", noise_level=0.01),
        )
        out_dir = generate_case(SCENARIOS["chest_pain_consultation"], config)
        assert (out_dir / "audio.wav").exists()

    @pytest.mark.skipif(not _TTS_AVAILABLE, reason="pyttsx3 not available")
    def test_deterministic_generation(self, tmp_path):
        """Same seed + scenario → same audio output."""
        dir1 = tmp_path / "run1"
        dir2 = tmp_path / "run2"
        scenario = SCENARIOS["abdominal_pain_consultation"]

        c1 = GeneratorConfig(output_dir=str(dir1), seed=42)
        c2 = GeneratorConfig(output_dir=str(dir2), seed=42)

        generate_case(scenario, c1)
        generate_case(scenario, c2)

        wav1 = (dir1 / "abdominal_pain_consultation" / "audio.wav").read_bytes()
        wav2 = (dir2 / "abdominal_pain_consultation" / "audio.wav").read_bytes()
        assert wav1 == wav2

    @pytest.mark.skipif(not _TTS_AVAILABLE, reason="pyttsx3 not available")
    def test_speaker_env_override(self, tmp_path):
        """Patient far from mic → quieter patient audio in mix."""
        config = GeneratorConfig(
            output_dir=str(tmp_path),
            speaker_env_overrides={
                "spk_1": AudioEnvConfig(mode="distance_far"),
            },
        )
        out_dir = generate_case(SCENARIOS["chest_pain_consultation"], config)
        assert (out_dir / "audio.wav").exists()


# ══════════════════════════════════════════════════════════════════
# CLI entry point
# ══════════════════════════════════════════════════════════════════


class TestCLI:
    def test_list_flag(self, capsys):
        from tools.generate_synthetic_cases import main
        ret = main(["--list"])
        assert ret == 0
        out = capsys.readouterr().out
        assert "chest_pain_consultation" in out
        assert "cough_fever_telephone" in out

    def test_invalid_case(self):
        from tools.generate_synthetic_cases import main
        ret = main(["--case", "nonexistent"])
        assert ret == 1

    def test_explain_flag(self, capsys):
        from tools.generate_synthetic_cases import main
        ret = main(["--explain"])
        assert ret == 0
        out = capsys.readouterr().out
        assert "How It Works" in out
        assert "Step 1" in out
        assert "audio.wav" in out

    def test_show_requires_case(self, capsys):
        from tools.generate_synthetic_cases import main
        ret = main(["--show"])
        assert ret == 1
        err = capsys.readouterr().err
        assert "--show requires --case" in err

    def test_open_requires_case(self, capsys):
        from tools.generate_synthetic_cases import main
        ret = main(["--open"])
        assert ret == 1
        err = capsys.readouterr().err
        assert "--open requires --case" in err

    def test_show_missing_case(self, capsys):
        from tools.generate_synthetic_cases import main
        ret = main(["--show", "--case", "chest_pain_consultation",
                     "--output-dir", "nonexistent_dir_12345"])
        assert ret == 1
        err = capsys.readouterr().err
        assert "Case not found" in err

    def test_open_missing_case(self, capsys):
        from tools.generate_synthetic_cases import main
        ret = main(["--open", "--case", "chest_pain_consultation",
                     "--output-dir", "nonexistent_dir_12345"])
        assert ret == 1
        err = capsys.readouterr().err
        assert "Case not found" in err

    @pytest.mark.skipif(not _TTS_AVAILABLE, reason="pyttsx3 not available")
    def test_show_existing_case(self, tmp_path, capsys):
        from tools.generate_synthetic_cases import main
        # Generate first
        main(["--case", "chest_pain_consultation",
              "--output-dir", str(tmp_path)])
        capsys.readouterr()  # clear output
        # Then show
        ret = main(["--show", "--case", "chest_pain_consultation",
                     "--output-dir", str(tmp_path)])
        assert ret == 0
        out = capsys.readouterr().out
        assert "chest_pain_consultation" in out
        assert "audio.wav" in out
        assert "Ground truth" in out

    def test_list_shows_columns(self, capsys):
        from tools.generate_synthetic_cases import main
        ret = main(["--list"])
        assert ret == 0
        out = capsys.readouterr().out
        assert "turns" in out
        assert "symptoms" in out

    @pytest.mark.skipif(not _TTS_AVAILABLE, reason="pyttsx3 not available")
    def test_single_case_prints_summary(self, tmp_path, capsys):
        from tools.generate_synthetic_cases import main
        ret = main(["--case", "chest_pain_consultation",
                     "--output-dir", str(tmp_path)])
        assert ret == 0
        out = capsys.readouterr().out
        assert "Case ID:" in out
        assert "chest_pain_consultation" in out
        assert "Ground truth:" in out

    @pytest.mark.skipif(not _TTS_AVAILABLE, reason="pyttsx3 not available")
    def test_batch_prints_summary(self, tmp_path, capsys):
        from tools.generate_synthetic_cases import main
        ret = main(["--output-dir", str(tmp_path)])
        assert ret == 0
        out = capsys.readouterr().out
        assert "Generated" in out
        assert "case(s)" in out
