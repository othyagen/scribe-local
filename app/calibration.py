"""Calibration profiles and embedding matching."""

from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path

import numpy as np

from app.config import AppConfig


def load_profile(profile_path: str) -> dict:
    """Load a calibration profile from a JSON file.

    Raises FileNotFoundError if the file does not exist.
    """
    p = Path(profile_path)
    if not p.exists():
        raise FileNotFoundError(f"Calibration profile not found: {profile_path}")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def save_profile(profile_path: str, data: dict) -> Path:
    """Write a calibration profile to a JSON file."""
    p = Path(profile_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return p


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(a * b for a, b in zip(vec1, vec2))
    mag1 = math.sqrt(sum(a * a for a in vec1))
    mag2 = math.sqrt(sum(b * b for b in vec2))
    if mag1 == 0.0 or mag2 == 0.0:
        return 0.0
    return dot / (mag1 * mag2)


def match_turn_embeddings(
    turns: list[dict],
    profile: dict,
    threshold: float,
) -> list[dict]:
    """Match turn embeddings against a calibration profile.

    For each turn that has an ``"embedding"`` key, compare against every
    speaker embedding in ``profile["speakers"]``.  If the best similarity
    exceeds *threshold*, override ``turn["speaker"]`` with the internal
    ``spk_N`` ID from ``profile["speaker_id_map"]``.

    Returns a new list — does not mutate *turns*.
    """
    result = copy.deepcopy(turns)
    speakers = profile.get("speakers", {})
    id_map = profile.get("speaker_id_map", {})
    if not speakers:
        return result

    for turn in result:
        if "embedding" not in turn:
            continue
        emb = turn["embedding"]
        best_name = None
        best_sim = -1.0
        for name, info in speakers.items():
            sim = cosine_similarity(emb, info["embedding"])
            if sim > best_sim:
                best_sim = sim
                best_name = name
        if best_name is not None and best_sim > threshold:
            mapped_id = id_map.get(best_name)
            if mapped_id is not None:
                turn["speaker"] = mapped_id

    return result


def extract_embedding(audio: np.ndarray, sample_rate: int) -> list[float]:
    """Extract a speaker embedding from an audio array via pyannote.

    Requires HF_TOKEN environment variable and pyannote model access.
    Returns an L2-normalized embedding as a plain Python list of floats.
    """
    import torch
    from pyannote.audio import Inference

    inference = Inference("pyannote/embedding", use_auth_token=True)
    waveform = torch.from_numpy(audio).unsqueeze(0).float()
    embedding = inference({"waveform": waveform, "sample_rate": sample_rate})
    # L2-normalize
    norm = float(np.linalg.norm(embedding))
    if norm > 0:
        embedding = embedding / norm
    return embedding.tolist()


def record_and_build_profile(
    config: AppConfig,
    num_speakers: int,
    duration_sec: float,
) -> dict:
    """Record voice samples and build a calibration profile.

    For each speaker, records audio for *duration_sec* seconds using the
    configured microphone, extracts an embedding, and builds a profile dict.
    """
    from app.audio import AudioCapture

    audio = AudioCapture(config)
    audio.start()

    speakers: dict[str, dict] = {}
    speaker_id_map: dict[str, str] = {}

    try:
        for i in range(num_speakers):
            label = f"Speaker {chr(65 + i)}"
            spk_id = f"spk_{i}"
            print(f"Calibration: {label}, speak now ({duration_sec}s)...")

            chunks: list[np.ndarray] = []
            elapsed = 0.0
            while elapsed < duration_sec:
                try:
                    chunk = audio.get_chunk(timeout=1.0)
                    chunks.append(chunk)
                    elapsed += len(chunk) / config.audio.sample_rate
                except Exception:
                    continue

            recording = np.concatenate(chunks)
            print(f"  Extracting embedding for {label}...")
            emb = extract_embedding(recording, config.audio.sample_rate)
            speakers[label] = {"embedding": emb}
            speaker_id_map[label] = spk_id
    finally:
        audio.stop()

    return {"speakers": speakers, "speaker_id_map": speaker_id_map}
