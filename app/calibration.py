"""Calibration profiles and embedding matching."""

from __future__ import annotations

import copy
import json
import math
import time
import wave
from pathlib import Path

import numpy as np

from app.config import AppConfig

# Module-level singleton for pyannote Inference model (lazy-init)
_EMBED_INFERENCE = None


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


def build_cluster_embeddings(turns: list[dict]) -> dict[str, list[float]]:
    """Compute one mean embedding per diarization speaker cluster.

    Groups turns by ``turn["speaker"]`` (values like "spk_0", "spk_1", ...),
    averages their ``"embedding"`` vectors element-wise, and L2-normalizes.
    Turns without an ``"embedding"`` key are skipped.

    Returns ``{speaker_id: normalized_mean_embedding}``.
    """
    clusters: dict[str, list[list[float]]] = {}
    for turn in turns:
        if "embedding" not in turn:
            continue
        spk = turn["speaker"]
        clusters.setdefault(spk, []).append(turn["embedding"])

    result: dict[str, list[float]] = {}
    for spk, embs in clusters.items():
        if not embs:
            continue
        mean = [sum(col) / len(col) for col in zip(*embs)]
        norm = math.sqrt(sum(x * x for x in mean))
        if norm > 0:
            mean = [x / norm for x in mean]
        result[spk] = mean
    return result


def assign_clusters_to_profile(
    cluster_embs: dict[str, list[float]],
    profile: dict,
    threshold: float,
    margin: float,
) -> dict[str, str]:
    """Assign diarization clusters to profile speakers via greedy 1:1 matching.

    Builds a cosine similarity matrix (cluster × profile speaker), then
    greedily assigns the highest-scoring pair if it meets *threshold* and
    the gap to the second-best profile speaker for that cluster >= *margin*.
    When only one profile speaker remains, the margin check always passes.

    Returns ``{cluster_speaker_id: profile_spk_N}`` mapping.
    """
    speakers = profile.get("speakers", {})
    id_map = profile.get("speaker_id_map", {})
    if not speakers or not cluster_embs:
        return {}

    cluster_ids = list(cluster_embs.keys())
    profile_names = list(speakers.keys())

    # Build similarity matrix
    sim: dict[str, dict[str, float]] = {}
    for cid in cluster_ids:
        sim[cid] = {}
        for pname in profile_names:
            sim[cid][pname] = cosine_similarity(
                cluster_embs[cid], speakers[pname]["embedding"]
            )

    mapping: dict[str, str] = {}
    remaining_clusters = set(cluster_ids)
    remaining_profiles = set(profile_names)

    while remaining_clusters and remaining_profiles:
        best_score = -1.0
        best_cid = None
        best_pname = None
        for cid in remaining_clusters:
            for pname in remaining_profiles:
                if sim[cid][pname] > best_score:
                    best_score = sim[cid][pname]
                    best_cid = cid
                    best_pname = pname

        if best_cid is None or best_score < threshold:
            break

        # Margin check: second-best profile speaker for this cluster
        second_best = -float("inf")
        for pname in remaining_profiles:
            if pname != best_pname:
                second_best = max(second_best, sim[best_cid][pname])

        if (best_score - second_best) < margin:
            # Ambiguous — skip this cluster
            remaining_clusters.discard(best_cid)
            continue

        mapped_id = id_map.get(best_pname)
        if mapped_id is not None:
            mapping[best_cid] = mapped_id
        remaining_clusters.discard(best_cid)
        remaining_profiles.discard(best_pname)

    return mapping


def apply_cluster_override(
    turns: list[dict],
    mapping: dict[str, str],
) -> list[dict]:
    """Override ``turn["speaker"]`` using a cluster→profile mapping.

    Returns a new list — does not mutate *turns*.
    """
    result = copy.deepcopy(turns)
    for turn in result:
        mapped = mapping.get(turn["speaker"])
        if mapped is not None:
            turn["speaker"] = mapped
    return result


def build_calibration_report(
    cluster_embs: dict[str, list[float]],
    profile: dict,
    threshold: float,
    margin: float,
    mapping: dict[str, str],
    profile_name: str,
) -> dict:
    """Build a calibration diagnostics report.

    Computes the full similarity matrix between cluster embeddings and
    profile speakers, and documents the assignment decision for each cluster.
    Does not modify any calibration state.
    """
    speakers = profile.get("speakers", {})
    id_map = profile.get("speaker_id_map", {})
    cluster_ids = sorted(cluster_embs.keys())
    profile_names = sorted(speakers.keys())

    # Similarity matrix
    similarity: dict[str, dict[str, float]] = {}
    for cid in cluster_ids:
        similarity[cid] = {}
        for pname in profile_names:
            similarity[cid][pname] = round(
                cosine_similarity(cluster_embs[cid], speakers[pname]["embedding"]),
                4,
            )

    # Per-cluster decisions
    decisions: dict[str, dict] = {}
    for cid in cluster_ids:
        scores = similarity[cid]
        if not scores:
            continue
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        best_name, best_score = ranked[0]
        second_name, second_score = ranked[1] if len(ranked) > 1 else (None, -float("inf"))
        gap = round(best_score - second_score, 4) if second_name else float("inf")

        assigned = cid in mapping
        if assigned:
            reason = "ok"
        elif best_score < threshold:
            reason = "below_threshold"
        else:
            reason = "below_margin"

        decision: dict = {
            "best": best_name,
            "best_score": best_score,
            "second": second_name,
            "second_score": second_score if second_name else None,
            "margin": gap if second_name else None,
            "assigned": assigned,
            "reason": reason,
        }
        if assigned:
            decision["mapped_to"] = mapping[cid]
        decisions[cid] = decision

    return {
        "profile_name": profile_name,
        "threshold": threshold,
        "margin_required": margin,
        "clusters": cluster_ids,
        "profile_speakers": profile_names,
        "profile_speaker_id_map": {k: v for k, v in id_map.items() if k in profile_names},
        "similarity": similarity,
        "decisions": decisions,
        "final_mapping": mapping,
    }


def print_calibration_debug(report: dict) -> None:
    """Print a human-readable calibration diagnostics summary."""
    print(f"[CAL] Profile: {report['profile_name']}"
          f" threshold={report['threshold']} margin={report['margin_required']}")
    for cid, dec in report["decisions"].items():
        best_part = f"best {dec['best']}={dec['best_score']:.2f}"
        if dec["second"] is not None:
            second_part = f" second {dec['second']}={dec['second_score']:.2f}"
            margin_part = f" margin={dec['margin']:.2f}"
        else:
            second_part = ""
            margin_part = ""
        if dec["assigned"]:
            result = f"ASSIGNED {cid}->{dec['mapped_to']}"
        else:
            result = f"NOT ASSIGNED ({dec['reason']})"
        print(f"[CAL] {cid}: {best_part}{second_part}{margin_part} -> {result}")
    print(f"[CAL] mapping: {json.dumps(report['final_mapping'])}")


def _get_inference():
    """Return a cached pyannote Inference model (singleton)."""
    global _EMBED_INFERENCE
    if _EMBED_INFERENCE is None:
        from pyannote.audio import Inference
        _EMBED_INFERENCE = Inference("pyannote/embedding", use_auth_token=True)
    return _EMBED_INFERENCE


def embed_turns(
    turns: list[dict],
    wav_path: str | Path,
    min_duration_sec: float = 0.0,
) -> list[dict]:
    """Attach speaker embeddings to diarization turns.

    Loads the session WAV, obtains a cached pyannote Inference model,
    and for each turn slices the audio, extracts an L2-normalised
    embedding, and sets ``turn["embedding"]``.

    Turns shorter than *min_duration_sec* are skipped (no embedding
    attached), which causes ``build_cluster_embeddings`` to naturally
    ignore them.

    Returns the same list (mutated in place for efficiency).
    """
    import torch

    wav_path = Path(wav_path)
    with wave.open(str(wav_path), "rb") as wf:
        sample_rate = wf.getframerate()
        pcm_bytes = wf.readframes(wf.getnframes())
    samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32767
    n_samples = len(samples)

    inference = _get_inference()

    for turn in turns:
        duration = turn["end"] - turn["start"]
        if duration < min_duration_sec:
            continue
        start_idx = max(0, int(round(turn["start"] * sample_rate)))
        end_idx = min(n_samples, int(round(turn["end"] * sample_rate)))
        segment = samples[start_idx:end_idx]
        waveform = torch.from_numpy(segment).unsqueeze(0).float()
        raw = inference({"waveform": waveform, "sample_rate": sample_rate})
        norm = float(np.linalg.norm(raw))
        if norm > 0:
            raw = raw / norm
        turn["embedding"] = raw.tolist()

    return turns


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
