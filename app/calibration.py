"""Calibration profiles and embedding matching."""

from __future__ import annotations

import copy
import inspect
import json
import math
import os
import time
import wave
from pathlib import Path

import numpy as np

from app.config import AppConfig

# Module-level singleton for pyannote Inference model (lazy-init)
_EMBED_INFERENCE = None

# Minimum turn duration (seconds) for prototype embedding computation.
# Turns shorter than this are excluded from embedding and cluster prototypes.
MIN_PROTOTYPE_DURATION_SEC = 1.2


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


def detect_and_mark_overlap(turns: list[dict]) -> list[dict]:
    """Detect overlapping diarization turns and mark them.

    Sorts turns by ``start`` time and marks pairs where
    ``turns[j]["start"] < turns[i]["end"]`` (j > i) with
    ``overlap=True`` and ``overlap_with=<other speaker>``.
    Non-overlapping turns get ``overlap=False``.

    Returns a new list — does not mutate *turns*.
    """
    result = copy.deepcopy(turns)
    result.sort(key=lambda t: t["start"])

    for i, turn_i in enumerate(result):
        if "overlap" not in turn_i:
            turn_i["overlap"] = False
        for j in range(i + 1, len(result)):
            turn_j = result[j]
            if turn_j["start"] >= turn_i["end"]:
                break  # sorted — no further overlaps with turn_i
            # Mark both as overlapping (don't overwrite existing overlap_with)
            if not turn_i["overlap"]:
                turn_i["overlap"] = True
                turn_i["overlap_with"] = turn_j["speaker"]
            if not turn_j.get("overlap", False):
                turn_j["overlap"] = True
                turn_j["overlap_with"] = turn_i["speaker"]

    return result


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


def build_cluster_embeddings_with_stats(
    turns: list[dict],
    min_duration_sec: float = 0.0,
) -> tuple[dict[str, list[float]], dict[str, dict]]:
    """Compute cluster embeddings and per-cluster statistics.

    Turns marked as overlapping (``overlap=True``) or shorter than
    *min_duration_sec* are excluded from prototype computation.

    Returns ``(cluster_embs, cluster_stats)`` where *cluster_stats* maps
    each speaker id to ``{"embedded_turn_count": int, "embedded_total_sec": float}``.
    """
    cluster_emb_lists: dict[str, list[list[float]]] = {}
    cluster_durations: dict[str, list[float]] = {}

    for turn in turns:
        spk = turn["speaker"]
        if "embedding" not in turn:
            continue
        if turn.get("overlap"):
            continue
        duration = turn["end"] - turn["start"]
        if duration < min_duration_sec:
            continue
        cluster_emb_lists.setdefault(spk, []).append(turn["embedding"])
        cluster_durations.setdefault(spk, []).append(duration)

    cluster_embs: dict[str, list[float]] = {}
    cluster_stats: dict[str, dict] = {}
    for spk, embs in cluster_emb_lists.items():
        if not embs:
            continue
        mean = [sum(col) / len(col) for col in zip(*embs)]
        norm = math.sqrt(sum(x * x for x in mean))
        if norm > 0:
            mean = [x / norm for x in mean]
        cluster_embs[spk] = mean
        cluster_stats[spk] = {
            "embedded_turn_count": len(embs),
            "embedded_total_sec": round(sum(cluster_durations[spk]), 3),
        }

    return cluster_embs, cluster_stats


def filter_eligible_clusters(
    cluster_embs: dict[str, list[float]],
    cluster_stats: dict[str, dict],
    min_cluster_turns: int = 0,
    min_cluster_voiced_sec: float = 0.0,
) -> tuple[dict[str, list[float]], dict[str, str]]:
    """Filter clusters by eligibility criteria.

    Returns ``(eligible_embs, ineligible_reasons)`` where
    *ineligible_reasons* maps excluded cluster IDs to their reason string.
    """
    eligible: dict[str, list[float]] = {}
    ineligible: dict[str, str] = {}

    for cid, emb in cluster_embs.items():
        stats = cluster_stats.get(cid, {})
        count = stats.get("embedded_turn_count", 0)
        voiced = stats.get("embedded_total_sec", 0.0)

        if min_cluster_turns > 0 and count < min_cluster_turns:
            ineligible[cid] = "ineligible_too_few_turns"
        elif min_cluster_voiced_sec > 0 and voiced < min_cluster_voiced_sec:
            ineligible[cid] = "ineligible_too_little_voiced_sec"
        else:
            eligible[cid] = emb

    return eligible, ineligible


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
    eligible_cluster_ids: set[str] | None = None,
    allow_partial: bool = True,
) -> list[dict]:
    """Override ``turn["speaker"]`` using a cluster→profile mapping.

    Overlap turns (``overlap=True``) are never overridden (speaker freeze).
    Eligible clusters that are NOT in *mapping* get ``"UNKNOWN"`` as their
    speaker ID (unless the turn is an overlap turn).

    If *allow_partial* is False and not all eligible clusters are assigned,
    returns a deep copy with no overrides applied.

    Returns a new list — does not mutate *turns*.
    """
    if not allow_partial and eligible_cluster_ids is not None:
        unassigned = eligible_cluster_ids - set(mapping.keys())
        if unassigned:
            return copy.deepcopy(turns)

    result = copy.deepcopy(turns)
    for turn in result:
        # Overlap turns keep their original speaker_id (freeze rule)
        if turn.get("overlap"):
            continue
        mapped = mapping.get(turn["speaker"])
        if mapped is not None:
            turn["speaker"] = mapped
        elif eligible_cluster_ids is not None and turn["speaker"] in eligible_cluster_ids:
            turn["speaker"] = "UNKNOWN"
    return result


def build_calibration_report(
    cluster_embs: dict[str, list[float]],
    profile: dict,
    threshold: float,
    margin: float,
    mapping: dict[str, str],
    profile_name: str,
    cluster_stats: dict[str, dict] | None = None,
    ineligible_reasons: dict[str, str] | None = None,
    partial_assignment_applied: bool | None = None,
) -> dict:
    """Build a calibration diagnostics report.

    Computes the full similarity matrix between cluster embeddings and
    profile speakers, and documents the assignment decision for each cluster.
    Does not modify any calibration state.
    """
    speakers = profile.get("speakers", {})
    id_map = profile.get("speaker_id_map", {})
    if ineligible_reasons is None:
        ineligible_reasons = {}
    all_cluster_ids = sorted(set(cluster_embs.keys()) | set(ineligible_reasons.keys()))
    profile_names = sorted(speakers.keys())

    # Similarity matrix (only for clusters that have embeddings)
    similarity: dict[str, dict[str, float]] = {}
    for cid in all_cluster_ids:
        if cid not in cluster_embs:
            continue
        similarity[cid] = {}
        for pname in profile_names:
            similarity[cid][pname] = round(
                cosine_similarity(cluster_embs[cid], speakers[pname]["embedding"]),
                4,
            )

    # Per-cluster decisions
    decisions: dict[str, dict] = {}
    for cid in all_cluster_ids:
        decision: dict = {}

        # Add cluster stats if available
        if cluster_stats and cid in cluster_stats:
            decision["turn_count"] = cluster_stats[cid]["embedded_turn_count"]
            decision["voiced_sec"] = cluster_stats[cid]["embedded_total_sec"]

        # Ineligible clusters
        if cid in ineligible_reasons:
            decision["assigned"] = False
            decision["reason"] = ineligible_reasons[cid]
            decisions[cid] = decision
            continue

        scores = similarity.get(cid, {})
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

        decision.update({
            "best": best_name,
            "best_score": best_score,
            "second": second_name,
            "second_score": second_score if second_name else None,
            "margin": gap if second_name else None,
            "assigned": assigned,
            "reason": reason,
        })
        if assigned:
            decision["mapped_to"] = mapping[cid]
        decisions[cid] = decision

    report: dict = {
        "profile_name": profile_name,
        "threshold": threshold,
        "margin_required": margin,
        "clusters": all_cluster_ids,
        "profile_speakers": profile_names,
        "profile_speaker_id_map": {k: v for k, v in id_map.items() if k in profile_names},
        "similarity": similarity,
        "decisions": decisions,
        "final_mapping": mapping,
    }
    if partial_assignment_applied is not None:
        report["partial_assignment_applied"] = partial_assignment_applied
    return report


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


def _build_auth_kwargs(cls_or_fn) -> dict:
    """Detect which auth keyword *cls_or_fn* accepts and return kwargs.

    Inspects the callable's signature for known HuggingFace token
    parameter names.  Returns an empty dict when HF_TOKEN is unset or
    the callable accepts none of the known names.
    """
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        return {}
    try:
        sig = inspect.signature(cls_or_fn)
    except (ValueError, TypeError):
        return {}
    for candidate in ("token", "use_auth_token", "auth_token", "hf_token"):
        if candidate in sig.parameters:
            return {candidate: hf_token}
    return {}


def _load_embedding_model():
    """Load the pyannote embedding model via Model.from_pretrained."""
    from pyannote.audio import Model

    return Model.from_pretrained(
        "pyannote/embedding", **_build_auth_kwargs(Model.from_pretrained)
    )


def _get_inference():
    """Return a cached pyannote Inference model (singleton)."""
    global _EMBED_INFERENCE
    if _EMBED_INFERENCE is None:
        from pyannote.audio import Inference

        model = _load_embedding_model()
        _EMBED_INFERENCE = Inference(model)
    return _EMBED_INFERENCE


def _unwrap_embedding(raw) -> np.ndarray:
    """Convert pyannote inference output to a 1-D numpy embedding vector.

    Handles SlidingWindowFeature (has ``.data``), plain numpy arrays,
    and torch tensors.  If the result is 2-D (frames × dim), mean-pool
    across frames to produce a single vector.
    """
    arr = raw.data if hasattr(raw, "data") else raw
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr.mean(axis=0)
    return arr


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
        if turn.get("overlap"):
            continue
        duration = turn["end"] - turn["start"]
        if duration < min_duration_sec:
            continue
        start_idx = max(0, int(round(turn["start"] * sample_rate)))
        end_idx = min(n_samples, int(round(turn["end"] * sample_rate)))
        segment = samples[start_idx:end_idx]
        waveform = torch.from_numpy(segment).unsqueeze(0).float()
        raw = inference({"waveform": waveform, "sample_rate": sample_rate})
        vec = _unwrap_embedding(raw)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec = vec / norm
        turn["embedding"] = vec.tolist()

    return turns


def extract_embedding(audio: np.ndarray, sample_rate: int) -> list[float]:
    """Extract a speaker embedding from an audio array via pyannote.

    Requires HF_TOKEN environment variable and pyannote model access.
    Returns an L2-normalized embedding as a plain Python list of floats.
    """
    import torch
    from pyannote.audio import Inference

    model = _load_embedding_model()
    inference = Inference(model)
    waveform = torch.from_numpy(audio).unsqueeze(0).float()
    raw = inference({"waveform": waveform, "sample_rate": sample_rate})
    vec = _unwrap_embedding(raw)
    # L2-normalize
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm
    return vec.tolist()


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
