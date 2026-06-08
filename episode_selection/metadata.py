"""Episode metadata extraction and cache management."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .data import Episode, estimate_episode_storage_bytes
from .embeddings import build_embedder, combine_embeddings, l2_normalize, sample_frames
from .quality import combine_quality_components, episode_quality_components


@dataclass
class EpisodeMetadata:
    episode_id: str
    num_transitions: int
    storage_bytes: int
    storage_estimated: bool
    language: Optional[str]
    language_embedding: Optional[List[float]]
    vision_embedding: Optional[List[float]]
    multimodal_embedding: Optional[List[float]]
    quality_score: float
    quality_components: Dict[str, float]
    task: Optional[str]
    success: Optional[float]
    reward: Optional[float]


def _fingerprint(config: dict, episodes: Sequence[Episode]) -> str:
    episode_manifest = []
    for ep in episodes:
        file_info = None
        if ep.path and Path(ep.path).exists():
            stat = Path(ep.path).stat()
            file_info = {"path": ep.path, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
        episode_manifest.append(
            {
                "episode_id": ep.episode_id,
                "file": file_info,
                "num_transitions": ep.num_transitions,
                "language": ep.language,
                "task": ep.task,
                "has_images": ep.images is not None,
                "has_gripper": ep.gripper is not None,
            }
        )
    payload = {
        "metadata_schema_version": 2,
        "config": config,
        "episodes": episode_manifest,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def metadata_to_dicts(items: Sequence[EpisodeMetadata]) -> List[dict]:
    return [asdict(item) for item in items]


def write_metadata_cache(path: str | Path, metadata: Sequence[EpisodeMetadata], fingerprint: str) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w") as f:
        f.write(json.dumps({"fingerprint": fingerprint}) + "\n")
        for item in metadata:
            f.write(json.dumps(asdict(item)) + "\n")
    tmp.replace(out)


def read_metadata_cache(path: str | Path, expected_fingerprint: str) -> Optional[List[EpisodeMetadata]]:
    cache = Path(path)
    if not cache.exists():
        return None
    try:
        lines = cache.read_text().splitlines()
    except OSError as exc:
        raise RuntimeError(f"Failed to read metadata cache {cache}") from exc
    if not lines:
        return None
    try:
        header = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Metadata cache header is invalid JSON: {cache}") from exc
    if header.get("fingerprint") != expected_fingerprint:
        return None
    try:
        return [EpisodeMetadata(**json.loads(line)) for line in lines[1:] if line.strip()]
    except (json.JSONDecodeError, TypeError) as exc:
        raise RuntimeError(f"Metadata cache body is invalid or incomplete: {cache}") from exc


def extract_metadata(episodes: Sequence[Episode], config: dict, *, cache_path: str | Path | None = None) -> List[EpisodeMetadata]:
    if not episodes:
        raise ValueError("Cannot extract metadata from an empty episode list")
    fp = _fingerprint(config, episodes)
    if cache_path:
        cached = read_metadata_cache(cache_path, fp)
        if cached is not None:
            return cached

    embedder = build_embedder(config.get("embeddings", {}))
    frame_cfg = config.get("frame_sampling", {})
    quality_cfg = config.get("quality", {})
    lang_weight = float(config.get("multimodal_language_weight", 0.5))

    texts = [ep.language or "" for ep in episodes]
    lang_embeddings = embedder.encode_text(texts) if any(texts) else [None] * len(episodes)
    quality_components = [
        episode_quality_components(
            ep,
            min_len=int(quality_cfg.get("min_len", 1)),
            max_len=quality_cfg.get("max_len"),
        )
        for ep in episodes
    ]
    reward_values = [c["reward"] for c in quality_components if "reward" in c and np.isfinite(c["reward"])]
    if reward_values:
        r_min, r_max = min(reward_values), max(reward_values)
        for comps in quality_components:
            if "reward" in comps:
                comps["reward"] = 1.0 if r_max == r_min else float((comps["reward"] - r_min) / (r_max - r_min))

    metadata: List[EpisodeMetadata] = []
    for idx, ep in enumerate(episodes):
        lang_emb = l2_normalize(lang_embeddings[idx]).tolist() if ep.language else None
        frames = sample_frames(ep.images, frame_cfg.get("strategy", "middle"), int(frame_cfg.get("max_frames", 1)))
        vis_emb = None
        if frames:
            frame_embs = embedder.encode_images(frames)
            vis_emb = l2_normalize(np.mean(frame_embs, axis=0)).tolist()
        multi = combine_embeddings(
            np.asarray(lang_emb) if lang_emb is not None else None,
            np.asarray(vis_emb) if vis_emb is not None else None,
            lang_weight,
        )
        storage, estimated = estimate_episode_storage_bytes(ep)
        q_components = quality_components[idx]
        q_score = combine_quality_components(q_components, weights=quality_cfg.get("weights"))
        metadata.append(
            EpisodeMetadata(
                episode_id=ep.episode_id,
                num_transitions=ep.num_transitions,
                storage_bytes=storage,
                storage_estimated=estimated,
                language=ep.language,
                language_embedding=lang_emb,
                vision_embedding=vis_emb,
                multimodal_embedding=multi.tolist() if multi is not None else None,
                quality_score=q_score,
                quality_components=q_components,
                task=ep.task,
                success=ep.success,
                reward=ep.reward,
            )
        )
    if cache_path:
        write_metadata_cache(cache_path, metadata, fp)
    return metadata
