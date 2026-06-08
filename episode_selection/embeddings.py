"""Embedding providers for language and representative visual frames."""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=float)
    norm = np.linalg.norm(arr)
    if not np.isfinite(norm) or norm == 0:
        return np.zeros_like(arr, dtype=float)
    return arr / norm


@dataclass
class DeterministicHashEmbedder:
    """Small deterministic embedder for tests, smoke runs, and explicit debug use."""

    dim: int = 64

    def encode_text(self, texts: Sequence[str]) -> np.ndarray:
        return np.vstack([self._hash_tokens(t or "") for t in texts])

    def encode_images(self, images: Sequence[np.ndarray]) -> np.ndarray:
        vectors = []
        for image in images:
            arr = np.asarray(image, dtype=float).ravel()
            if arr.size == 0:
                vectors.append(np.zeros(self.dim))
                continue
            stats = np.array([arr.mean(), arr.std(), arr.min(), arr.max()])
            vec = np.resize(stats, self.dim)
            vectors.append(l2_normalize(vec))
        return np.vstack(vectors)

    def _hash_tokens(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=float)
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        return l2_normalize(vec)


def build_embedder(config: dict) -> DeterministicHashEmbedder:
    provider = config.get("provider", "none")
    allow_fallback = bool(config.get("allow_deterministic_fallback", False))
    dim = int(config.get("dim", 64))
    if provider in {"deterministic", "hash"}:
        return DeterministicHashEmbedder(dim=dim)
    if provider in {"none", None} and allow_fallback:
        warnings.warn("Using deterministic fallback embeddings because explicitly enabled.", RuntimeWarning)
        return DeterministicHashEmbedder(dim=dim)
    raise RuntimeError(
        "No production embedding provider is configured. Set embeddings.provider to an available "
        "implementation, or explicitly enable allow_deterministic_fallback for tests/smoke runs."
    )


def sample_frames(images: Optional[np.ndarray], strategy: str = "middle", max_frames: int = 1) -> list[np.ndarray]:
    if images is None:
        return []
    arr = np.asarray(images)
    if arr.size == 0:
        return []
    if arr.ndim < 2:
        arr = arr.reshape(1, -1)
    n = len(arr)
    max_frames = max(1, int(max_frames))
    if strategy == "first":
        idx = [0]
    elif strategy == "last":
        idx = [n - 1]
    elif strategy == "middle":
        idx = [n // 2]
    elif strategy == "uniform":
        idx = np.linspace(0, n - 1, min(max_frames, n), dtype=int).tolist()
    else:
        raise ValueError(f"Unknown frame sampling strategy: {strategy}")
    return [arr[i] for i in idx[:max_frames]]


def combine_embeddings(language: Optional[np.ndarray], vision: Optional[np.ndarray], lang_weight: float = 0.5) -> Optional[np.ndarray]:
    if language is None and vision is None:
        return None
    if language is None:
        return l2_normalize(np.asarray(vision, dtype=float))
    if vision is None:
        return l2_normalize(np.asarray(language, dtype=float))
    lang = l2_normalize(language)
    vis = l2_normalize(vision)
    if lang.shape != vis.shape:
        size = max(lang.size, vis.size)
        lang = np.resize(lang, size)
        vis = np.resize(vis, size)
    weight = float(lang_weight)
    return l2_normalize(weight * lang + (1.0 - weight) * vis)
