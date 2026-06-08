"""Subset selection strategies for robot demonstration episodes."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence

import numpy as np

from .budgets import Budget, resolve_budget_limit, selection_stats, take_under_byte_budget
from .embeddings import build_embedder, l2_normalize


def _metadata_by_id(metadata: Sequence[Mapping]) -> dict:
    by_id = {}
    for m in metadata:
        episode_id = str(m["episode_id"])
        if episode_id in by_id:
            raise ValueError(f"Duplicate episode_id in metadata: {episode_id}")
        by_id[episode_id] = dict(m)
    return by_id


def _embedding(m: Mapping, key: str) -> np.ndarray:
    value = m.get(key)
    if value is None:
        raise ValueError(f"Episode {m.get('episode_id')} is missing {key}")
    return l2_normalize(np.asarray(value, dtype=float))


@dataclass
class SelectionResult:
    selected_episode_ids: List[str]
    stats: dict
    selector: str
    budget: str


class BaseSelector:
    name = "base"

    def select(self, metadata: Sequence[Mapping], budget: Budget, seed: int = 0, config: Optional[dict] = None) -> SelectionResult:
        raise NotImplementedError

    def _finish(self, ordered_ids: Sequence[str], metadata: Sequence[Mapping], budget: Budget, config: Optional[dict] = None) -> SelectionResult:
        if not metadata:
            raise ValueError(f"{self.name} selector received empty metadata")
        by_id = _metadata_by_id(metadata)
        if len(set(ordered_ids)) != len(ordered_ids):
            raise ValueError(f"{self.name} selector produced duplicate episode IDs")
        unknown = [episode_id for episode_id in ordered_ids if episode_id not in by_id]
        if unknown:
            raise ValueError(f"{self.name} selector produced unknown episode IDs: {unknown[:5]}")
        if budget.kind in {"episodes", "num_episodes"}:
            selected = list(ordered_ids[: budget.episode_count(len(ordered_ids))])
            if not selected:
                raise ValueError("Budget produced an empty subset")
            return SelectionResult(selected, selection_stats(selected, by_id), self.name, f"{budget.kind}:{budget.value}")
        max_bytes = resolve_budget_limit(metadata, budget)
        allow_oversized_first = bool((config or {}).get("allow_oversized_first", False))
        selected = take_under_byte_budget(ordered_ids, by_id, max_bytes, allow_oversized_first=allow_oversized_first)
        return SelectionResult(selected, selection_stats(selected, by_id), self.name, f"{budget.kind}:{budget.value}")


class RandomSelector(BaseSelector):
    name = "random"

    def select(self, metadata: Sequence[Mapping], budget: Budget, seed: int = 0, config: Optional[dict] = None) -> SelectionResult:
        _metadata_by_id(metadata)
        ids = sorted(str(m["episode_id"]) for m in metadata)
        rng = np.random.default_rng(seed)
        ordered = [ids[i] for i in rng.permutation(len(ids))]
        return self._finish(ordered, metadata, budget, config)


class LanguageSimilaritySelector(BaseSelector):
    name = "language_similarity"

    def select(self, metadata: Sequence[Mapping], budget: Budget, seed: int = 0, config: Optional[dict] = None) -> SelectionResult:
        config = config or {}
        queries = config.get("queries") or sorted({m.get("task") for m in metadata if m.get("task")})
        if not queries:
            raise ValueError("Language similarity selector requires queries, task labels, or language descriptions")
        if not metadata:
            raise ValueError("Language similarity selector received empty metadata")
        missing = [m["episode_id"] for m in metadata if m.get("language_embedding") is None]
        if missing:
            raise ValueError("Language similarity selector requires language embeddings for all candidate episodes")
        embedding_config = dict(config.get("embedding_config", {}))
        embedding_config.setdefault("dim", len(metadata[0]["language_embedding"]))
        embedder = build_embedder(embedding_config)
        query_embs = embedder.encode_text([str(q) for q in queries])
        aggregate = config.get("query_aggregation", "max")
        if aggregate not in {"max", "mean"}:
            raise ValueError("query_aggregation must be 'max' or 'mean'")
        scored = []
        for m in metadata:
            emb = _embedding(m, "language_embedding")
            sims = query_embs @ emb
            score = float(np.mean(sims) if aggregate == "mean" else np.max(sims))
            scored.append((score, str(m["episode_id"])))
        ordered = [eid for _, eid in sorted(scored, key=lambda x: (-x[0], x[1]))]
        return self._finish(ordered, metadata, budget, config)


class LanguageVisionDiversitySelector(BaseSelector):
    name = "language_vision_diversity"

    def select(self, metadata: Sequence[Mapping], budget: Budget, seed: int = 0, config: Optional[dict] = None) -> SelectionResult:
        config = config or {}
        if not metadata:
            raise ValueError("Language+vision diversity selector received empty metadata")
        allow_language_only = bool(config.get("allow_language_only_fallback", False))
        key = "multimodal_embedding"
        if any(m.get(key) is None or m.get("vision_embedding") is None for m in metadata):
            if not allow_language_only:
                raise ValueError("Vision embeddings are unavailable; enable allow_language_only_fallback to degrade to language-only diversity")
            warnings.warn("Vision embeddings unavailable; using language-only diversity fallback.", RuntimeWarning)
            key = "language_embedding"
        embeddings = np.vstack([_embedding(m, key) for m in metadata])
        ids = [str(m["episode_id"]) for m in metadata]
        by_id = _metadata_by_id(metadata)
        max_bytes = None if budget.is_episode_count else resolve_budget_limit(metadata, budget)
        target_count = budget.episode_count(len(ids)) if budget.is_episode_count else None
        allow_oversized_first = bool(config.get("allow_oversized_first", False))
        queries = config.get("queries")
        lambda_ = float(config.get("mmr_lambda", 0.5))
        if not 0.0 <= lambda_ <= 1.0:
            raise ValueError("mmr_lambda must be in [0, 1]")
        relevance = np.zeros(len(ids))
        if queries:
            embedding_config = dict(config.get("embedding_config", {}))
            embedding_config.setdefault("dim", embeddings.shape[1])
            embedder = build_embedder(embedding_config)
            q = embedder.encode_text([str(x) for x in queries])
            relevance = np.max(q @ embeddings.T, axis=0)
        selected_idx: List[int] = []
        remaining = set(range(len(ids)))
        used = 0
        while remaining:
            best = None
            for idx in sorted(remaining, key=lambda i: ids[i]):
                size = int(by_id[ids[idx]]["storage_bytes"])
                if target_count is not None and len(selected_idx) >= target_count:
                    continue
                if max_bytes is not None and used + size > max_bytes and not (allow_oversized_first and not selected_idx):
                    continue
                if not selected_idx:
                    diversity_penalty = 0.0
                    diversity_gain = 0.0
                else:
                    sims = embeddings[selected_idx] @ embeddings[idx]
                    diversity_penalty = float(np.max(sims))
                    diversity_gain = float(1.0 - diversity_penalty)
                if queries:
                    score = lambda_ * float(relevance[idx]) - (1.0 - lambda_) * diversity_penalty
                else:
                    score = diversity_gain if selected_idx else float(np.linalg.norm(embeddings[idx]))
                candidate = (score, -size, ids[idx], idx)
                if best is None or candidate > best:
                    best = candidate
            if best is None:
                break
            idx = best[3]
            selected_idx.append(idx)
            remaining.remove(idx)
            used += int(by_id[ids[idx]]["storage_bytes"])
            if target_count is not None and len(selected_idx) >= target_count:
                break
            if max_bytes is not None and used >= max_bytes:
                break
        if not selected_idx:
            raise ValueError("Budget produced an empty subset")
        selected = [ids[i] for i in selected_idx]
        return SelectionResult(selected, selection_stats(selected, by_id), self.name, f"{budget.kind}:{budget.value}")


class QualityFilteredSelector(BaseSelector):
    name = "quality_filtered"

    def select(self, metadata: Sequence[Mapping], budget: Budget, seed: int = 0, config: Optional[dict] = None) -> SelectionResult:
        ordered = [
            str(m["episode_id"])
            for m in sorted(metadata, key=lambda m: (-float(m.get("quality_score", 0.0)), str(m["episode_id"])))
        ]
        return self._finish(ordered, metadata, budget, config)


class FullDataSelector(BaseSelector):
    name = "full_data"

    def select(self, metadata: Sequence[Mapping], budget: Budget, seed: int = 0, config: Optional[dict] = None) -> SelectionResult:
        ordered = sorted(str(m["episode_id"]) for m in metadata)
        return self._finish(ordered, metadata, Budget("episodes", len(ordered)), config)


SELECTORS = {
    "random": RandomSelector,
    "language_similarity": LanguageSimilaritySelector,
    "language_vision_diversity": LanguageVisionDiversitySelector,
    "quality_filtered": QualityFilteredSelector,
    "full_data": FullDataSelector,
}


def build_selector(name: str) -> BaseSelector:
    try:
        return SELECTORS[name]()
    except KeyError as exc:
        raise ValueError(f"Unknown selector {name!r}. Available: {sorted(SELECTORS)}") from exc
