"""Quality scoring for demonstration episodes."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Tuple

import numpy as np

from .data import Episode


DEFAULT_WEIGHTS = {
    "success": 3.0,
    "reward": 1.0,
    "human_rating": 1.0,
    "completion": 1.0,
    "intervention_free": 1.0,
    "smoothness": 0.5,
    "validity": 1.0,
    "length_sanity": 0.5,
}


def _finite01(value: Any) -> float | None:
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(f):
        return None
    return max(0.0, min(1.0, f))


def episode_quality_components(ep: Episode, *, min_len: int = 1, max_len: int | None = None) -> Dict[str, float]:
    comps: Dict[str, float] = {}
    for name in ("success", "human_rating", "completion", "intervention_free"):
        value = _finite01(getattr(ep, name))
        if value is not None:
            comps[name] = value
    if ep.reward is not None and np.isfinite(ep.reward):
        comps["reward"] = float(ep.reward)

    actions = np.asarray(ep.actions, dtype=float) if ep.actions is not None else np.empty((0,))
    valid = actions.size > 0 and np.all(np.isfinite(actions))
    comps["validity"] = 1.0 if valid else 0.0
    length = ep.num_transitions
    upper_ok = True if max_len is None else length <= max_len
    comps["length_sanity"] = 1.0 if length >= min_len and upper_ok else 0.0
    if valid and len(actions) >= 3:
        diffs = np.diff(actions.reshape(len(actions), -1), n=2, axis=0)
        jerk = float(np.mean(np.square(diffs)))
        comps["smoothness"] = 1.0 / (1.0 + jerk)
    elif valid:
        comps["smoothness"] = 1.0
    else:
        comps["smoothness"] = 0.0
    return comps


def combine_quality_components(components: Mapping[str, float], weights: Mapping[str, float] | None = None) -> float:
    weights = dict(DEFAULT_WEIGHTS if weights is None else weights)
    weighted = []
    total_weight = 0.0
    for key, value in components.items():
        weight = float(weights.get(key, 0.0))
        if weight <= 0:
            continue
        weighted.append(weight * float(value))
        total_weight += weight
    if total_weight <= 0:
        raise ValueError("Invalid quality configuration: all weights are zero or no components are available")
    return float(sum(weighted) / total_weight)


def score_episode_quality(ep: Episode, weights: Mapping[str, float] | None = None, **kwargs: Any) -> Tuple[float, Dict[str, float]]:
    comps = episode_quality_components(ep, **kwargs)
    return combine_quality_components(comps, weights), comps
