"""Dataset adapters for episode-organized robot demonstrations."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Episode:
    episode_id: str
    observations: Any = None
    actions: Optional[np.ndarray] = None
    gripper: Optional[np.ndarray] = None
    language: Optional[str] = None
    images: Optional[np.ndarray] = None
    task: Optional[str] = None
    success: Optional[float] = None
    reward: Optional[float] = None
    human_rating: Optional[float] = None
    completion: Optional[float] = None
    intervention_free: Optional[float] = None
    path: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_transitions(self) -> int:
        if self.actions is not None:
            return int(len(self.actions))
        if self.gripper is not None:
            return int(len(self.gripper))
        return 0


def _first_present(record: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in record:
            return record[key]
    return None


def _first_non_none(record: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        if key in record and record[key] is not None:
            return record[key]
    return None


def _as_array(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    arr = np.asarray(value)
    return arr


def _validate_episode(ep: Episode) -> None:
    if not ep.episode_id:
        raise ValueError("Episode is missing a stable episode_id")
    if ep.actions is None:
        raise ValueError(f"Episode {ep.episode_id} is missing required actions")
    actions = np.asarray(ep.actions)
    if actions.ndim == 0:
        raise ValueError(f"Episode {ep.episode_id} actions must have at least one transition")
    if len(actions) == 0:
        raise ValueError(f"Episode {ep.episode_id} has an empty action sequence")
    try:
        actions.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Episode {ep.episode_id} actions must be numeric") from exc
    if ep.observations is not None and len(np.asarray(ep.observations)) != len(actions):
        raise ValueError(
            f"Episode {ep.episode_id} observations length {len(np.asarray(ep.observations))} "
            f"does not match actions length {len(actions)}"
        )
    if ep.gripper is not None and len(np.asarray(ep.gripper)) != len(actions):
        raise ValueError(
            f"Episode {ep.episode_id} gripper length {len(np.asarray(ep.gripper))} "
            f"does not match actions length {len(actions)}"
        )


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    raise TypeError(f"Cannot serialize {type(obj)!r}")


def episode_from_mapping(
    record: Mapping[str, Any],
    *,
    language_keys: Sequence[str] = ("language", "instruction", "task_description"),
    image_keys: Sequence[str] = ("images", "frames", "rgb"),
    action_key: str = "actions",
    gripper_key: Optional[str] = "gripper",
    episode_id_key: str = "episode_id",
    path: Optional[str] = None,
) -> Episode:
    actions = _as_array(record.get(action_key))
    gripper = _as_array(record.get(gripper_key)) if gripper_key else None
    images = _as_array(_first_present(record, image_keys))
    language = _first_present(record, language_keys)
    if language is not None:
        language = str(language)
    episode_id = str(record.get(episode_id_key, record.get("id", path or len(str(record)))))
    extra = {k: v for k, v in record.items() if k not in {action_key, gripper_key, episode_id_key}}
    ep = Episode(
        episode_id=episode_id,
        observations=record.get("observations"),
        actions=actions,
        gripper=gripper,
        language=language,
        images=images,
        task=_first_non_none(record, ("task", "task_id", "task_name")),
        success=_maybe_float(record.get("success")),
        reward=_maybe_float(record.get("reward")),
        human_rating=_maybe_float(_first_non_none(record, ("human_rating", "rating"))),
        completion=_maybe_float(_first_non_none(record, ("completion", "completed"))),
        intervention_free=_maybe_float(record.get("intervention_free")),
        path=path,
        extra=extra,
    )
    _validate_episode(ep)
    return ep


def _maybe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        if isinstance(value, (list, tuple, np.ndarray)):
            arr = np.asarray(value, dtype=float)
            return float(np.nanmean(arr))
        return float(value)
    except (TypeError, ValueError):
        return None


def load_episodes(path: str | Path, **kwargs: Any) -> List[Episode]:
    """Load episodes from a directory, JSON, JSONL, or NPZ file.

    Directory mode reads each ``.json``, ``.jsonl``, and ``.npz`` file as episode data.
    JSON may contain either a list of episode records or an object with an ``episodes`` list.
    """

    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {root}")
    if root.is_dir():
        episodes: List[Episode] = []
        for file_path in sorted(root.iterdir()):
            if file_path.suffix.lower() in {".json", ".jsonl", ".npz"}:
                episodes.extend(load_episodes(file_path, **kwargs))
        if not episodes:
            raise ValueError(f"No episode files found in dataset directory: {root}")
        _validate_unique_episode_ids(episodes)
        return episodes
    if root.suffix.lower() == ".json":
        data = json.loads(root.read_text())
        records = data.get("episodes", data) if isinstance(data, dict) else data
        if not isinstance(records, list):
            raise ValueError("JSON dataset must be a list or contain an 'episodes' list")
        episodes = [episode_from_mapping(r, path=str(root), **kwargs) for r in records]
        _validate_unique_episode_ids(episodes)
        return episodes
    if root.suffix.lower() == ".jsonl":
        episodes = []
        for line in root.read_text().splitlines():
            if line.strip():
                episodes.append(episode_from_mapping(json.loads(line), path=str(root), **kwargs))
        _validate_unique_episode_ids(episodes)
        return episodes
    if root.suffix.lower() == ".npz":
        with np.load(root, allow_pickle=True) as data:
            record = {k: data[k].tolist() if data[k].dtype == object else data[k] for k in data.files}
        return [episode_from_mapping(record, path=str(root), **kwargs)]
    raise ValueError(f"Unsupported dataset file type: {root.suffix}")


def _validate_unique_episode_ids(episodes: Sequence[Episode]) -> None:
    seen = set()
    for ep in episodes:
        if ep.episode_id in seen:
            raise ValueError(f"Duplicate episode_id in dataset: {ep.episode_id}")
        seen.add(ep.episode_id)


def estimate_episode_storage_bytes(ep: Episode) -> Tuple[int, bool]:
    if ep.path and Path(ep.path).exists() and Path(ep.path).suffix.lower() == ".npz":
        return int(os.path.getsize(ep.path)), False
    total = 0
    for arr in (ep.actions, ep.gripper, ep.images):
        if arr is not None:
            total += int(np.asarray(arr).nbytes)
    if ep.language:
        total += len(ep.language.encode("utf-8"))
    if total == 0:
        total = max(1, ep.num_transitions) * 128
    return total, True


def make_synthetic_dataset(
    out_path: str | Path,
    *,
    num_episodes: int = 16,
    transitions: int = 8,
    action_dim: int = 4,
    seed: int = 0,
) -> Path:
    """Create a tiny deterministic JSON dataset for smoke tests."""

    rng = np.random.default_rng(seed)
    tasks = ["pick red block", "place blue cup", "open drawer", "close drawer"]
    episodes = []
    weights = rng.normal(size=(len(tasks), action_dim))
    for i in range(num_episodes):
        task_idx = i % len(tasks)
        t = np.linspace(0, 1, transitions)
        obs = np.stack([t, np.full_like(t, task_idx), np.sin(t * math.pi)], axis=1)
        noise = rng.normal(scale=0.04 + 0.01 * (i % 3), size=(transitions, action_dim))
        actions = obs[:, :1] @ weights[task_idx : task_idx + 1, :] + noise
        gripper = (t > 0.5).astype(int)
        success = 1.0 if i % 5 != 0 else 0.0
        episodes.append(
            {
                "episode_id": f"ep_{i:03d}",
                "observations": obs.tolist(),
                "actions": actions.tolist(),
                "gripper": gripper.tolist(),
                "language": tasks[task_idx],
                "task": tasks[task_idx],
                "success": success,
                "reward": float(success * (1.0 - 0.02 * (i % 4))),
            }
        )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps({"episodes": episodes}, default=_json_default, indent=2))
    tmp.replace(out)
    return out


def split_episode_ids(
    episodes: Sequence[Episode],
    *,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    seed: int = 0,
) -> Dict[str, List[str]]:
    if len(episodes) < 3:
        raise ValueError("At least three episodes are required to create train/validation/test splits")
    _validate_unique_episode_ids(episodes)
    if not 0 < train_fraction < 1 or not 0 <= val_fraction < 1:
        raise ValueError("Split fractions must be in valid ranges")
    if train_fraction + val_fraction >= 1:
        raise ValueError("train_fraction + val_fraction must be < 1")
    ids = np.array([ep.episode_id for ep in episodes], dtype=object)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(ids))
    n_train = max(1, int(round(len(ids) * train_fraction)))
    n_val = int(round(len(ids) * val_fraction))
    if n_train + n_val >= len(ids):
        n_train = max(1, len(ids) - 2)
        n_val = 1 if len(ids) - n_train > 1 else 0
    return {
        "train": ids[order[:n_train]].tolist(),
        "val": ids[order[n_train : n_train + n_val]].tolist(),
        "test": ids[order[n_train + n_val :]].tolist(),
    }


def transitions_for_ids(episodes: Sequence[Episode], ids: Iterable[str]) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    requested = list(ids)
    if len(set(requested)) != len(requested):
        raise ValueError("Selected episode IDs contain duplicates")
    wanted = set(requested)
    xs, ys, gs = [], [], []
    seen = set()
    saw_gripper = False
    saw_missing_gripper = False
    for ep in episodes:
        if ep.episode_id not in wanted:
            continue
        seen.add(ep.episode_id)
        if ep.actions is None:
            raise ValueError(f"Episode {ep.episode_id} is missing actions")
        actions = np.asarray(ep.actions, dtype=float)
        obs = ep.observations if ep.observations is not None else np.arange(len(actions))[:, None]
        obs_arr = np.asarray(obs, dtype=float)
        if len(obs_arr) != len(actions):
            raise ValueError(
                f"Episode {ep.episode_id} observations length {len(obs_arr)} does not match actions length {len(actions)}"
            )
        xs.append(obs_arr.reshape(len(actions), -1))
        ys.append(actions.reshape(len(actions), -1))
        if ep.gripper is not None:
            saw_gripper = True
            gs.append(np.asarray(ep.gripper).reshape(len(actions), -1)[:, 0])
        else:
            saw_missing_gripper = True
    missing = sorted(wanted - seen)
    if missing:
        raise ValueError(f"Requested episode IDs were not found: {missing[:5]}")
    if saw_gripper and saw_missing_gripper:
        raise ValueError("Selected episodes mix present and missing gripper labels")
    if not xs:
        raise ValueError("No transitions found for selected episode IDs")
    gripper = np.concatenate(gs) if gs else None
    return np.vstack(xs), np.vstack(ys), gripper
