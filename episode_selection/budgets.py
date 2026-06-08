"""Budget parsing and deterministic enforcement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Mapping, Sequence


@dataclass(frozen=True)
class Budget:
    kind: str
    value: float

    @classmethod
    def parse(cls, raw: str | int | float, default_kind: str = "storage_fraction") -> "Budget":
        if isinstance(raw, (int, float)):
            return cls(default_kind, float(raw))
        text = str(raw).strip().lower()
        if not text:
            raise ValueError("Budget string cannot be empty")
        if text.endswith("%"):
            return cls("storage_fraction", float(text[:-1]) / 100.0)
        for suffix, mult in (("gb", 1024**3), ("mb", 1024**2), ("kb", 1024)):
            if text.endswith(suffix):
                return cls("bytes", float(text[: -len(suffix)]) * mult)
        if text.endswith("episodes"):
            return cls("episodes", float(text[: -len("episodes")].strip()))
        if ":" in text:
            kind, value = text.split(":", 1)
            return cls(kind.strip(), float(value))
        return cls(default_kind, float(text))

    @property
    def is_episode_count(self) -> bool:
        return self.kind in {"episodes", "num_episodes"}

    def episode_count(self, dataset_size: int) -> int:
        if not self.is_episode_count:
            raise ValueError(f"Budget {self.kind} is not an episode-count budget")
        count = int(self.value)
        if count <= 0:
            raise ValueError("Episode count budget must be positive")
        return min(count, dataset_size)


def resolve_budget_limit(metadata: Sequence[Mapping], budget: Budget, *, total_train_episodes: int | None = None) -> int:
    total_bytes = sum(int(m["storage_bytes"]) for m in metadata)
    total_eps = total_train_episodes or len(metadata)
    if budget.kind in {"storage_fraction", "fraction", "frac"}:
        if not 0 < budget.value <= 1:
            raise ValueError(f"Storage fraction budget must be in (0, 1], got {budget.value}")
        return int(total_bytes * budget.value)
    if budget.kind in {"episode_fraction", "episodes_fraction"}:
        if not 0 < budget.value <= 1:
            raise ValueError(f"Episode fraction budget must be in (0, 1], got {budget.value}")
        count = max(1, int(round(total_eps * budget.value)))
        sorted_sizes = sorted(int(m["storage_bytes"]) for m in metadata)
        return sum(sorted_sizes[:count])
    if budget.kind in {"episodes", "num_episodes"}:
        count = int(budget.value)
        if count <= 0:
            raise ValueError("Episode count budget must be positive")
        sorted_sizes = sorted(int(m["storage_bytes"]) for m in metadata)
        return sum(sorted_sizes[: min(count, len(sorted_sizes))])
    if budget.kind in {"bytes", "max_bytes"}:
        if budget.value <= 0:
            raise ValueError("Byte budget must be positive")
        return int(budget.value)
    raise ValueError(f"Unknown budget kind: {budget.kind}")


def take_under_byte_budget(
    ordered_ids: Sequence[str],
    metadata_by_id: Mapping[str, Mapping],
    max_bytes: int,
    *,
    allow_oversized_first: bool = False,
) -> List[str]:
    selected: List[str] = []
    used = 0
    for episode_id in ordered_ids:
        size = int(metadata_by_id[episode_id]["storage_bytes"])
        if used + size <= max_bytes or (allow_oversized_first and not selected):
            selected.append(episode_id)
            used += size
    if not selected:
        raise ValueError("Budget produced an empty subset")
    return selected


def selection_stats(selected_ids: Sequence[str], metadata_by_id: Mapping[str, Mapping]) -> dict:
    storage = sum(int(metadata_by_id[i]["storage_bytes"]) for i in selected_ids)
    transitions = sum(int(metadata_by_id[i]["num_transitions"]) for i in selected_ids)
    return {"num_episodes": len(selected_ids), "num_transitions": transitions, "storage_bytes": storage}
