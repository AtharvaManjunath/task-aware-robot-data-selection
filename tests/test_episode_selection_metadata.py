from pathlib import Path

import numpy as np

from episode_selection.data import Episode
from episode_selection.metadata import extract_metadata


def test_metadata_cache_roundtrip(tmp_path: Path):
    episodes = [Episode("e1", actions=np.ones((3, 2)), language="pick block", success=1.0)]
    cfg = {"embeddings": {"provider": "deterministic", "dim": 8, "allow_deterministic_fallback": True}}
    cache = tmp_path / "metadata.jsonl"
    first = extract_metadata(episodes, cfg, cache_path=cache)
    second = extract_metadata(episodes, cfg, cache_path=cache)
    assert first[0].episode_id == second[0].episode_id
    assert second[0].language_embedding is not None


def test_metadata_cache_invalidates_on_embedding_config_change(tmp_path: Path):
    episodes = [Episode("e1", actions=np.ones((3, 2)), language="pick block", success=1.0)]
    cache = tmp_path / "metadata.jsonl"
    cfg_a = {"embeddings": {"provider": "deterministic", "dim": 8, "allow_deterministic_fallback": True}}
    cfg_b = {"embeddings": {"provider": "deterministic", "dim": 12, "allow_deterministic_fallback": True}}
    first = extract_metadata(episodes, cfg_a, cache_path=cache)
    second = extract_metadata(episodes, cfg_b, cache_path=cache)
    assert len(first[0].language_embedding) == 8
    assert len(second[0].language_embedding) == 12
