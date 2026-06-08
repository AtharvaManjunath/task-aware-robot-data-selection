import json
from pathlib import Path

import numpy as np
import pytest

from episode_selection.data import Episode, load_episodes, split_episode_ids, transitions_for_ids


def _record(episode_id: str) -> dict:
    return {
        "episode_id": episode_id,
        "observations": [[0.0], [1.0]],
        "actions": [[0.0, 1.0], [1.0, 2.0]],
        "gripper": [0, 1],
        "language": "pick",
    }


def test_load_json_jsonl_npz_and_directory(tmp_path: Path):
    json_path = tmp_path / "episodes.json"
    json_path.write_text(json.dumps({"episodes": [_record("json")]}))
    assert load_episodes(json_path)[0].episode_id == "json"

    jsonl_path = tmp_path / "episodes.jsonl"
    jsonl_path.write_text(json.dumps(_record("jsonl")) + "\n")
    assert load_episodes(jsonl_path)[0].episode_id == "jsonl"

    npz_path = tmp_path / "episode.npz"
    np.savez(npz_path, episode_id="npz", observations=np.array([[0.0], [1.0]]), actions=np.ones((2, 2)), gripper=np.array([0, 1]))
    assert load_episodes(npz_path)[0].episode_id == "npz"

    directory = tmp_path / "dir"
    directory.mkdir()
    (directory / "a.json").write_text(json.dumps({"episodes": [_record("a")]}))
    (directory / "b.jsonl").write_text(json.dumps(_record("b")) + "\n")
    assert [ep.episode_id for ep in load_episodes(directory)] == ["a", "b"]


def test_duplicate_episode_ids_rejected(tmp_path: Path):
    path = tmp_path / "episodes.json"
    path.write_text(json.dumps({"episodes": [_record("dup"), _record("dup")]}))
    with pytest.raises(ValueError, match="Duplicate episode_id"):
        load_episodes(path)


def test_zero_quality_fields_are_preserved(tmp_path: Path):
    record = _record("zero")
    record["rating"] = 0
    record["completed"] = 0
    record["task_id"] = 0
    path = tmp_path / "episodes.json"
    path.write_text(json.dumps({"episodes": [record]}))
    ep = load_episodes(path)[0]
    assert ep.human_rating == 0.0
    assert ep.completion == 0.0
    assert ep.task == 0


def test_mismatched_observation_lengths_rejected(tmp_path: Path):
    bad = _record("bad")
    bad["observations"] = [[0.0]]
    path = tmp_path / "episodes.json"
    path.write_text(json.dumps({"episodes": [bad]}))
    with pytest.raises(ValueError, match="observations length"):
        load_episodes(path)


def test_transition_lookup_rejects_missing_and_mixed_gripper():
    episodes = [
        Episode("a", observations=np.ones((2, 1)), actions=np.ones((2, 2)), gripper=np.array([0, 1])),
        Episode("b", observations=np.ones((2, 1)), actions=np.ones((2, 2))),
    ]
    with pytest.raises(ValueError, match="mix present and missing gripper"):
        transitions_for_ids(episodes, ["a", "b"])
    with pytest.raises(ValueError, match="not found"):
        transitions_for_ids(episodes, ["missing"])


def test_split_requires_unique_ids_and_no_contamination():
    episodes = [Episode(str(i), actions=np.ones((2, 1))) for i in range(8)]
    split = split_episode_ids(episodes, seed=2)
    assert not (set(split["train"]) & set(split["val"]))
    assert not (set(split["train"]) & set(split["test"]))
    assert not (set(split["val"]) & set(split["test"]))
