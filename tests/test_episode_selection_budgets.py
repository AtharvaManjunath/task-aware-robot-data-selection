from episode_selection.budgets import Budget, resolve_budget_limit
import pytest

from episode_selection.selectors import RandomSelector


META = [
    {"episode_id": "a", "storage_bytes": 10, "num_transitions": 1},
    {"episode_id": "b", "storage_bytes": 20, "num_transitions": 2},
    {"episode_id": "c", "storage_bytes": 30, "num_transitions": 3},
]


def test_budget_parse_storage_fraction_and_bytes():
    assert Budget.parse("20%").kind == "storage_fraction"
    assert Budget.parse("1MB").value == 1024 * 1024
    assert resolve_budget_limit(META, Budget("storage_fraction", 0.5)) == 30


def test_exact_episode_budget_enforced_by_count():
    result = RandomSelector().select(META, Budget("episodes", 2), seed=0)
    assert len(result.selected_episode_ids) == 2


def test_strict_byte_budget_does_not_exceed_smallest_episode():
    with pytest.raises(ValueError, match="empty subset"):
        RandomSelector().select(META, Budget("bytes", 5), seed=0)


def test_oversized_first_requires_explicit_config():
    result = RandomSelector().select(META, Budget("bytes", 5), seed=0, config={"allow_oversized_first": True})
    assert len(result.selected_episode_ids) == 1
    assert result.stats["storage_bytes"] > 5


def test_invalid_budgets_raise():
    with pytest.raises(ValueError):
        Budget.parse("")
    with pytest.raises(ValueError):
        resolve_budget_limit(META, Budget("storage_fraction", 0))
