from episode_selection.pareto import pareto_frontier


def test_pareto_min_metric():
    pts = [
        {"storage_bytes": 10, "action_mse": 5},
        {"storage_bytes": 20, "action_mse": 4},
        {"storage_bytes": 30, "action_mse": 6},
    ]
    frontier = pareto_frontier(pts, metric_key="action_mse")
    assert len(frontier) == 2
    assert {p["storage_bytes"] for p in frontier} == {10, 20}


def test_pareto_max_metric():
    pts = [
        {"storage_bytes": 10, "gripper_f1": 0.5},
        {"storage_bytes": 20, "gripper_f1": 0.8},
        {"storage_bytes": 30, "gripper_f1": 0.7},
    ]
    frontier = pareto_frontier(pts, metric_key="gripper_f1", higher_is_better=True)
    assert {p["storage_bytes"] for p in frontier} == {10, 20}


def test_equal_duplicate_points_are_both_nondominated():
    pts = [
        {"storage_bytes": 10, "action_mse": 1.0, "id": "a"},
        {"storage_bytes": 10, "action_mse": 1.0, "id": "b"},
    ]
    assert len(pareto_frontier(pts, metric_key="action_mse")) == 2
