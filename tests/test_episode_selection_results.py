from episode_selection.results import summarize


def test_result_aggregation_summary():
    rows = [
        {"selector": "random", "budget": "b", "action_mse": 2.0, "storage_bytes": 10},
        {"selector": "random", "budget": "b", "action_mse": 4.0, "storage_bytes": 20},
    ]
    summary = summarize(rows)
    assert summary[0]["action_mse_mean"] == 3.0
    assert summary[0]["storage_bytes_mean"] == 15.0
