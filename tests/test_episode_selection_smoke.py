from pathlib import Path

from episode_selection.data import make_synthetic_dataset
from episode_selection.runner import run_experiment


def test_smoke_experiment_runs_end_to_end(tmp_path: Path):
    dataset = make_synthetic_dataset(tmp_path / "synthetic.json", num_episodes=12, transitions=5)
    cfg = {
        "dataset": {"path": str(dataset)},
        "output_dir": str(tmp_path / "out"),
        "selectors": ["random", "language_similarity", "language_vision_diversity", "quality_filtered"],
        "budgets": [0.5],
        "budget_type": "storage_fraction",
        "seeds": [0],
        "split": {"train_fraction": 0.7, "val_fraction": 0.15, "seed": 0},
        "metadata": {
            "embeddings": {"provider": "deterministic", "dim": 16, "allow_deterministic_fallback": True},
            "quality": {"min_len": 1},
        },
        "selector_config": {
            "language_similarity": {"queries": ["open drawer"]},
            "language_vision_diversity": {"queries": ["open drawer"], "allow_language_only_fallback": True},
        },
        "plotting": {"enabled": False},
    }
    rows = run_experiment(cfg)
    assert len(rows) == 5
    assert (tmp_path / "out" / "aggregate" / "results.csv").exists()
