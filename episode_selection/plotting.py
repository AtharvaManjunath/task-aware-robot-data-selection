"""Plotting and analysis file generation."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Iterable, Mapping

from .pareto import pareto_frontier
from .results import write_csv


def analyze_results(rows: list[Mapping], output_dir: str | Path, *, make_plots: bool = True) -> None:
    out = Path(output_dir)
    aggregate = out / "aggregate"
    plots = out / "plots"
    write_csv(aggregate / "results.csv", rows)
    pareto_rows = []
    for metric, higher in (("action_mse", False), ("action_mae", False), ("gripper_f1", True)):
        valid = [r for r in rows if metric in r and r.get(metric) is not None]
        for selector in sorted({r.get("selector") for r in valid}):
            pts = [r for r in valid if r.get("selector") == selector]
            for p in pareto_frontier(pts, metric_key=metric, higher_is_better=higher):
                q = dict(p)
                q["pareto_metric"] = metric
                pareto_rows.append(q)
    write_csv(aggregate / "pareto_points.csv", pareto_rows)
    if not make_plots:
        return
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment dependent
        warnings.warn(f"Plotting skipped because matplotlib is unavailable: {exc}", RuntimeWarning)
        return
    plots.mkdir(parents=True, exist_ok=True)
    for metric, ylabel in (
        ("action_mse", "Action MSE"),
        ("action_mae", "Action MAE"),
        ("gripper_f1", "Gripper F1"),
        ("training_time_seconds", "Training time (s)"),
    ):
        fig, ax = plt.subplots(figsize=(6, 4))
        for selector in sorted({r.get("selector") for r in rows}):
            pts = sorted([r for r in rows if r.get("selector") == selector and metric in r], key=lambda r: float(r["storage_bytes"]))
            if not pts:
                continue
            ax.plot([float(p["storage_bytes"]) / 1e6 for p in pts], [float(p[metric]) for p in pts], marker="o", label=selector)
        ax.set_xlabel("Storage used (MB)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(plots / f"{metric}_vs_storage.{ext}")
        plt.close(fig)
    for metric, ylabel, higher in (
        ("action_mse", "Action MSE Pareto frontier", False),
        ("action_mae", "Action MAE Pareto frontier", False),
        ("gripper_f1", "Gripper F1 Pareto frontier", True),
    ):
        fig, ax = plt.subplots(figsize=(6, 4))
        for selector in sorted({r.get("selector") for r in rows}):
            pts = [r for r in rows if r.get("selector") == selector and metric in r]
            if not pts:
                continue
            front = pareto_frontier(pts, metric_key=metric, higher_is_better=higher)
            front = sorted(front, key=lambda r: float(r["storage_bytes"]))
            ax.plot(
                [float(p["storage_bytes"]) / 1e6 for p in front],
                [float(p[metric]) for p in front],
                marker="o",
                linewidth=2,
                label=selector,
            )
        ax.set_xlabel("Storage used (MB)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(plots / f"pareto_{metric}.{ext}")
        plt.close(fig)
