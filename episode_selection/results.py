"""Result persistence and aggregation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, List, Mapping


def atomic_write_json(path: str | Path, payload: Mapping) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    tmp.replace(out)


def collect_metrics(output_dir: str | Path) -> List[dict]:
    root = Path(output_dir) / "runs"
    if not root.exists():
        raise FileNotFoundError(f"No runs directory found under {output_dir}")
    rows = []
    for metrics_path in sorted(root.glob("*/*/*/metrics.json")):
        rows.append(json.loads(metrics_path.read_text()))
    if not rows:
        raise FileNotFoundError(f"No metrics.json files found under {root}")
    return rows


def write_csv(path: str | Path, rows: Iterable[Mapping]) -> None:
    rows = [dict(r) for r in rows]
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        out.write_text("")
        return
    keys = sorted({k for row in rows for k in row})
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: Iterable[Mapping]) -> List[dict]:
    grouped = {}
    for row in rows:
        key = (row.get("selector"), row.get("budget"))
        grouped.setdefault(key, []).append(row)
    summary = []
    for (selector, budget), items in sorted(grouped.items()):
        out = {"selector": selector, "budget": budget, "n": len(items)}
        for metric in ("action_mse", "action_mae", "gripper_f1", "training_time_seconds", "storage_bytes"):
            vals = [float(x[metric]) for x in items if metric in x and x[metric] is not None]
            if vals:
                out[f"{metric}_mean"] = sum(vals) / len(vals)
                if len(vals) > 1:
                    mean = out[f"{metric}_mean"]
                    out[f"{metric}_std"] = (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
        summary.append(out)
    return summary
