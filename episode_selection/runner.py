"""Experiment runner for semantic and quality-aware episode selection."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
from typing import Optional

import yaml

from .bc import evaluate_linear_bc, train_linear_bc
from .budgets import Budget
from .data import Episode, load_episodes, make_synthetic_dataset, split_episode_ids
from .metadata import extract_metadata, metadata_to_dicts
from .plotting import analyze_results
from .results import atomic_write_json, collect_metrics, summarize, write_csv
from .selectors import build_selector


def load_config(path: str | Path) -> dict:
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return data


def resolve_config(config: dict, args: argparse.Namespace) -> dict:
    cfg = copy.deepcopy(config)
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    if args.selectors:
        cfg["selectors"] = args.selectors.split(",")
    if args.budgets:
        cfg["budgets"] = args.budgets.split(",")
    if args.seeds:
        cfg["seeds"] = [int(s) for s in args.seeds.split(",")]
    return cfg


def _config_fingerprint(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _candidate_metadata(meta: list[dict], ids: list[str]) -> list[dict]:
    wanted = set(ids)
    return [m for m in meta if m["episode_id"] in wanted]


def run_experiment(
    cfg: dict,
    *,
    dry_run: bool = False,
    resume: bool = False,
    overwrite: bool = False,
    skip_training: bool = False,
    only_analyze: bool = False,
    analysis_input_dir: str | Path | None = None,
) -> list[dict]:
    output_dir = Path(cfg["output_dir"])
    if only_analyze:
        rows = collect_metrics(analysis_input_dir or output_dir)
        write_csv(output_dir / "aggregate" / "summary.csv", summarize(rows))
        analyze_results(rows, output_dir, make_plots=bool(cfg.get("plotting", {}).get("enabled", True)))
        return rows

    dataset_cfg = cfg.get("dataset", {})
    episodes = load_episodes(
        dataset_cfg["path"],
        language_keys=dataset_cfg.get("language_keys", ["language", "instruction", "task_description"]),
        image_keys=dataset_cfg.get("image_keys", ["images", "frames", "rgb"]),
        action_key=dataset_cfg.get("action_key", "actions"),
        gripper_key=dataset_cfg.get("gripper_key", "gripper"),
    )
    split_cfg = cfg.get("split", {})
    splits = split_episode_ids(
        episodes,
        train_fraction=float(split_cfg.get("train_fraction", 0.7)),
        val_fraction=float(split_cfg.get("val_fraction", 0.15)),
        seed=int(split_cfg.get("seed", 0)),
    )
    metadata_cache = output_dir / "metadata" / "episode_metadata.jsonl"
    metadata_cfg = copy.deepcopy(cfg.get("metadata", {}))
    metadata_cfg["dataset_adapter"] = {
        "path": dataset_cfg["path"],
        "language_keys": dataset_cfg.get("language_keys", ["language", "instruction", "task_description"]),
        "image_keys": dataset_cfg.get("image_keys", ["images", "frames", "rgb"]),
        "action_key": dataset_cfg.get("action_key", "actions"),
        "gripper_key": dataset_cfg.get("gripper_key", "gripper"),
    }
    metadata = metadata_to_dicts(extract_metadata(episodes, metadata_cfg, cache_path=metadata_cache))
    train_meta = _candidate_metadata(metadata, splits["train"])
    selectors = list(cfg.get("selectors", ["random", "language_similarity", "language_vision_diversity", "quality_filtered"]))
    include_full_data = bool(cfg.get("include_full_data_baseline", True))
    budgets = [Budget.parse(b, cfg.get("budget_type", "storage_fraction")) for b in cfg.get("budgets", [1.0])]
    seeds = [int(s) for s in cfg.get("seeds", [0])]

    planned = [(sel, b, seed) for sel in selectors if sel != "full_data" for b in budgets for seed in seeds]
    if include_full_data or "full_data" in selectors:
        planned.extend(("full_data", Budget("storage_fraction", 1.0), seed) for seed in seeds)
    if dry_run:
        print(json.dumps({"planned_runs": [(s, f"{b.kind}:{b.value}", seed) for s, b, seed in planned], "splits": splits}, indent=2))
        return []

    rows = []
    for selector_name, budget, seed in planned:
        run_dir = output_dir / "runs" / selector_name / f"{budget.kind}_{budget.value:g}" / str(seed)
        metrics_path = run_dir / "metrics.json"
        config_hash = _config_fingerprint(cfg)
        if resume and metrics_path.exists():
            existing = json.loads(metrics_path.read_text())
            if existing.get("config_fingerprint") != config_hash:
                raise RuntimeError(f"Cannot resume {run_dir}: config fingerprint differs")
            rows.append(existing)
            continue
        if metrics_path.exists() and not overwrite:
            raise RuntimeError(f"Run already exists at {run_dir}; use --resume or --overwrite")
        selector = build_selector(selector_name)
        selector_cfg = copy.deepcopy(cfg.get("selector_config", {}).get(selector_name, {}))
        selector_cfg.setdefault("embedding_config", cfg.get("metadata", {}).get("embeddings", {}))
        selector_cfg.setdefault("allow_oversized_first", bool(cfg.get("budget", {}).get("allow_oversized_first", False)))
        selection = selector.select(train_meta, budget, seed=seed, config=selector_cfg)
        train_ids = set(splits["train"])
        leaked = sorted(set(selection.selected_episode_ids) - train_ids)
        if leaked:
            raise RuntimeError(f"Selector {selector_name} selected non-training episodes: {leaked[:5]}")
        atomic_write_json(run_dir / "selected_episodes.json", {"episode_ids": selection.selected_episode_ids, "stats": selection.stats})
        atomic_write_json(run_dir / "config_resolved.json", cfg)
        if skip_training:
            raise RuntimeError("--skip-training requires pre-existing metrics; rerun with --resume or --only-analyze")
        start = time.time()
        model, train_time = train_linear_bc(episodes, selection.selected_episode_ids, ridge=float(cfg.get("training", {}).get("ridge", 1e-4)))
        eval_ids = splits["test"] or splits["val"]
        metrics = evaluate_linear_bc(model, episodes, eval_ids, gripper_threshold=float(cfg.get("evaluation", {}).get("gripper_threshold", 0.5)))
        row = {
            "selector": selector_name,
            "budget": f"{budget.kind}:{budget.value}",
            "seed": seed,
            "storage_bytes": selection.stats["storage_bytes"],
            "num_episodes": selection.stats["num_episodes"],
            "num_transitions": selection.stats["num_transitions"],
            "training_time_seconds": train_time,
            "started_at_unix": start,
            "ended_at_unix": time.time(),
            "config_fingerprint": config_hash,
            **metrics,
        }
        atomic_write_json(metrics_path, row)
        rows.append(row)
    all_rows = collect_metrics(output_dir)
    write_csv(output_dir / "aggregate" / "summary.csv", summarize(all_rows))
    analyze_results(all_rows, output_dir, make_plots=bool(cfg.get("plotting", {}).get("enabled", True)))
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/episode_selection_bc.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--input-dir", help="Existing experiment output directory to analyze when --only-analyze is used")
    parser.add_argument("--selectors")
    parser.add_argument("--budgets")
    parser.add_argument("--seeds")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--only-analyze", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    cfg = load_config(args.config)
    cfg = resolve_config(cfg, args)
    if args.smoke_test:
        output_dir = Path(cfg.get("output_dir", "outputs/episode_selection_bc_smoke"))
        cfg["output_dir"] = str(output_dir)
        cfg.setdefault("metadata", {}).setdefault("embeddings", {})["provider"] = "deterministic"
        cfg.setdefault("metadata", {}).setdefault("embeddings", {})["allow_deterministic_fallback"] = True
        dataset_path = output_dir / "synthetic_dataset.json"
        make_synthetic_dataset(dataset_path)
        cfg.setdefault("dataset", {})["path"] = str(dataset_path)
        cfg["budgets"] = cfg.get("budgets", [0.5, 1.0])
    run_experiment(
        cfg,
        dry_run=args.dry_run,
        resume=args.resume,
        overwrite=args.overwrite,
        skip_training=args.skip_training,
        only_analyze=args.only_analyze,
        analysis_input_dir=args.input_dir,
    )


if __name__ == "__main__":
    main()
