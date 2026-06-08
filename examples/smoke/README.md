# Smoke-Test Example

This directory documents the local synthetic smoke-test workflow. It intentionally does not commit generated outputs.

Run from the repository root:

```bash
python scripts/run_episode_selection_experiment.py \
  --config configs/episode_selection_bc.yaml \
  --output-dir outputs/episode_selection_bc_smoke \
  --smoke-test \
  --overwrite
```

Then inspect:

```text
outputs/episode_selection_bc_smoke/aggregate/results.csv
outputs/episode_selection_bc_smoke/aggregate/summary.csv
outputs/episode_selection_bc_smoke/aggregate/pareto_points.csv
outputs/episode_selection_bc_smoke/plots/
```

Smoke-test artifacts are generated from synthetic data and are not research results.

Regenerate plots and aggregate tables:

```bash
python scripts/analyze_episode_selection_results.py \
  --input-dir outputs/episode_selection_bc_smoke \
  --output-dir outputs/episode_selection_bc_smoke/reanalysis \
  --overwrite
```
