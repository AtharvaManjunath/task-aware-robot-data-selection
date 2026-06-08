# Semantic and Quality-Aware Episode Selection for Data-Efficient Behavior Cloning

This repository provides a research software framework for comparing robot demonstration episode-selection strategies for behavior cloning. It asks whether semantic or quality-aware subsets can approach full-data action-prediction performance while using less storage and training time.

> **Responsible claims note:** this repo currently provides the experiment framework, deterministic tests, and a synthetic smoke-test pipeline. It does **not** include real robot results. Smoke-test artifacts are generated from synthetic data and are not research results.

## Research Question

Can semantic or quality-aware robot demonstration episode selection preserve behavior-cloning action prediction performance while reducing demonstration storage and training time?

## Features

- Episode metadata extraction with cache invalidation.
- Configurable storage and episode-count budgets.
- Deterministic subset selectors with explicit missing-data failures.
- Language, vision, multimodal, and quality-aware selection interfaces.
- Lightweight NumPy linear behavior-cloning adapter for smoke and integration checks.
- Fixed train/validation/test splits shared across methods.
- Action MSE, action MAE, gripper precision/recall/F1, training time, and storage metrics.
- Aggregate CSVs plus storage/performance and Pareto plots.
- Fast pytest suite and GitHub Actions CI.

## Methods Compared

- `random`: seeded uniform random episode ordering under budget.
- `language_similarity`: cosine similarity between episode language embeddings and target queries or task labels.
- `language_vision_diversity`: MMR-style relevance/diversity selection over multimodal embeddings, with explicit language-only fallback.
- `quality_filtered`: quality ranking from success/reward/rating/completion/intervention labels plus deterministic proxy metrics.
- `full_data`: full training split baseline included by default.

## Metrics

- `action_mse`: mean squared error over continuous action dimensions.
- `action_mae`: mean absolute error over continuous action dimensions.
- `gripper_precision`, `gripper_recall`, `gripper_f1`: binary gripper metrics.
- `training_time_seconds`: wall-clock training time for the included adapter.
- `storage_bytes`: selected episode storage.
- Pareto frontiers: lower storage is better; lower MSE/MAE is better; higher F1 is better.

## Repository Layout

```text
configs/                         Experiment configuration
docs/                            Detailed method and integration guide
episode_selection/               Python package
scripts/                         CLI entry points
tests/                           Unit and smoke tests
examples/smoke/                  Smoke-test usage notes
outputs/                         Local generated artifacts, gitignored
```

## Installation

Recommended conventional setup:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

For plots, `dev` already includes matplotlib. For a smaller runtime-only install:

```bash
python -m pip install -e .
```

If you prefer `uv`, the same checks can be run without a persistent environment:

```bash
uv run --with pytest --with numpy --with pyyaml --with matplotlib python -m pytest -q
```

## Quickstart Smoke Test

Run a complete synthetic smoke experiment:

```bash
python scripts/run_episode_selection_experiment.py \
  --config configs/episode_selection_bc.yaml \
  --output-dir outputs/episode_selection_bc_smoke \
  --smoke-test \
  --overwrite
```

The smoke test creates a tiny synthetic dataset, runs all selectors, trains the lightweight linear adapter, writes aggregate tables, and generates plots when matplotlib is installed.

**Smoke-test artifacts are generated from synthetic data and are not research results.**

## Running a Real Experiment

Edit [configs/episode_selection_bc.yaml](configs/episode_selection_bc.yaml) with your dataset path, adapter keys, embedding settings, budgets, and output directory. Then run:

```bash
python scripts/run_episode_selection_experiment.py \
  --config configs/episode_selection_bc.yaml \
  --output-dir outputs/episode_selection_bc
```

For real research use, replace or wrap the lightweight BC adapter with your production behavior-cloning trainer and configure production embeddings. The deterministic hash embedder is only for tests and smoke runs.

## Standalone Analysis

Regenerate aggregate tables and plots from completed runs:

```bash
python scripts/analyze_episode_selection_results.py \
  --input-dir outputs/episode_selection_bc_smoke \
  --output-dir outputs/episode_selection_bc_smoke/reanalysis \
  --overwrite
```

## Expected Outputs

```text
outputs/episode_selection_bc/
  metadata/
    episode_metadata.jsonl
  runs/
    <selector>/<budget>/<seed>/
      selected_episodes.json
      metrics.json
      config_resolved.json
  aggregate/
    results.csv
    summary.csv
    pareto_points.csv
  plots/
    action_mse_vs_storage.png
    action_mae_vs_storage.png
    gripper_f1_vs_storage.png
    training_time_seconds_vs_storage.png
    pareto_action_mse.png
    pareto_action_mae.png
    pareto_gripper_f1.png
```

Example result artifacts are generated locally under `outputs/`; they are intentionally gitignored.

## Data Format Overview

Datasets may be a JSON file, JSONL file, NPZ file, or a directory of those files. JSON may be either a list of episode records or an object with an `episodes` list.

Minimal episode:

```json
{
  "episode_id": "episode_0001",
  "observations": [[0.0, 1.0], [0.1, 1.1]],
  "actions": [[0.2, 0.3], [0.25, 0.35]],
  "gripper": [0, 1],
  "language": "pick up the red block",
  "success": 1
}
```

Field names are configurable for language, image, action, and gripper keys.

## Configuration Overview

[configs/episode_selection_bc.yaml](configs/episode_selection_bc.yaml) controls:

- dataset path and adapter keys
- train/validation/test split seed and fractions
- selectors, budgets, and random seeds
- metadata cache and embedding settings
- quality-score weights
- MMR relevance/diversity settings
- training and evaluation settings
- plotting and output directory

Budgets support fractions, percentages, exact episode counts, and byte strings such as `500MB`.

## Testing

```bash
python -m pytest -q
```

The test suite uses deterministic synthetic fixtures and does not download models or require real robot data.

## Data and Artifact Policy

- Do not commit robot datasets.
- Do not commit large model checkpoints.
- Do not commit large generated outputs.
- Keep local artifacts under `outputs/`.
- Keep real datasets under external storage or ignored local directories such as `data/` or `datasets/`.
- Commit only lightweight examples and documentation.

## Current Limitations

- The default trainer is a NumPy linear behavior-cloning adapter for smoke and integration validation.
- Production semantic embeddings are not implemented in this repo.
- Deterministic embeddings are explicit test/smoke tooling, not semantic models for real experiments.
- Real paper claims require a real robot dataset, production embeddings, and a production BC training backend.

## Roadmap

- Add adapters for common robot dataset formats.
- Add production embedding backends such as sentence-transformers or CLIP behind optional extras.
- Add trainer integration hooks for project-specific BC pipelines.
- Add richer statistical summaries across repeated seeds.
- Add example notebooks using only synthetic or public toy data.

## Citation

If this framework helps your work, please cite the repository. A placeholder [CITATION.cff](CITATION.cff) is included and should be edited by the repository owner before archival release.

## License

This repository includes an MIT License in [LICENSE](LICENSE). Repository owners should confirm this is the intended license before public release.
