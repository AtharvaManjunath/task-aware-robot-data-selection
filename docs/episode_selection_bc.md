# Episode Selection Behavior-Cloning Experiment Guide

This document is the detailed method and integration guide for the semantic and quality-aware episode-selection framework. The root `README.md` is the landing page; this file explains data schemas, selector behavior, budgets, metrics, and how to adapt the framework to real robot experiments.

## GitHub Reviewer Quick Path

From a clean clone:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python -m pytest -q
python scripts/run_episode_selection_experiment.py \
  --config configs/episode_selection_bc.yaml \
  --output-dir outputs/episode_selection_bc_smoke \
  --smoke-test \
  --overwrite
python scripts/analyze_episode_selection_results.py \
  --input-dir outputs/episode_selection_bc_smoke \
  --output-dir outputs/episode_selection_bc_smoke/reanalysis \
  --overwrite
```

Inspect:

- `outputs/episode_selection_bc_smoke/aggregate/results.csv`
- `outputs/episode_selection_bc_smoke/aggregate/summary.csv`
- `outputs/episode_selection_bc_smoke/aggregate/pareto_points.csv`
- `outputs/episode_selection_bc_smoke/plots/`

Smoke-test artifacts are generated from synthetic data and are not research results.

## Research Motivation

Behavior-cloning pipelines often train on every available demonstration episode. This framework tests whether selecting semantically relevant, diverse, or high-quality episodes can reduce storage and training cost while preserving action-prediction metrics on a fixed evaluation split.

The current repository provides framework, tests, and smoke validation only. Real conclusions require a real robot dataset, production embeddings, and a production behavior-cloning trainer.

## Supported Dataset Layouts

### JSON

```json
{
  "episodes": [
    {
      "episode_id": "episode_0001",
      "observations": [[0.0, 1.0], [0.1, 1.1]],
      "actions": [[0.2, 0.3], [0.25, 0.35]],
      "gripper": [0, 1],
      "language": "pick up the red block",
      "success": 1,
      "reward": 0.92
    }
  ]
}
```

JSON may also be a bare list of episode objects.

### JSONL

Each line is one episode object:

```jsonl
{"episode_id":"episode_0001","observations":[[0.0]],"actions":[[0.1]],"language":"pick"}
{"episode_id":"episode_0002","observations":[[1.0]],"actions":[[1.1]],"language":"place"}
```

### NPZ

Each `.npz` file represents one episode with arrays such as:

- `episode_id`
- `observations`
- `actions`
- `gripper`
- `images`

### Directory

A directory may contain `.json`, `.jsonl`, and `.npz` files. Traversal is deterministic by sorted filename.

## Episode Fields

Required:

- `episode_id`
- `actions`

Recommended:

- `observations`
- `gripper`
- `language`, `instruction`, or `task_description`
- `task`, `task_id`, or `task_name`
- image frames under `images`, `frames`, or `rgb`

Optional quality signals:

- `success`
- `reward`
- `human_rating` or `rating`
- `completion` or `completed`
- `intervention_free`

Adapter key names are configured in `configs/episode_selection_bc.yaml`.

## Configuration Fields

Key sections:

- `dataset`: dataset path and field names.
- `output_dir`: where local artifacts are written.
- `selectors`: subset selectors to run.
- `budgets` and `budget_type`: storage or episode-count budgets.
- `budget.allow_oversized_first`: whether tiny budgets may select one oversized episode.
- `seeds`: repeated selector/training seeds.
- `include_full_data_baseline`: adds the full training split baseline.
- `split`: deterministic train/validation/test split settings.
- `metadata`: embedding, frame sampling, multimodal weighting, and quality settings.
- `selector_config`: query aggregation, MMR lambda, and fallback behavior.
- `training`: lightweight adapter settings.
- `evaluation`: gripper threshold and metric settings.
- `plotting`: plot generation toggle.

## Metadata Extraction

Metadata includes:

- episode ID
- transition count
- storage bytes and whether storage is estimated
- language text
- language, vision, and multimodal embeddings when available
- quality score and components
- task label
- success and reward fields

The cache is written as JSONL under `metadata/episode_metadata.jsonl`. It includes a fingerprint over metadata config, adapter choices, episode IDs, file path/size/mtime, transition counts, language/task identity, and whether image/gripper fields exist. Stale caches are not reused silently.

## Embeddings

The included deterministic hash embedder exists for tests and smoke runs. It is not a production semantic embedding model.

For real experiments, add a production embedding provider behind `episode_selection.embeddings.build_embedder`, then set:

```yaml
metadata:
  embeddings:
    provider: your_provider_name
    allow_deterministic_fallback: false
```

Vision embeddings use sampled representative frames. Supported frame sampling strategies are `first`, `middle`, `last`, and `uniform`.

Multimodal embeddings combine normalized language and vision embeddings with `multimodal_language_weight`.

## Selectors

### Random

Uniform seeded random episode ordering under budget.

### Language Similarity

Ranks episodes by cosine similarity between normalized language embeddings and target queries. If queries are omitted, task labels are used when available. Multiple queries use `max` or `mean` aggregation.

### Language + Vision Diversity

Uses MMR when queries exist:

```text
score = lambda * relevance_to_query - (1 - lambda) * max_similarity_to_selected
```

Without queries, it performs greedy farthest-first diversity. If vision embeddings are unavailable, language-only fallback requires `allow_language_only_fallback: true`.

### Quality Filtered

Ranks episodes by quality score. Higher is better. Explicit signals are used when present; proxy signals are used for validity, length sanity, and action smoothness.

### Full Data

Selects the full training split. It is included by default as a labeled baseline.

## Quality Scoring

Explicit components:

- success
- reward, normalized across the candidate dataset during metadata extraction
- human rating
- completion
- intervention-free flag

Proxy components:

- finite non-empty action sequence
- configured length sanity
- action smoothness from the second difference of action trajectories

Weights are configured under `metadata.quality.weights`.

## Budget Enforcement

Budgets support:

- storage fraction: `0.2` with `budget_type: storage_fraction`
- percentage string: `20%`
- exact episode count: `episodes:10` or `10episodes`
- byte strings: `500MB`, `2GB`

Byte budgets are strict unless `budget.allow_oversized_first: true` is enabled. Exact episode-count budgets select exactly that count unless the candidate set is smaller.

## Behavior-Cloning Adapter

The included `episode_selection.bc` module trains a small ridge-regularized linear model in NumPy. It is useful for:

- unit and smoke tests
- end-to-end pipeline validation
- checking output schemas and analysis tooling

It is not a full robot policy trainer. For real experiments, replace or wrap:

- `train_linear_bc`
- `evaluate_linear_bc`
- the runner call sites in `episode_selection.runner`

Keep the same selected episode IDs, fixed splits, evaluation batches, and metric output schema to preserve fair comparisons.

## Metrics

- `action_mse`: mean squared error over continuous action dimensions.
- `action_mae`: mean absolute error over continuous action dimensions.
- `gripper_precision`: binary gripper precision.
- `gripper_recall`: binary gripper recall.
- `gripper_f1`: binary gripper F1.
- `training_time_seconds`: wall-clock time for training.
- `storage_bytes`: selected storage.

NaN and shape issues are treated as errors rather than silently ignored when they would make metrics misleading.

## Pareto Analysis

For MSE and MAE, lower metric and lower storage are better. For F1, higher metric and lower storage are better. `aggregate/pareto_points.csv` contains nondominated points per selector and metric.

Plots are generated when matplotlib is installed:

- metric vs storage curves
- Pareto frontier plots

## Real-Data Integration Guidance

1. Put real datasets outside git, for example under ignored `data/` or external storage.
2. Edit `configs/episode_selection_bc.yaml` to point at the dataset and correct field names.
3. Add production embeddings and disable deterministic fallback.
4. Replace the lightweight BC adapter with the project trainer.
5. Run a small real-data dry run.
6. Run the full selector/budget/seed sweep.
7. Archive configs, selected episode IDs, metrics, and aggregate outputs.

## Data and Artifact Policy

Do not commit:

- robot datasets
- model checkpoints
- private videos/images
- large generated outputs
- metadata caches from real experiments

Use `outputs/` for local artifacts. It is gitignored.

## Known Limitations

- No real robot dataset is included.
- No production embedding backend is bundled.
- The default trainer is intentionally lightweight.
- Synthetic smoke metrics do not imply real-world method performance.
- Statistical analysis is currently limited to mean/std summaries and Pareto frontiers.
