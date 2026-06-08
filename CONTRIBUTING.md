# Contributing

Thank you for helping improve this research software. The project is intended to stay reproducible, lightweight, and honest about what has and has not been experimentally validated.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

## Tests

Run the full suite:

```bash
python -m pytest -q
```

Run the smoke experiment:

```bash
python scripts/run_episode_selection_experiment.py \
  --config configs/episode_selection_bc.yaml \
  --output-dir outputs/episode_selection_bc_smoke \
  --smoke-test \
  --overwrite
```

## Pull Requests

- Keep core research logic modular and tested.
- Add or update tests for selector, budget, metric, cache, or data-loader changes.
- Do not commit real robot datasets, model checkpoints, large generated outputs, or private experiment logs.
- Keep synthetic examples clearly labeled as synthetic.
- Do not present smoke-test metrics as research findings.
- Prefer small, reviewable PRs with clear motivation.

## Documentation

Update `README.md` for user-facing workflow changes and `docs/episode_selection_bc.md` for method or integration details.

## Dependency Policy

Avoid heavyweight dependencies unless they are optional and clearly documented. Tests must not require internet downloads or large pretrained models.
