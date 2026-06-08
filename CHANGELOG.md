# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

- Added episode metadata extraction with cache fingerprinting.
- Added random, language-similarity, language+vision diversity/MMR, quality-filtered, and full-data selectors.
- Added budget parsing and enforcement for storage fractions, percentages, bytes, and exact episode counts.
- Added deterministic synthetic smoke-test dataset generation.
- Added lightweight NumPy linear behavior-cloning adapter for integration checks.
- Added action MSE, action MAE, and gripper precision/recall/F1 metrics.
- Added aggregate CSV outputs, summary tables, Pareto frontier computation, and optional matplotlib plots.
- Added CLI scripts for experiments and standalone analysis.
- Added pytest coverage for budgets, selectors, data loading, metadata caching, metrics, Pareto logic, and smoke execution.
- Added GitHub-facing documentation, packaging metadata, CI, citation, contribution, and license files.
