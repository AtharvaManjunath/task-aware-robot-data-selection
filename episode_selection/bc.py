"""Small reproducible behavior cloning adapter used when no project trainer exists."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .data import Episode, transitions_for_ids
from .metrics import action_mae, action_mse, gripper_f1


@dataclass
class LinearBCModel:
    weights: np.ndarray
    gripper_weights: Optional[np.ndarray] = None

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, Optional[np.ndarray]]:
        design = np.c_[np.asarray(x, dtype=float), np.ones(len(x))]
        action_pred = design @ self.weights
        grip = None if self.gripper_weights is None else design @ self.gripper_weights
        return action_pred, grip


def train_linear_bc(episodes: list[Episode], selected_ids: list[str], *, ridge: float = 1e-4) -> tuple[LinearBCModel, float]:
    start = time.perf_counter()
    x, y, gripper = transitions_for_ids(episodes, selected_ids)
    design = np.c_[x, np.ones(len(x))]
    reg = ridge * np.eye(design.shape[1])
    weights = np.linalg.solve(design.T @ design + reg, design.T @ y)
    grip_w = None
    if gripper is not None:
        grip_w = np.linalg.solve(design.T @ design + reg, design.T @ gripper.astype(float))
    return LinearBCModel(weights, grip_w), time.perf_counter() - start


def evaluate_linear_bc(model: LinearBCModel, episodes: list[Episode], eval_ids: list[str], *, gripper_threshold: float = 0.5) -> dict:
    x, y, gripper = transitions_for_ids(episodes, eval_ids)
    pred, grip_pred = model.predict(x)
    metrics = {"action_mse": action_mse(pred, y), "action_mae": action_mae(pred, y)}
    if gripper is not None:
        if grip_pred is None:
            raise ValueError("Gripper labels exist but model has no gripper head")
        metrics.update({f"gripper_{k}": v for k, v in gripper_f1(grip_pred, gripper, gripper_threshold).items()})
    return metrics
