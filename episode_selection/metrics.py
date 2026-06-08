"""Evaluation metrics for behavior cloning action prediction."""

from __future__ import annotations

from typing import Optional

import numpy as np


def _continuous_parts(pred: np.ndarray, true: np.ndarray, gripper_index: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(pred, dtype=float)
    t = np.asarray(true, dtype=float)
    if p.shape != t.shape:
        raise ValueError(f"Prediction and target shapes differ: {p.shape} vs {t.shape}")
    if gripper_index is not None and p.ndim == 2:
        if not -p.shape[1] <= gripper_index < p.shape[1]:
            raise ValueError(f"gripper_index {gripper_index} is out of bounds for action dimension {p.shape[1]}")
        p = np.delete(p, gripper_index, axis=1)
        t = np.delete(t, gripper_index, axis=1)
    mask = np.isfinite(p) & np.isfinite(t)
    if not np.any(mask):
        raise ValueError("No finite action values available for metric")
    return p[mask], t[mask]


def action_mse(pred: np.ndarray, true: np.ndarray, gripper_index: Optional[int] = None) -> float:
    p, t = _continuous_parts(pred, true, gripper_index)
    return float(np.mean(np.square(p - t)))


def action_mae(pred: np.ndarray, true: np.ndarray, gripper_index: Optional[int] = None) -> float:
    p, t = _continuous_parts(pred, true, gripper_index)
    return float(np.mean(np.abs(p - t)))


def gripper_f1(pred: np.ndarray, true: np.ndarray, threshold: float = 0.5) -> dict:
    if not np.isfinite(threshold):
        raise ValueError("Gripper threshold must be finite")
    p = np.asarray(pred)
    t = np.asarray(true)
    if p.ndim > 1 and p.shape[-1] > 1:
        p_bin = np.argmax(p, axis=-1).reshape(-1).astype(int)
    else:
        p_bin = (p.reshape(-1).astype(float) >= threshold).astype(int)
    if t.ndim > 1 and t.shape[-1] > 1:
        t_bin = np.argmax(t, axis=-1).reshape(-1).astype(int)
    else:
        t_bin = (t.reshape(-1).astype(float) >= threshold).astype(int)
    if p_bin.shape != t_bin.shape:
        raise ValueError(f"Gripper prediction and target lengths differ: {p_bin.shape} vs {t_bin.shape}")
    tp = int(np.sum((p_bin == 1) & (t_bin == 1)))
    fp = int(np.sum((p_bin == 1) & (t_bin == 0)))
    fn = int(np.sum((p_bin == 0) & (t_bin == 1)))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}
