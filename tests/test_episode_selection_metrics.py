import numpy as np
import pytest

from episode_selection.metrics import action_mae, action_mse, gripper_f1


def test_action_metrics():
    pred = np.array([[1.0, 2.0], [3.0, 4.0]])
    true = np.array([[1.0, 1.0], [1.0, 4.0]])
    assert action_mse(pred, true) == 1.25
    assert action_mae(pred, true) == 0.75


def test_gripper_f1():
    result = gripper_f1(np.array([0.9, 0.2, 0.8]), np.array([1, 0, 0]))
    assert result["precision"] == 0.5
    assert result["recall"] == 1.0
    assert round(result["f1"], 4) == 0.6667


def test_gripper_shape_mismatch_raises():
    with pytest.raises(ValueError, match="lengths differ"):
        gripper_f1(np.array([1, 0]), np.array([1]))
