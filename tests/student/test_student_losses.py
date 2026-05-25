from __future__ import annotations

import pytest
import torch

from scene_analysis.student.config import StudentLossConfig
from scene_analysis.student.losses import StudentHeatmapLoss


def test_student_loss_is_finite() -> None:
    criterion = StudentHeatmapLoss(StudentLossConfig())
    obstacle_logits = torch.randn(2, 1, 8, 12)
    obstacle_prob = torch.sigmoid(obstacle_logits)
    outputs = {
        "obstacle_logits": obstacle_logits,
        "obstacle_prob": obstacle_prob,
        "final_heatmap": obstacle_prob,
    }
    obstacle_target = torch.zeros(2, 1, 8, 12)
    obstacle_target[:, :, 2:4, 3:6] = 1.0
    valid_mask = torch.ones(2, 1, 8, 12)
    ignore_mask = torch.zeros(2, 1, 8, 12)
    teacher = obstacle_target.clone()

    loss, parts = criterion(outputs, obstacle_target, valid_mask, ignore_mask, teacher)

    assert torch.isfinite(loss)
    assert set(parts) == {"loss_bce", "loss_dice", "loss_distill", "loss_offroad"}


def test_student_loss_handles_extreme_half_precision_logits() -> None:
    criterion = StudentHeatmapLoss(StudentLossConfig())
    obstacle_logits = torch.tensor([[[[20.0, -20.0], [20.0, -20.0]]]], dtype=torch.float16)
    obstacle_prob = torch.sigmoid(obstacle_logits)
    outputs = {
        "obstacle_logits": obstacle_logits,
        "obstacle_prob": obstacle_prob,
        "final_heatmap": obstacle_prob,
    }
    obstacle_target = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=torch.float16)
    valid_mask = torch.ones_like(obstacle_target)
    ignore_mask = torch.zeros_like(obstacle_target)
    teacher = torch.zeros_like(obstacle_target)

    loss, parts = criterion(outputs, obstacle_target, valid_mask, ignore_mask, teacher)

    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in parts.values())


def test_student_loss_raises_on_zero_valid_pixels() -> None:
    criterion = StudentHeatmapLoss(StudentLossConfig())
    prob = torch.full((1, 1, 4, 4), 0.5)
    outputs = {
        "obstacle_logits": torch.zeros_like(prob),
        "obstacle_prob": prob,
        "final_heatmap": prob,
    }

    with pytest.raises(ValueError, match="zero valid pixels"):
        criterion(
            outputs,
            torch.zeros_like(prob),
            torch.zeros_like(prob),
            torch.ones_like(prob),
            torch.zeros_like(prob),
        )
