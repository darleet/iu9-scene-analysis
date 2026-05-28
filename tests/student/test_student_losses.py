from __future__ import annotations

import pytest
import torch

from scene_analysis.student.config import StudentLossConfig
from scene_analysis.student.losses import StudentHeatmapLoss, build_teacher_soft_target


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


def test_distillation_loss_zeros_teacher_background() -> None:
    criterion = StudentHeatmapLoss(
        StudentLossConfig(
            bce_weight=0.0,
            dice_weight=0.0,
            distill_mse_weight=1.0,
            offroad_weight=0.0,
        )
    )
    prob = torch.zeros((1, 1, 4, 4))
    outputs = {
        "obstacle_logits": torch.zeros_like(prob),
        "obstacle_prob": prob,
        "final_heatmap": prob,
    }
    obstacle_target = torch.zeros_like(prob)
    obstacle_target[:, :, 1, 2] = 1.0
    valid_mask = torch.ones_like(prob)
    ignore_mask = torch.zeros_like(prob)
    teacher = torch.ones_like(prob)

    loss, parts = criterion(outputs, obstacle_target, valid_mask, ignore_mask, teacher)

    assert torch.isclose(parts["loss_distill"], torch.tensor(1.0 / 16.0))
    assert torch.isclose(loss, torch.tensor(1.0 / 16.0))


def test_distillation_loss_does_not_expand_gt_gate() -> None:
    criterion = StudentHeatmapLoss(
        StudentLossConfig(
            bce_weight=0.0,
            dice_weight=0.0,
            distill_mse_weight=1.0,
            offroad_weight=0.0,
        )
    )
    prob = torch.zeros((1, 1, 5, 5))
    outputs = {
        "obstacle_logits": torch.zeros_like(prob),
        "obstacle_prob": prob,
        "final_heatmap": prob,
    }
    obstacle_target = torch.zeros_like(prob)
    obstacle_target[:, :, 2, 2] = 1.0
    valid_mask = torch.ones_like(prob)
    ignore_mask = torch.zeros_like(prob)
    teacher = torch.ones_like(prob)

    _, parts = criterion(outputs, obstacle_target, valid_mask, ignore_mask, teacher)

    assert torch.isclose(parts["loss_distill"], torch.tensor(1.0 / 25.0))


def test_mask_losses_can_use_teacher_soft_target() -> None:
    criterion = StudentHeatmapLoss(
        StudentLossConfig(
            bce_weight=1.0,
            dice_weight=0.0,
            distill_mse_weight=0.0,
            use_teacher_soft_target=True,
            teacher_soft_target_alpha=0.0,
            offroad_weight=0.0,
            positive_class_weight=1.0,
        )
    )
    logits = torch.full((1, 1, 2, 2), -10.0)
    outputs = {
        "obstacle_logits": logits,
        "obstacle_prob": torch.sigmoid(logits),
        "final_heatmap": torch.sigmoid(logits),
    }
    obstacle_target = torch.ones_like(logits)
    valid_mask = torch.ones_like(logits)
    ignore_mask = torch.zeros_like(logits)
    teacher = torch.zeros_like(logits)

    loss, parts = criterion(outputs, obstacle_target, valid_mask, ignore_mask, teacher)

    assert parts["loss_bce"] < torch.tensor(0.001)
    assert loss < torch.tensor(0.001)


def test_teacher_soft_target_blends_gt_and_teacher_heatmap() -> None:
    teacher = torch.tensor([[[[0.0, 0.5, 1.0, 1.0]]]], dtype=torch.float32)
    obstacle_target = torch.tensor([[[[1.0, 1.0, 1.0, 0.0]]]], dtype=torch.float32)

    target = build_teacher_soft_target(teacher, obstacle_target, alpha=0.2)

    expected = torch.tensor([[[[0.2, 0.6, 1.0, 0.0]]]], dtype=torch.float32)
    assert torch.allclose(target, expected)


def test_teacher_soft_target_uses_neutral_bce_class_weight() -> None:
    criterion = StudentHeatmapLoss(
        StudentLossConfig(
            bce_weight=1.0,
            dice_weight=0.0,
            distill_mse_weight=0.0,
            use_teacher_soft_target=True,
            offroad_weight=0.0,
            positive_class_weight=100.0,
        )
    )
    logits = torch.zeros((1, 1, 2, 2))
    outputs = {
        "obstacle_logits": logits,
        "obstacle_prob": torch.sigmoid(logits),
        "final_heatmap": torch.sigmoid(logits),
    }
    obstacle_target = torch.ones_like(logits)
    valid_mask = torch.ones_like(logits)
    ignore_mask = torch.zeros_like(logits)
    teacher = torch.full_like(logits, 0.5)

    _, parts = criterion(outputs, obstacle_target, valid_mask, ignore_mask, teacher)

    assert torch.isclose(parts["loss_bce"], torch.tensor(0.6931472), atol=1e-6)


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
