from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from scene_analysis.student.config import StudentLossConfig


def masked_bce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    positive_class_weight: float,
) -> torch.Tensor:
    valid_pixels = _valid_pixel_count(valid_mask)
    logits_float = logits.float()
    target_float = target.float()
    valid_float = valid_mask.float()
    pos_weight = torch.as_tensor(
        positive_class_weight,
        dtype=logits_float.dtype,
        device=logits_float.device,
    )
    bce = F.binary_cross_entropy_with_logits(
        logits_float,
        target_float,
        pos_weight=pos_weight,
        reduction="none",
    )
    return (bce * valid_float).sum() / valid_pixels


def masked_dice_loss(
    pred_prob: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    valid_pixels = _valid_pixel_count(valid_mask)
    valid_float = valid_mask.float()
    pred = pred_prob.float() * valid_float
    gt = target.float() * valid_float
    intersection = (pred * gt).sum()
    denominator = pred.sum() + gt.sum()
    dice = (2.0 * intersection + eps) / (denominator + eps)
    if valid_pixels <= 0:
        raise ValueError("Batch contains zero valid pixels")
    return 1.0 - dice


def distillation_mse(
    student_heatmap: torch.Tensor,
    teacher_heatmap: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    valid_pixels = _valid_pixel_count(valid_mask)
    valid_float = valid_mask.float()
    mse = (student_heatmap.float() - teacher_heatmap.float()).pow(2)
    return (mse * valid_float).sum() / valid_pixels


def build_teacher_soft_target(
    teacher_heatmap: torch.Tensor,
    obstacle_target: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    gate = (obstacle_target.float() > 0.5).float()
    alpha_value = float(alpha)
    teacher_foreground = teacher_heatmap.float().clamp(0.0, 1.0) * gate
    return alpha_value * gate + (1.0 - alpha_value) * teacher_foreground


def offroad_loss(student_heatmap: torch.Tensor, ignore_mask: torch.Tensor) -> torch.Tensor:
    """Loss для объектов вне дороги"""
    ignore_float = ignore_mask.float()
    ignore_pixels = ignore_float.sum()
    if float(ignore_pixels.detach().cpu()) <= 0.0:
        return student_heatmap.sum() * 0.0
    return (student_heatmap.float() * ignore_float).sum() / ignore_pixels.clamp_min(1.0)


class StudentHeatmapLoss(nn.Module):
    def __init__(self, config: StudentLossConfig) -> None:
        super().__init__()
        self.config = config

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        obstacle_target: torch.Tensor,
        valid_mask: torch.Tensor,
        ignore_mask: torch.Tensor,
        teacher_heatmap: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        distill_target = build_teacher_soft_target(
            teacher_heatmap=teacher_heatmap,
            obstacle_target=obstacle_target,
            alpha=self.config.teacher_soft_target_alpha,
        )
        supervised_target = distill_target if self.config.use_teacher_soft_target else obstacle_target
        bce_positive_class_weight = 1.0 if self.config.use_teacher_soft_target else self.config.positive_class_weight

        loss_bce = masked_bce_loss(
            outputs["obstacle_logits"],
            supervised_target,
            valid_mask,
            bce_positive_class_weight,
        )
        loss_dice = masked_dice_loss(outputs["obstacle_prob"], supervised_target, valid_mask, self.config.eps)
        loss_distill = distillation_mse(outputs["final_heatmap"], distill_target, valid_mask)
        loss_offroad = offroad_loss(outputs["final_heatmap"], ignore_mask)

        total = (
            self.config.bce_weight * loss_bce
            + self.config.dice_weight * loss_dice
            + self.config.distill_mse_weight * loss_distill
            + self.config.offroad_weight * loss_offroad
        )
        if not torch.isfinite(total):
            raise FloatingPointError("Student loss became non-finite")
        return total, {
            "loss_bce": loss_bce.detach(),
            "loss_dice": loss_dice.detach(),
            "loss_distill": loss_distill.detach(),
            "loss_offroad": loss_offroad.detach(),
        }


def _valid_pixel_count(valid_mask: torch.Tensor) -> torch.Tensor:
    valid_pixels = valid_mask.sum()
    if float(valid_pixels.detach().cpu()) <= 0.0:
        raise ValueError("Batch contains zero valid pixels; cannot compute student heatmap loss")
    return valid_pixels.clamp_min(1.0)
