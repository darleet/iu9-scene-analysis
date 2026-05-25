from __future__ import annotations

import numpy as np
import torch

from scene_analysis.evaluation.metrics import compute_average_precision


def collect_scores_and_labels(
    final_heatmap: torch.Tensor,
    obstacle_target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    """Сбор скоров и лейблов воедино (подготовка для подсчета AP)"""
    heatmap = final_heatmap.detach().float().cpu().numpy()
    target = obstacle_target.detach().float().cpu().numpy()
    valid = valid_mask.detach().float().cpu().numpy() > 0.5
    scores = heatmap[valid].astype(np.float32, copy=False)
    labels = (target[valid] > 0.5).astype(np.uint8, copy=False)
    return scores.reshape(-1), labels.reshape(-1)


def compute_global_average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    return compute_average_precision(scores, labels)


def compute_binary_confusion(
    final_heatmap: torch.Tensor,
    obstacle_target: torch.Tensor,
    valid_mask: torch.Tensor,
    threshold: float,
) -> dict[str, int]:
    pred = final_heatmap.detach().float().cpu() >= float(threshold)
    target = obstacle_target.detach().float().cpu() > 0.5
    valid = valid_mask.detach().float().cpu() > 0.5

    pred = pred & valid
    target = target & valid
    true_positive = pred & target
    false_positive = pred & (~target)
    false_negative = (~pred) & target & valid
    true_negative = (~pred) & (~target) & valid
    return {
        "tp": int(true_positive.sum().item()),
        "fp": int(false_positive.sum().item()),
        "fn": int(false_negative.sum().item()),
        "tn": int(true_negative.sum().item()),
    }


def compute_f1_iou_from_confusion(tp: float, fp: float, fn: float) -> dict[str, float]:
    f1_denominator = 2.0 * tp + fp + fn
    iou_denominator = tp + fp + fn
    return {
        "f1": float("nan") if f1_denominator <= 0.0 else float((2.0 * tp) / f1_denominator),
        "iou": float("nan") if iou_denominator <= 0.0 else float(tp / iou_denominator),
    }


def compute_heatmap_stats(
    final_heatmap: torch.Tensor,
    valid_mask: torch.Tensor,
    ignore_mask: torch.Tensor,
    obstacle_target: torch.Tensor | None = None,
) -> dict[str, float | int]:
    heatmap = final_heatmap.detach().float().cpu()
    valid = valid_mask.detach().float().cpu() > 0.5
    ignore = ignore_mask.detach().float().cpu() > 0.5
    valid_values = heatmap[valid]
    ignore_values = heatmap[ignore]

    stats: dict[str, float | int] = {
        "heatmap_min": float(torch.min(heatmap).item()) if heatmap.numel() else float("nan"),
        "heatmap_max": float(torch.max(heatmap).item()) if heatmap.numel() else float("nan"),
        "heatmap_mean": float(torch.mean(heatmap).item()) if heatmap.numel() else float("nan"),
        "valid_mean": float(torch.mean(valid_values).item()) if valid_values.numel() else float("nan"),
        "ignore_mean": float(torch.mean(ignore_values).item()) if ignore_values.numel() else 0.0,
        "valid_pixels": int(valid.sum().item()),
    }
    if obstacle_target is not None:
        target = obstacle_target.detach().float().cpu() > 0.5
        positives = target & valid
        stats["positive_pixels"] = int(positives.sum().item())
        stats["negative_pixels"] = int(valid.sum().item() - positives.sum().item())
    else:
        stats["positive_pixels"] = 0
        stats["negative_pixels"] = int(valid.sum().item())
    return stats
