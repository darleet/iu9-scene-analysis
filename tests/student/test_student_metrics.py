from __future__ import annotations

import numpy as np
import torch

from scene_analysis.student.metrics import (
    collect_scores_and_labels,
    compute_binary_confusion,
    compute_f1_iou_from_confusion,
    compute_global_average_precision,
    compute_heatmap_stats,
)


def test_student_metrics_compute_global_ap_and_stats() -> None:
    heatmap = torch.tensor([[[[0.9, 0.1], [0.8, 0.2]]]], dtype=torch.float32)
    target = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]], dtype=torch.float32)
    valid = torch.tensor([[[[1.0, 1.0], [1.0, 0.0]]]], dtype=torch.float32)
    ignore = 1.0 - valid

    scores, labels = collect_scores_and_labels(heatmap, target, valid)
    ap = compute_global_average_precision(scores, labels)
    stats = compute_heatmap_stats(heatmap, valid, ignore, target)

    assert np.isclose(ap, 1.0)
    assert scores.shape == (3,)
    assert labels.tolist() == [1, 0, 1]
    assert stats["valid_pixels"] == 3
    assert stats["positive_pixels"] == 2
    assert stats["negative_pixels"] == 1
    assert np.isclose(stats["ignore_mean"], 0.2)


def test_student_metrics_compute_f1_and_iou_from_binary_confusion() -> None:
    heatmap = torch.tensor([[[[0.9, 0.6], [0.1, 0.2]]]], dtype=torch.float32)
    target = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]], dtype=torch.float32)
    valid = torch.ones_like(target)

    confusion = compute_binary_confusion(heatmap, target, valid, threshold=0.5)
    metrics = compute_f1_iou_from_confusion(confusion["tp"], confusion["fp"], confusion["fn"])

    assert confusion == {"tp": 1, "fp": 1, "fn": 1, "tn": 1}
    assert np.isclose(metrics["f1"], 0.5)
    assert np.isclose(metrics["iou"], 1.0 / 3.0)
