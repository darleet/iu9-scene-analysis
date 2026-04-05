from __future__ import annotations

import numpy as np

from scene_analysis.evaluation.metrics import (
    compute_average_precision,
    compute_dataset_ap,
    compute_precision_recall_curve_data,
)


def test_compute_average_precision_on_perfect_ranking() -> None:
    scores = np.array([0.9, 0.8, 0.2, 0.1], dtype=np.float32)
    labels = np.array([1, 1, 0, 0], dtype=np.uint8)

    ap = compute_average_precision(scores, labels)

    assert np.isclose(ap, 1.0)


def test_compute_dataset_ap_aggregates_all_pixels() -> None:
    all_scores = [
        np.array([0.9, 0.2], dtype=np.float32),
        np.array([0.8, 0.1], dtype=np.float32),
    ]
    all_labels = [
        np.array([1, 0], dtype=np.uint8),
        np.array([1, 0], dtype=np.uint8),
    ]

    ap = compute_dataset_ap(all_scores, all_labels)
    precision, recall, thresholds = compute_precision_recall_curve_data(
        np.concatenate(all_scores),
        np.concatenate(all_labels),
    )

    assert np.isclose(ap, 1.0)
    assert precision.size > 0
    assert recall.size > 0
    assert thresholds.size > 0


def test_metric_edge_cases_return_nan_or_empty_curve() -> None:
    empty_ap = compute_average_precision(np.array([], dtype=np.float32), np.array([], dtype=np.uint8))
    positive_only_ap = compute_average_precision(
        np.array([0.9, 0.8], dtype=np.float32),
        np.array([1, 1], dtype=np.uint8),
    )
    precision, recall, thresholds = compute_precision_recall_curve_data(
        np.array([0.9, 0.8], dtype=np.float32),
        np.array([1, 1], dtype=np.uint8),
    )

    assert np.isnan(empty_ap)
    assert np.isnan(positive_only_ap)
    assert precision.size == 0
    assert recall.size == 0
    assert thresholds.size == 0
