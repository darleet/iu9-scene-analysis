from __future__ import annotations

import numpy as np

try:
    from sklearn.metrics import average_precision_score, precision_recall_curve
except ImportError:
    average_precision_score = None
    precision_recall_curve = None


def compute_average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    prepared_scores, prepared_labels = _prepare_scores_and_labels(scores, labels)
    if not _is_valid_binary_problem(prepared_labels):
        return float("nan")

    if average_precision_score is not None:
        return float(average_precision_score(prepared_labels, prepared_scores))

    precision, recall, _ = compute_precision_recall_curve_data(prepared_scores, prepared_labels)
    if precision.size == 0 or recall.size == 0:
        return float("nan")
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def compute_precision_recall_curve_data(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Подготовить данные PR-кривой в формате recall asc [0..1]"""
    prepared_scores, prepared_labels = _prepare_scores_and_labels(scores, labels)
    if not _is_valid_binary_problem(prepared_labels):
        return (
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
        )

    if precision_recall_curve is not None:
        precision, recall, thresholds = precision_recall_curve(prepared_labels, prepared_scores)
        precision = precision[::-1].astype(np.float32, copy=False)
        recall = recall[::-1].astype(np.float32, copy=False)
        thresholds = thresholds[::-1].astype(np.float32, copy=False)
        return precision, recall, thresholds

    order = np.argsort(-prepared_scores, kind="mergesort")
    sorted_scores = prepared_scores[order]
    sorted_labels = prepared_labels[order]

    true_positives = np.cumsum(sorted_labels == 1)
    false_positives = np.cumsum(sorted_labels == 0)
    total_positives = int(np.sum(sorted_labels == 1))

    precision = true_positives / np.maximum(true_positives + false_positives, 1)
    recall = true_positives / total_positives

    precision_curve = np.concatenate(([1.0], precision.astype(np.float32, copy=False)))
    recall_curve = np.concatenate(([0.0], recall.astype(np.float32, copy=False)))
    return (
        precision_curve.astype(np.float32, copy=False),
        recall_curve.astype(np.float32, copy=False),
        sorted_scores.astype(np.float32, copy=False),
    )


def compute_dataset_ap(all_scores: list[np.ndarray], all_labels: list[np.ndarray]) -> float:
    """Посчитать один global AP по всем валидным пикселям датасета"""
    if not all_scores or not all_labels:
        return float("nan")

    concatenated_scores = np.concatenate([np.asarray(scores, dtype=np.float32) for scores in all_scores])
    concatenated_labels = np.concatenate([np.asarray(labels, dtype=np.uint8) for labels in all_labels])
    return compute_average_precision(concatenated_scores, concatenated_labels)


def _prepare_scores_and_labels(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    prepared_scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    prepared_labels = np.asarray(labels, dtype=np.uint8).reshape(-1)
    if prepared_scores.shape != prepared_labels.shape:
        raise ValueError("Scores and labels must have the same flattened shape")
    return prepared_scores, prepared_labels


def _is_valid_binary_problem(labels: np.ndarray) -> bool:
    if labels.size == 0:
        return False
    has_positive = bool(np.any(labels == 1))
    has_negative = bool(np.any(labels == 0))
    return has_positive and has_negative
