from __future__ import annotations

import numpy as np

from scene_analysis.utils import ensure_float32_array, maybe_resize_float_map


def normalize_prediction_map(
    pred: np.ndarray,
    clip_to_unit_range: bool = True,
) -> np.ndarray:
    """Привести prediction heatmap к float32"""
    prediction = ensure_float32_array(pred)
    prediction = np.nan_to_num(prediction, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32, copy=False)
    if clip_to_unit_range:
        prediction = np.clip(prediction, 0.0, 1.0)
    return prediction.astype(np.float32, copy=False)


def resize_prediction_to_mask(
    pred: np.ndarray,
    mask_shape: tuple[int, int],
) -> np.ndarray:
    """Подогнать prediction heatmap к размеру GT mask"""
    return maybe_resize_float_map(pred, mask_shape)


def build_valid_label_mask(
    gt_mask: np.ndarray,
    obstacle_values: list[int],
    background_values: list[int],
    ignore_values: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Построить valid_mask и positive_mask с учетом ignore"""
    mask = np.asarray(gt_mask)
    if mask.ndim != 2:
        raise ValueError("Ground truth mask must be 2D")

    obstacle_mask = np.isin(mask, obstacle_values)
    background_mask = np.isin(mask, background_values)
    ignore_mask = np.isin(mask, ignore_values)

    valid_mask = (obstacle_mask | background_mask) & (~ignore_mask)
    positive_mask = obstacle_mask & valid_mask
    return valid_mask.astype(bool, copy=False), positive_mask.astype(bool, copy=False)


def flatten_scores_and_labels(
    pred: np.ndarray,
    valid_mask: np.ndarray,
    positive_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Преобразовать 2D карты в плоские массивы scores и binary labels"""
    prediction = ensure_float32_array(pred)
    if prediction.ndim != 2:
        raise ValueError("Prediction heatmap must be 2D")
    if prediction.shape != valid_mask.shape or prediction.shape != positive_mask.shape:
        raise ValueError("Prediction map and masks must have the same shape")

    scores_flat = prediction[valid_mask].astype(np.float32, copy=False)
    labels_flat = positive_mask[valid_mask].astype(np.uint8, copy=False)
    return scores_flat, labels_flat
