from __future__ import annotations

import cv2
import numpy as np

from scene_analysis.utils import ensure_float32_array


def _validate_percentiles(min_percentile: float, max_percentile: float) -> tuple[float, float]:
    min_value = float(min_percentile)
    max_value = float(max_percentile)
    if not 0.0 <= min_value <= 100.0 or not 0.0 <= max_value <= 100.0:
        raise ValueError("Depth visualization percentiles must be in the range [0, 100]")
    if min_value >= max_value:
        raise ValueError("Depth visualization min percentile must be smaller than max percentile")
    return min_value, max_value


def normalize_depth_for_display(
    depth_map: np.ndarray,
    min_percentile: float = 2.0,
    max_percentile: float = 98.0,
) -> np.ndarray:
    """Нормализовать depth map для визуализации"""
    if not isinstance(depth_map, np.ndarray) or depth_map.size == 0:
        raise ValueError("Depth map must be a non-empty numpy array")
    if depth_map.ndim != 2:
        raise ValueError("Depth map must be a 2D array")

    min_value, max_value = _validate_percentiles(min_percentile, max_percentile)
    depth = ensure_float32_array(depth_map)
    finite_mask = np.isfinite(depth)
    if not np.any(finite_mask):
        raise ValueError("Depth map does not contain finite values")

    valid_depth = depth[finite_mask]
    lower = float(np.percentile(valid_depth, min_value))
    upper = float(np.percentile(valid_depth, max_value))

    if abs(upper - lower) < 1e-8:
        normalized = np.zeros(depth.shape, dtype=np.uint8)
        normalized[finite_mask] = 255
        return normalized

    clipped = np.clip(depth, lower, upper)
    normalized_float = (clipped - lower) / (upper - lower)
    normalized_float[~finite_mask] = 0.0
    return np.clip(normalized_float * 255.0, 0.0, 255.0).astype(np.uint8)


def colorize_depth_map(
    depth_map: np.ndarray,
    min_percentile: float,
    max_percentile: float,
) -> np.ndarray:
    """Преобразовать depth map в цветное изображение"""
    grayscale = normalize_depth_for_display(
        depth_map=depth_map,
        min_percentile=min_percentile,
        max_percentile=max_percentile,
    )
    return cv2.applyColorMap(grayscale, cv2.COLORMAP_INFERNO)
