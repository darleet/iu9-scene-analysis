from __future__ import annotations

import numpy as np

from scene_analysis.depth.visualization import colorize_depth_map, normalize_depth_for_display


def test_normalize_depth_for_display_returns_uint8_with_same_shape() -> None:
    depth_map = np.linspace(0.0, 10.0, num=20, dtype=np.float32).reshape(4, 5)

    normalized = normalize_depth_for_display(depth_map)

    assert normalized.dtype == np.uint8
    assert normalized.shape == depth_map.shape


def test_colorize_depth_map_returns_bgr_image() -> None:
    depth_map = np.linspace(1.0, 5.0, num=12, dtype=np.float32).reshape(3, 4)

    colorized = colorize_depth_map(depth_map, min_percentile=2.0, max_percentile=98.0)

    assert colorized.dtype == np.uint8
    assert colorized.shape == (3, 4, 3)


def test_constant_depth_map_does_not_fail() -> None:
    depth_map = np.full((6, 8), fill_value=3.14, dtype=np.float32)

    normalized = normalize_depth_for_display(depth_map)
    colorized = colorize_depth_map(depth_map, min_percentile=2.0, max_percentile=98.0)

    assert normalized.shape == depth_map.shape
    assert colorized.shape == (6, 8, 3)
