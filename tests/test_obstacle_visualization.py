from __future__ import annotations

import numpy as np

from scene_analysis.obstacle_map.visualization import heatmap_to_bgr, overlay_heatmap_on_image


def test_heatmap_to_bgr_returns_uint8_bgr_image() -> None:
    heatmap = np.linspace(0.0, 1.0, num=64, dtype=np.float32).reshape(8, 8)

    result = heatmap_to_bgr(heatmap, colormap="inferno")

    assert result.shape == (8, 8, 3)
    assert result.dtype == np.uint8


def test_overlay_heatmap_on_image_returns_bgr_uint8_image() -> None:
    image = np.zeros((12, 16, 3), dtype=np.uint8)
    heatmap = np.linspace(0.0, 1.0, num=12 * 16, dtype=np.float32).reshape(12, 16)

    result = overlay_heatmap_on_image(image=image, heatmap=heatmap, alpha=0.5, colormap="turbo")

    assert result.shape == image.shape
    assert result.dtype == np.uint8


def test_constant_heatmap_does_not_crash_visualization() -> None:
    image = np.full((10, 10, 3), 64, dtype=np.uint8)
    heatmap = np.ones((10, 10), dtype=np.float32) * 0.25

    colorized = heatmap_to_bgr(heatmap, colormap="magma")
    overlay = overlay_heatmap_on_image(image=image, heatmap=heatmap, alpha=0.4)

    assert colorized.shape == (10, 10, 3)
    assert overlay.shape == image.shape
