from __future__ import annotations

import numpy as np

from scene_analysis.config import ObstacleHeatmapConfig
from scene_analysis.obstacle_map.heatmap_builder import DepthToObstacleHeatmapBuilder
from scene_analysis.types import DepthResult


def _make_test_depth(height: int = 48, width: int = 64) -> np.ndarray:
    road_profile = np.repeat(
        np.linspace(0.15, 1.0, num=height, dtype=np.float32)[:, None],
        width,
        axis=1,
    )
    road_profile[height // 3 : height // 2, width // 3 : width // 2] = 1.25
    return road_profile


def test_builder_returns_no_depth_status_when_depth_missing() -> None:
    builder = DepthToObstacleHeatmapBuilder(ObstacleHeatmapConfig())
    image = np.zeros((32, 48, 3), dtype=np.uint8)

    result = builder.build(
        DepthResult(depth_map=None, confidence_map=None, metadata={"scale_type": "relative"}),
        image=image,
    )

    assert result.heatmap is None
    assert result.metadata["status"] == "no_depth"
    assert result.metadata["depth_available"] is False


def test_builder_produces_unit_range_heatmap_with_expected_shape() -> None:
    builder = DepthToObstacleHeatmapBuilder(ObstacleHeatmapConfig())
    image = np.zeros((48, 64, 3), dtype=np.uint8)
    image[:, 30:34] = 255
    depth_map = _make_test_depth()

    result = builder.build(
        DepthResult(
            depth_map=depth_map,
            confidence_map=None,
            metadata={"scale_type": "relative", "status": "ok"},
        ),
        image=image,
    )

    assert result.heatmap is not None
    assert result.heatmap.shape == depth_map.shape
    assert result.heatmap.dtype == np.float32
    assert np.min(result.heatmap) >= 0.0
    assert np.max(result.heatmap) <= 1.0
    assert result.heatmap_visualization is not None
    assert result.heatmap_visualization.shape == (*depth_map.shape, 3)
    assert result.overlay_image is not None
    assert result.metadata["status"] == "ok"
    assert result.metadata["heatmap_nonzero_ratio"] is not None
    assert result.metadata["suppression_mode"] == "row_baseline"
    obstacle_activation = float(result.heatmap[18:24, 24:40].mean())
    road_activation = float(result.heatmap[-8:, :].mean())
    assert obstacle_activation >= road_activation


def test_road_suppression_handles_constant_depth_without_crashing() -> None:
    builder = DepthToObstacleHeatmapBuilder(ObstacleHeatmapConfig())
    image = np.full((40, 60, 3), 127, dtype=np.uint8)
    depth_map = np.ones((40, 60), dtype=np.float32)

    result = builder.build(
        DepthResult(
            depth_map=depth_map,
            confidence_map=None,
            metadata={"scale_type": "relative", "status": "ok"},
        ),
        image=image,
    )

    assert result.heatmap is not None
    assert result.metadata["status"] == "ok"
    assert np.isfinite(result.heatmap).all()
    assert result.metadata["heatmap_mean"] is not None
