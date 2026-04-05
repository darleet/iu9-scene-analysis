from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from scene_analysis.config import ObstacleHeatmapConfig
from scene_analysis.types import DepthResult, ObstacleHeatmapResult


class ObstacleHeatmapBuilder(ABC):
    @abstractmethod
    def build(self, depth: DepthResult, image: np.ndarray) -> ObstacleHeatmapResult:
        """Построить obstacle heatmap для текущего кадра"""


class DummyObstacleHeatmapBuilder(ObstacleHeatmapBuilder):
    """Генератор obstacle heatmap (заглушка)"""

    def build(self, depth: DepthResult, image: np.ndarray) -> ObstacleHeatmapResult:
        return ObstacleHeatmapResult(
            heatmap=None,
            heatmap_visualization=None,
            overlay_image=None,
            metadata={
                "status": "disabled",
                "depth_available": depth.depth_map is not None,
                "scale_type": depth.metadata.get("scale_type", "unknown"),
                "message": "Obstacle heatmap builder is disabled",
            },
        )


def create_obstacle_heatmap_builder(config: ObstacleHeatmapConfig) -> ObstacleHeatmapBuilder:
    if not config.enabled:
        return DummyObstacleHeatmapBuilder()

    from scene_analysis.obstacle_map.heatmap_builder import DepthToObstacleHeatmapBuilder

    return DepthToObstacleHeatmapBuilder(config)
