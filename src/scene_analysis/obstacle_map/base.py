from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from scene_analysis.types import DepthResult, ObstacleMapResult


class ObstacleMapBuilder(ABC):
    """Абстрактный интерфейс генератора карты препятствий"""

    @abstractmethod
    def build(self, depth: DepthResult, image: np.ndarray) -> ObstacleMapResult:
        """Построить карту препятствий для текущего кадра"""


class DummyObstacleMapBuilder(ObstacleMapBuilder):
    """Генератор карты препятствий (заглушка)"""

    def build(self, depth: DepthResult, image: np.ndarray) -> ObstacleMapResult:
        return ObstacleMapResult(
            obstacle_mask=None,
            occupancy_grid=None,
            costmap=None,
            metadata={
                "status": "dummy",
                "message": "Obstacle map builder is not connected yet",
            },
        )
