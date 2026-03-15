from __future__ import annotations

from typing import Protocol

import numpy as np

from scene_analysis.types import DepthResult


class DepthEstimator(Protocol):
    """Протокол для подключаемых моделей оценки глубины"""

    def predict(self, image: np.ndarray) -> DepthResult:
        """Оценить глубину на изображении"""


class DummyDepthEstimator:
    """Оценщик глубины (заглушка)"""

    def predict(self, image: np.ndarray) -> DepthResult:
        """Вернуть результат оценки глубины"""
        return DepthResult(
            depth_map=None,
            confidence_map=None,
            metadata={
                "status": "dummy",
                "message": "Depth model is not connected yet",
            },
        )
