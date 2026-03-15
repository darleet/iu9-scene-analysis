from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from scene_analysis.types import DynamicObject


class DynamicObjectDetector(ABC):
    """Абстрактный интерфейс детекции динамических объектов"""

    @abstractmethod
    def detect(self, image: np.ndarray) -> list[DynamicObject]:
        """Найти динамические объекты на текущем кадре"""


class DummyDynamicObjectDetector(DynamicObjectDetector):
    """Детектор динамических объектов (заглушка)"""

    def detect(self, image: np.ndarray) -> list[DynamicObject]:
        """Вернуть список динамических объектов"""
        return []
