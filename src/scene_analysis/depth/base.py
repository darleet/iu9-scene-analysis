from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from scene_analysis.types import DepthResult

if TYPE_CHECKING:
    from scene_analysis.config import DepthConfig


class DepthEstimator(ABC):
    """Абстрактный интерфейс для подключаемых моделей оценки глубины"""

    @abstractmethod
    def predict(self, image: np.ndarray) -> DepthResult:
        """Оценить глубину на изображении"""


class DummyDepthEstimator(DepthEstimator):
    """Оценщик глубины (заглушка)"""

    def predict(self, image: np.ndarray) -> DepthResult:
        """Вернуть результат оценки глубины"""
        height, width = image.shape[:2] if isinstance(image, np.ndarray) and image.size > 0 else (None, None)
        return DepthResult(
            depth_map=None,
            confidence_map=None,
            metadata={
                "status": "dummy",
                "provider": "dummy",
                "model": "dummy",
                "device": "cpu",
                "inference_ms": 0.0,
                "original_height": height,
                "original_width": width,
                "output_height": height,
                "output_width": width,
                "depth_min": None,
                "depth_max": None,
                "depth_mean": None,
                "scale_type": "relative",
                "message": "Depth model is not connected yet",
            },
        )


def _create_depth_anything_v2_estimator(depth_config: DepthConfig) -> DepthEstimator:
    from scene_analysis.depth.depth_anything_estimator import DepthAnythingV2Estimator

    return DepthAnythingV2Estimator(
        model_name=depth_config.model,
        device=depth_config.device,
        cache_dir=depth_config.cache_dir,
        use_fp16=depth_config.use_fp16,
        compile_model=depth_config.compile_model,
    )


def create_depth_estimator(depth_config: DepthConfig) -> DepthEstimator:
    """Создать depth estimator по конфигурации"""
    if not depth_config.enabled:
        return DummyDepthEstimator()
    if depth_config.provider == "depth_anything_v2":
        return _create_depth_anything_v2_estimator(depth_config)
    raise ValueError(f"Unsupported depth provider: {depth_config.provider}")
