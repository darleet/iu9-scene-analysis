from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class FrameData:
    """Кадр, полученный из видеоисточника"""

    frame_index: int
    timestamp_ms: float
    image: np.ndarray
    source_path: str | None
    width: int
    height: int


@dataclass(slots=True)
class DepthResult:
    """Результат оценки глубины для кадра"""

    depth_map: np.ndarray | None
    confidence_map: np.ndarray | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ObstacleMapResult:
    """Результат построения карты препятствий для кадра"""

    obstacle_mask: np.ndarray | None
    occupancy_grid: np.ndarray | None
    costmap: np.ndarray | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class DynamicObject:
    """Гипотеза о динамическом объекте"""

    track_id: int | None
    label: str
    confidence: float
    bbox: tuple[int, int, int, int] | None


@dataclass(slots=True)
class SceneAnalysisResult:
    """Результат пайплайна для одного кадра"""

    frame: FrameData
    preprocessed_image: np.ndarray
    depth: DepthResult
    obstacle_map: ObstacleMapResult
    dynamic_objects: list[DynamicObject]
    overlay_image: np.ndarray | None
    metadata: dict[str, Any] = field(default_factory=dict)
