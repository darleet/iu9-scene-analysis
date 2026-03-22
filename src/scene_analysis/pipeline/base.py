from __future__ import annotations

from abc import ABC, abstractmethod

from scene_analysis.types import FrameData, SceneAnalysisResult


class SceneAnalysisPipeline(ABC):
    """Абстрактный интерфейс покадровых пайплайнов"""

    @abstractmethod
    def process_frame(self, frame: FrameData) -> SceneAnalysisResult:
        """Обработать один кадр и вернуть структурированный результат"""
