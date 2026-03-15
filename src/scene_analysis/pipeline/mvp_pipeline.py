"""Минимальная реализация пайплайна"""

from __future__ import annotations

import cv2
import numpy as np

from scene_analysis.depth.base import DepthEstimator
from scene_analysis.dynamic.base import DynamicObjectDetector
from scene_analysis.obstacle_map.base import ObstacleMapBuilder
from scene_analysis.pipeline.base import SceneAnalysisPipeline
from scene_analysis.preprocessing.frame_preprocessor import FramePreprocessor
from scene_analysis.types import DepthResult, DynamicObject, FrameData, ObstacleMapResult, SceneAnalysisResult
from scene_analysis.utils import ensure_uint8_image, timestamp_to_str


class MVPSceneAnalysisPipeline(SceneAnalysisPipeline):
    """MVP-пайплайн с блоками обработки"""

    def __init__(
        self,
        preprocessor: FramePreprocessor,
        depth_estimator: DepthEstimator,
        obstacle_map_builder: ObstacleMapBuilder,
        dynamic_detector: DynamicObjectDetector,
    ) -> None:
        self.preprocessor = preprocessor
        self.depth_estimator = depth_estimator
        self.obstacle_map_builder = obstacle_map_builder
        self.dynamic_detector = dynamic_detector

    def process_frame(self, frame: FrameData) -> SceneAnalysisResult:
        """Запустить полный пайплайн для одного кадра"""
        preprocessed_image = self.preprocessor.process(frame.image)
        depth_result = self.depth_estimator.predict(preprocessed_image)
        obstacle_result = self.obstacle_map_builder.build(depth_result, preprocessed_image)
        dynamic_objects = self.dynamic_detector.detect(preprocessed_image)
        overlay_image = self._build_overlay(
            frame=frame,
            preprocessed_image=preprocessed_image,
            depth=depth_result,
            obstacle_map=obstacle_result,
            dynamic_objects=dynamic_objects,
        )

        metadata = {
            "depth_status": depth_result.metadata.get("status", "unknown"),
            "obstacle_status": obstacle_result.metadata.get("status", "unknown"),
            "dynamic_count": len(dynamic_objects),
            "preprocessed_shape": list(preprocessed_image.shape),
        }

        return SceneAnalysisResult(
            frame=frame,
            preprocessed_image=preprocessed_image,
            depth=depth_result,
            obstacle_map=obstacle_result,
            dynamic_objects=dynamic_objects,
            overlay_image=overlay_image,
            metadata=metadata,
        )

    def _build_overlay(
        self,
        frame: FrameData,
        preprocessed_image: np.ndarray,
        depth: DepthResult,
        obstacle_map: ObstacleMapResult,
        dynamic_objects: list[DynamicObject],
    ) -> np.ndarray:
        overlay = ensure_uint8_image(preprocessed_image)
        if overlay.ndim == 2:
            overlay = cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
        elif overlay.ndim == 3 and overlay.shape[2] == 1:
            overlay = cv2.cvtColor(overlay[:, :, 0], cv2.COLOR_GRAY2BGR)
        else:
            overlay = overlay.copy()

        lines = [
            f"Frame: {frame.frame_index}",
            f"Timestamp: {timestamp_to_str(frame.timestamp_ms)}",
            f"Depth: {depth.metadata.get('status', 'unknown')}",
            f"Obstacle map: {obstacle_map.metadata.get('status', 'unknown')}",
            f"Dynamic objects: {len(dynamic_objects)}",
        ]

        y_position = 28
        for line in lines:
            cv2.putText(
                overlay,
                line,
                (12, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                3,
                cv2.LINE_AA,
            )
            cv2.putText(
                overlay,
                line,
                (12, y_position),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
            y_position += 26

        return overlay
