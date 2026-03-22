from __future__ import annotations

import cv2
import numpy as np

from scene_analysis.depth.base import DepthEstimator
from scene_analysis.dynamic.base import DynamicObjectDetector
from scene_analysis.obstacle_map.base import ObstacleMapBuilder
from scene_analysis.pipeline.base import SceneAnalysisPipeline
from scene_analysis.preprocessing.frame_preprocessor import FramePreprocessor
from scene_analysis.types import DepthResult, DynamicObject, FrameData, ObstacleMapResult, SceneAnalysisResult
from scene_analysis.utils import ensure_uint8_image, shorten_model_name, timestamp_to_str


class MVPSceneAnalysisPipeline(SceneAnalysisPipeline):
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
            "depth_model": depth_result.metadata.get("model"),
            "depth_model_short": shorten_model_name(depth_result.metadata.get("model")),
            "depth_scale_type": depth_result.metadata.get("scale_type", "unknown"),
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
            f"Model: {shorten_model_name(depth.metadata.get('model'))}",
            f"Scale: {depth.metadata.get('scale_type', 'unknown')}",
            self._format_inference_line(depth),
            self._format_depth_range_line(depth),
            f"Dynamic objects: {len(dynamic_objects)}",
        ]
        if depth.depth_map is None:
            lines.insert(5, f"Depth: unavailable ({depth.metadata.get('status', 'unknown')})")
        lines.append(f"Obstacle map: {obstacle_map.metadata.get('status', 'unknown')}")

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

    @staticmethod
    def _format_inference_line(depth: DepthResult) -> str:
        inference_ms = depth.metadata.get("inference_ms")
        if inference_ms is None:
            return "Inference: n/a"
        return f"Inference: {float(inference_ms):.2f} ms"

    @staticmethod
    def _format_depth_range_line(depth: DepthResult) -> str:
        depth_min = depth.metadata.get("depth_min")
        depth_max = depth.metadata.get("depth_max")
        if depth.depth_map is None or depth_min is None or depth_max is None:
            return "Depth range: unavailable"
        return f"Depth range: {float(depth_min):.3f} .. {float(depth_max):.3f}"
