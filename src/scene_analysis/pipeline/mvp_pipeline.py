from __future__ import annotations

import cv2
import numpy as np

from scene_analysis.depth.base import DepthEstimator
from scene_analysis.obstacle_map.base import ObstacleHeatmapBuilder
from scene_analysis.pipeline.base import SceneAnalysisPipeline
from scene_analysis.preprocessing.frame_preprocessor import FramePreprocessor
from scene_analysis.types import DepthResult, FrameData, ObstacleHeatmapResult, SceneAnalysisResult
from scene_analysis.utils import ensure_uint8_image, shorten_model_name, timestamp_to_str


class MVPSceneAnalysisPipeline(SceneAnalysisPipeline):
    def __init__(
        self,
        preprocessor: FramePreprocessor,
        depth_estimator: DepthEstimator,
        obstacle_heatmap_builder: ObstacleHeatmapBuilder,
    ) -> None:
        self.preprocessor = preprocessor
        self.depth_estimator = depth_estimator
        self.obstacle_heatmap_builder = obstacle_heatmap_builder

    def process_frame(self, frame: FrameData) -> SceneAnalysisResult:
        preprocessed_image = self.preprocessor.process(frame.image)
        depth_result = self.depth_estimator.predict(preprocessed_image)
        obstacle_result = self.obstacle_heatmap_builder.build(depth_result, preprocessed_image)
        overlay_image = self._build_overlay(
            frame=frame,
            preprocessed_image=preprocessed_image,
            depth=depth_result,
            obstacle_heatmap=obstacle_result,
        )
        obstacle_result.overlay_image = overlay_image

        metadata = {
            "depth_status": depth_result.metadata.get("status", "unknown"),
            "obstacle_heatmap_status": obstacle_result.metadata.get("status", "unknown"),
            "preprocessed_shape": list(preprocessed_image.shape),
            "depth_model": depth_result.metadata.get("model"),
            "depth_model_short": shorten_model_name(depth_result.metadata.get("model")),
            "depth_scale_type": depth_result.metadata.get("scale_type", "unknown"),
            "heatmap_mean": obstacle_result.metadata.get("heatmap_mean"),
            "heatmap_max": obstacle_result.metadata.get("heatmap_max"),
        }

        return SceneAnalysisResult(
            frame=frame,
            preprocessed_image=preprocessed_image,
            depth=depth_result,
            obstacle_heatmap=obstacle_result,
            overlay_image=overlay_image,
            metadata=metadata,
        )

    def _build_overlay(
        self,
        frame: FrameData,
        preprocessed_image: np.ndarray,
        depth: DepthResult,
        obstacle_heatmap: ObstacleHeatmapResult,
    ) -> np.ndarray:
        overlay = obstacle_heatmap.overlay_image
        if overlay is None:
            overlay = self._prepare_base_overlay(preprocessed_image)
        else:
            overlay = self._prepare_base_overlay(overlay)

        lines = [
            f"Frame: {frame.frame_index}",
            f"Timestamp: {timestamp_to_str(frame.timestamp_ms)}",
            f"Model: {shorten_model_name(depth.metadata.get('model'))}",
            f"Scale: {depth.metadata.get('scale_type', 'unknown')}",
            self._format_inference_line(depth),
            f"Heatmap: {obstacle_heatmap.metadata.get('status', 'unknown')}",
            self._format_heatmap_metric_line("Heatmap mean", obstacle_heatmap.metadata.get("heatmap_mean")),
            self._format_heatmap_metric_line("Heatmap max", obstacle_heatmap.metadata.get("heatmap_max")),
        ]
        if depth.depth_map is None:
            lines.insert(5, f"Depth: unavailable ({depth.metadata.get('status', 'unknown')})")

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
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
            y_position += 26

        return overlay

    @staticmethod
    def _prepare_base_overlay(image: np.ndarray) -> np.ndarray:
        overlay = ensure_uint8_image(image)
        if overlay.ndim == 2:
            return cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)
        if overlay.ndim == 3 and overlay.shape[2] == 1:
            return cv2.cvtColor(overlay[:, :, 0], cv2.COLOR_GRAY2BGR)
        if overlay.ndim == 3 and overlay.shape[2] == 3:
            return overlay.copy()
        raise ValueError("Overlay image must have shape HxW or HxWx3")

    @staticmethod
    def _format_inference_line(depth: DepthResult) -> str:
        inference_ms = depth.metadata.get("inference_ms")
        if inference_ms is None:
            return "Inference: n/a"
        return f"Inference: {float(inference_ms):.2f} ms"

    @staticmethod
    def _format_heatmap_metric_line(label: str, value: float | None) -> str:
        if value is None:
            return f"{label}: n/a"
        return f"{label}: {float(value):.3f}"
