from __future__ import annotations

import numpy as np
import pytest

from scene_analysis.config import ObstacleHeatmapConfig, PreprocessingConfig, RoiConfig
from scene_analysis.depth.base import DepthEstimator, DummyDepthEstimator
from scene_analysis.obstacle_map.heatmap_builder import DepthToObstacleHeatmapBuilder
from scene_analysis.pipeline.mvp_pipeline import MVPSceneAnalysisPipeline
from scene_analysis.preprocessing.frame_preprocessor import FramePreprocessor
from scene_analysis.types import DepthResult, FrameData, SceneAnalysisResult


class FakeDepthEstimator(DepthEstimator):
    def predict(self, image: np.ndarray) -> DepthResult:
        height, width = image.shape[:2]
        depth_map = np.repeat(
            np.linspace(0.1, 1.0, num=height, dtype=np.float32)[:, None],
            width,
            axis=1,
        )
        depth_map[height // 3 : height // 2, width // 3 : width // 2] = 1.2
        return DepthResult(
            depth_map=depth_map,
            confidence_map=None,
            metadata={
                "status": "ok",
                "provider": "depth_anything_v2",
                "model": "depth-anything/Depth-Anything-V2-Small-hf",
                "device": "cpu",
                "inference_ms": 12.5,
                "original_height": height,
                "original_width": width,
                "output_height": height,
                "output_width": width,
                "depth_min": float(depth_map.min()),
                "depth_max": float(depth_map.max()),
                "depth_mean": float(depth_map.mean()),
                "scale_type": "relative",
            },
        )


@pytest.mark.parametrize(
    ("depth_estimator", "expected_depth_status", "expected_heatmap_status"),
    [
        (DummyDepthEstimator(), "dummy", "no_depth"),
        (FakeDepthEstimator(), "ok", "ok"),
    ],
)
def test_pipeline_returns_structured_result(
    depth_estimator: DepthEstimator,
    expected_depth_status: str,
    expected_heatmap_status: str,
) -> None:
    frame = FrameData(
        frame_index=0,
        timestamp_ms=0.0,
        image=np.zeros((24, 32, 3), dtype=np.uint8),
        source_path="data/raw/sample.mp4",
        width=32,
        height=24,
    )
    preprocessor = FramePreprocessor(
        PreprocessingConfig(
            resize_width=16,
            resize_height=12,
            normalize_to_float=False,
            roi=RoiConfig(),
        )
    )
    pipeline = MVPSceneAnalysisPipeline(
        preprocessor=preprocessor,
        depth_estimator=depth_estimator,
        obstacle_heatmap_builder=DepthToObstacleHeatmapBuilder(ObstacleHeatmapConfig()),
    )

    result = pipeline.process_frame(frame)

    assert isinstance(result, SceneAnalysisResult)
    assert result.overlay_image is not None
    assert result.overlay_image.shape[:2] == (12, 16)
    assert result.depth.metadata["status"] == expected_depth_status
    assert result.obstacle_heatmap.metadata["status"] == expected_heatmap_status
    assert result.metadata["depth_status"] == expected_depth_status
    assert result.metadata["obstacle_heatmap_status"] == expected_heatmap_status
    assert result.metadata["depth_model_short"] in {"dummy", "Depth-Anything-V2-Small-hf"}

    if expected_depth_status == "ok":
        assert result.depth.depth_map is not None
        assert result.depth.depth_map.shape == (12, 16)
        assert result.obstacle_heatmap.heatmap is not None
        assert result.obstacle_heatmap.heatmap.shape == (12, 16)
        assert result.metadata["depth_scale_type"] == "relative"
    else:
        assert result.depth.depth_map is None
        assert result.obstacle_heatmap.heatmap is None
