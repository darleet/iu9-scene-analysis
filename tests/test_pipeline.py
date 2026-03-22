from __future__ import annotations

import numpy as np
import pytest

from scene_analysis.config import PreprocessingConfig, RoiConfig
from scene_analysis.depth.base import DepthEstimator, DummyDepthEstimator
from scene_analysis.dynamic.base import DummyDynamicObjectDetector
from scene_analysis.obstacle_map.base import DummyObstacleMapBuilder
from scene_analysis.pipeline.mvp_pipeline import MVPSceneAnalysisPipeline
from scene_analysis.preprocessing.frame_preprocessor import FramePreprocessor
from scene_analysis.types import DepthResult, FrameData, SceneAnalysisResult


class FakeDepthEstimator(DepthEstimator):
    def predict(self, image: np.ndarray) -> DepthResult:
        height, width = image.shape[:2]
        depth_map = np.linspace(0.1, 1.0, num=height * width, dtype=np.float32).reshape(height, width)
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
    ("depth_estimator", "expected_status"),
    [
        (DummyDepthEstimator(), "dummy"),
        (FakeDepthEstimator(), "ok"),
    ],
)
def test_pipeline_returns_structured_result(
    depth_estimator: DepthEstimator,
    expected_status: str,
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
        obstacle_map_builder=DummyObstacleMapBuilder(),
        dynamic_detector=DummyDynamicObjectDetector(),
    )

    result = pipeline.process_frame(frame)

    assert isinstance(result, SceneAnalysisResult)
    assert result.overlay_image is not None
    assert result.overlay_image.shape[:2] == (12, 16)
    assert isinstance(result.dynamic_objects, list)
    assert result.depth.metadata["status"] == expected_status
    assert result.obstacle_map.metadata["status"] == "dummy"
    assert result.metadata["depth_status"] == expected_status
    assert result.metadata["obstacle_status"] == "dummy"
    assert result.metadata["depth_model_short"] in {"dummy", "Depth-Anything-V2-Small-hf"}

    if expected_status == "ok":
        assert result.depth.depth_map is not None
        assert result.depth.depth_map.shape == (12, 16)
        assert result.metadata["depth_scale_type"] == "relative"
    else:
        assert result.depth.depth_map is None
