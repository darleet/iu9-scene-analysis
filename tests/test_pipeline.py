from __future__ import annotations

import numpy as np

from scene_analysis.config import PreprocessingConfig, RoiConfig
from scene_analysis.depth.base import DummyDepthEstimator
from scene_analysis.dynamic.base import DummyDynamicObjectDetector
from scene_analysis.obstacle_map.base import DummyObstacleMapBuilder
from scene_analysis.pipeline.mvp_pipeline import MVPSceneAnalysisPipeline
from scene_analysis.preprocessing.frame_preprocessor import FramePreprocessor
from scene_analysis.types import FrameData, SceneAnalysisResult


def test_dummy_pipeline_returns_structured_result() -> None:
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
        depth_estimator=DummyDepthEstimator(),
        obstacle_map_builder=DummyObstacleMapBuilder(),
        dynamic_detector=DummyDynamicObjectDetector(),
    )

    result = pipeline.process_frame(frame)

    assert isinstance(result, SceneAnalysisResult)
    assert result.overlay_image is not None
    assert result.overlay_image.shape[:2] == (12, 16)
    assert isinstance(result.dynamic_objects, list)
    assert result.depth.metadata["status"] == "dummy"
    assert result.obstacle_map.metadata["status"] == "dummy"
    assert result.metadata["depth_status"] == "dummy"
    assert result.metadata["obstacle_status"] == "dummy"
