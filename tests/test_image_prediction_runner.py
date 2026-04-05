from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from scene_analysis.config import ObstacleHeatmapConfig, PreprocessingConfig, RoiConfig
from scene_analysis.depth.base import DepthEstimator
from scene_analysis.io.artifact_writer import ArtifactWriter
from scene_analysis.io.heatmap_prediction_writer import HeatmapPredictionWriter
from scene_analysis.io.image_reader import ImageDirectoryReader
from scene_analysis.obstacle_map.heatmap_builder import DepthToObstacleHeatmapBuilder
from scene_analysis.pipeline.image_prediction_runner import ImagePredictionRunner
from scene_analysis.pipeline.mvp_pipeline import MVPSceneAnalysisPipeline
from scene_analysis.preprocessing.frame_preprocessor import FramePreprocessor
from scene_analysis.types import DepthResult


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
                "inference_ms": 1.0,
                "depth_min": float(depth_map.min()),
                "depth_max": float(depth_map.max()),
                "depth_mean": float(depth_map.mean()),
                "scale_type": "relative",
            },
        )


def _write_image(path: Path, color: int) -> None:
    image = np.full((24, 32, 3), color, dtype=np.uint8)
    if not cv2.imwrite(str(path), image):
        raise IOError(f"Failed to write test image: {path}")


def test_image_prediction_runner_generates_predictions_by_stem(tmp_path: Path) -> None:
    images_dir = tmp_path / "images"
    predictions_dir = tmp_path / "predictions"
    artifacts_dir = tmp_path / "artifacts"
    images_dir.mkdir()

    _write_image(images_dir / "sample_b.png", 64)
    _write_image(images_dir / "sample_a.png", 128)

    pipeline = MVPSceneAnalysisPipeline(
        preprocessor=FramePreprocessor(
            PreprocessingConfig(
                resize_width=16,
                resize_height=12,
                normalize_to_float=False,
                roi=RoiConfig(),
            )
        ),
        depth_estimator=FakeDepthEstimator(),
        obstacle_heatmap_builder=DepthToObstacleHeatmapBuilder(ObstacleHeatmapConfig()),
    )
    runner = ImagePredictionRunner(
        pipeline=pipeline,
        image_reader=ImageDirectoryReader(images_dir, extension=".png"),
        prediction_writer=HeatmapPredictionWriter(predictions_dir),
        artifact_writer=ArtifactWriter(
            output_config=type(
                "OutputConfigStub",
                (),
                {
                    "output_dir": artifacts_dir,
                    "save_original_frames": True,
                    "save_preprocessed_frames": True,
                    "save_overlay_frames": True,
                    "save_jsonl": True,
                },
            )(),
            depth_config=type(
                "DepthConfigStub",
                (),
                {
                    "save_raw_depth_npy": True,
                    "save_depth_colormap": True,
                    "normalize_depth_for_viz": True,
                    "clip_percentiles": type("ClipStub", (), {"min": 2.0, "max": 98.0})(),
                },
            )(),
            obstacle_heatmap_config=type(
                "ObstacleConfigStub",
                (),
                {
                    "visualization": type(
                        "VisualizationStub",
                        (),
                        {
                            "save_heatmap_npy": True,
                            "save_heatmap_png": True,
                            "save_overlay_png": True,
                        },
                    )(),
                },
            )(),
        ),
    )

    processed_samples = runner.run()

    assert processed_samples == 2
    sample_a_path = predictions_dir / "sample_a.npy"
    sample_b_path = predictions_dir / "sample_b.npy"
    assert sample_a_path.exists()
    assert sample_b_path.exists()

    sample_a = np.load(sample_a_path)
    sample_b = np.load(sample_b_path)
    assert sample_a.shape == (12, 16)
    assert sample_b.shape == (12, 16)
    assert sample_a.dtype == np.float32
    assert sample_b.dtype == np.float32
    assert (artifacts_dir / "original_frames" / "sample_a.png").exists()
    assert (artifacts_dir / "preprocessed_frames" / "sample_a.png").exists()
    assert (artifacts_dir / "depth_npy" / "sample_a.npy").exists()
    assert (artifacts_dir / "depth_colormap" / "sample_a.png").exists()
    assert (artifacts_dir / "obstacle_heatmap_npy" / "sample_a.npy").exists()
    assert (artifacts_dir / "obstacle_heatmap_png" / "sample_a.png").exists()
    assert (artifacts_dir / "overlay_frames" / "sample_a.png").exists()
    assert (artifacts_dir / "results.jsonl").exists()
