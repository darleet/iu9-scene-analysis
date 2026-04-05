from __future__ import annotations

from pathlib import Path

import numpy as np

from scene_analysis.config import EvaluationDatasetConfig
from scene_analysis.evaluation.dataset import RoadObstacle21Dataset


def _write_png(path: Path, array: np.ndarray) -> None:
    import cv2

    if not cv2.imwrite(str(path), array):
        raise IOError(f"Failed to save test image: {path}")


def test_discover_samples_matches_by_stem_and_skips_missing_masks(tmp_path: Path) -> None:
    root_dir = tmp_path / "road_obstacle_21"
    images_dir = root_dir / "images"
    masks_dir = root_dir / "masks"
    predictions_dir = root_dir / "predictions"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)
    predictions_dir.mkdir(parents=True)

    np.save(predictions_dir / "sample_a.npy", np.ones((4, 4), dtype=np.float32))
    np.save(predictions_dir / "sample_b.npy", np.ones((4, 4), dtype=np.float32))
    _write_png(masks_dir / "sample_a.png", np.zeros((4, 4), dtype=np.uint8))
    _write_png(images_dir / "sample_a.png", np.zeros((4, 4, 3), dtype=np.uint8))

    dataset = RoadObstacle21Dataset(
        EvaluationDatasetConfig(
            name="road_obstacle_21",
            root_dir=root_dir,
            images_dir="images",
            masks_dir="masks",
            predictions_dir="predictions",
            split_file=None,
            file_extension_images=".png",
            file_extension_masks=".png",
            file_extension_predictions=".npy",
        )
    )

    samples = dataset.discover_samples()

    assert len(samples) == 1
    assert samples[0].sample_id == "sample_a"
    assert samples[0].image_path is not None
    assert samples[0].mask_path.name == "sample_a.png"
    assert samples[0].prediction_path.name == "sample_a.npy"
