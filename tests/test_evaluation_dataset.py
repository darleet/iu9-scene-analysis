from __future__ import annotations

from pathlib import Path

import numpy as np

from scene_analysis.config import EvaluationDatasetConfig
from scene_analysis.evaluation.dataset import CityscapesLikeDataset


def _write_png(path: Path, array: np.ndarray) -> None:
    import cv2

    if not cv2.imwrite(str(path), array):
        raise IOError(f"Failed to save test image: {path}")


def test_discover_samples_matches_by_cityscapes_suffix_and_skips_missing_masks(tmp_path: Path) -> None:
    root_dir = tmp_path / "lost_and_found"
    images_dir = root_dir / "train" / "scene"
    masks_dir = root_dir / "train" / "scene"
    predictions_dir = root_dir / "predictions"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True)

    np.save(predictions_dir / "sample_a.npy", np.ones((4, 4), dtype=np.float32))
    np.save(predictions_dir / "sample_b.npy", np.ones((4, 4), dtype=np.float32))
    _write_png(masks_dir / "sample_a_gtCoarse_labelIds.png", np.zeros((4, 4), dtype=np.uint8))
    _write_png(images_dir / "sample_a_leftImg8bit.png", np.zeros((4, 4, 3), dtype=np.uint8))

    dataset = CityscapesLikeDataset(
        EvaluationDatasetConfig(
            name="lost_and_found",
            root_dir=root_dir,
            images_dir=".",
            masks_dir=".",
            predictions_dir="predictions",
            split_file=None,
            file_extension_images="_leftImg8bit.png",
            file_extension_masks="_gtCoarse_labelIds.png",
            file_extension_predictions=".npy",
        )
    )

    samples = dataset.discover_samples()

    assert len(samples) == 1
    assert samples[0].sample_id == "sample_a"
    assert samples[0].image_path is not None
    assert samples[0].mask_path.name == "sample_a_gtCoarse_labelIds.png"
    assert samples[0].prediction_path.name == "sample_a.npy"
