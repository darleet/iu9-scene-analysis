from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from scene_analysis.student.dataset import StudentHeatmapDataset
from scene_analysis.student.split import create_or_load_split, discover_raw_samples


def test_student_dataset_returns_expected_tensors(train_config) -> None:
    dataset = StudentHeatmapDataset(
        train_config.dataset.prepared_root_dir,
        "train",
        train_config.dataset,
        train_config.input,
        train_config.augmentations,
        training=True,
    )

    item = dataset[0]

    assert item["image"].shape == (3, 64, 96)
    assert item["obstacle_target"].shape == (1, 64, 96)
    assert item["roi_target"].shape == (1, 64, 96)
    assert item["valid_mask"].shape == (1, 64, 96)
    assert item["ignore_mask"].shape == (1, 64, 96)
    assert item["teacher_heatmap"].shape == (1, 64, 96)
    assert float(item["teacher_heatmap"].min()) >= 0.0
    assert float(item["teacher_heatmap"].max()) <= 1.0
    assert item["obstacle_target"].sum() > 0
    assert item["ignore_mask"].sum() > 0


def test_discover_raw_samples_and_split_are_deterministic(tmp_path: Path, make_train_config) -> None:
    raw_root = tmp_path / "raw"
    images_dir = raw_root / "images"
    masks_dir = raw_root / "masks"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)

    for index in range(10):
        sample_id = f"sample_{index:03d}"
        cv2.imwrite(str(images_dir / f"{sample_id}.png"), np.zeros((8, 8, 3), dtype=np.uint8))
        if index != 9:
            cv2.imwrite(
                str(masks_dir / f"{sample_id}_labels_semantic.png"),
                np.zeros((8, 8), dtype=np.uint8),
            )

    config = make_train_config(tmp_path / "prepared")
    config.dataset.raw_root_dir = raw_root
    samples = discover_raw_samples(config.dataset)
    train_ids, val_ids = create_or_load_split(
        samples,
        config.dataset.prepared_root_dir,
        config.dataset.train_ratio,
        config.dataset.split_seed,
    )
    train_ids_again, val_ids_again = create_or_load_split(
        samples,
        config.dataset.prepared_root_dir,
        config.dataset.train_ratio,
        config.dataset.split_seed,
    )

    assert len(samples) == 9
    assert len(train_ids) == 7
    assert len(val_ids) == 2
    assert train_ids == train_ids_again
    assert val_ids == val_ids_again
