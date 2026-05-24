from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from scene_analysis.student.dataset import StudentHeatmapDataset
from scene_analysis.student.split import RawStudentSample, create_or_load_split, discover_raw_samples
from scene_analysis.student.teacher_prepare import _copy_sample


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


def test_discover_raw_samples_supports_cityscapes_style_suffixes(tmp_path: Path, make_train_config) -> None:
    raw_root = tmp_path / "lost_found"
    scene_dir = raw_root / "train" / "01_scene"
    scene_dir.mkdir(parents=True)
    sample_id = "01_scene_000000_000010"

    cv2.imwrite(str(scene_dir / f"{sample_id}_leftImg8bit.png"), np.zeros((8, 8, 3), dtype=np.uint8))
    cv2.imwrite(str(scene_dir / f"{sample_id}_gtCoarse_labelIds.png"), np.zeros((8, 8), dtype=np.uint8))

    config = make_train_config(tmp_path / "prepared")
    config.dataset.raw_root_dir = raw_root
    config.dataset.images_dir = Path(".")
    config.dataset.masks_dir = Path(".")
    config.dataset.image_suffix = "_leftImg8bit.png"
    config.dataset.mask_suffix = "_gtCoarse_labelIds.png"

    samples = discover_raw_samples(config.dataset)

    assert len(samples) == 1
    assert samples[0].sample_id == sample_id
    assert samples[0].raw_split == "train"


def test_create_split_uses_cityscapes_train_and_test_dirs(tmp_path: Path, make_train_config) -> None:
    raw_root = tmp_path / "lost_found"
    for split_name in ("train", "test"):
        scene_dir = raw_root / split_name / "scene"
        scene_dir.mkdir(parents=True)
        for index in range(2):
            sample_id = f"{split_name}_{index:03d}"
            cv2.imwrite(str(scene_dir / f"{sample_id}_leftImg8bit.png"), np.zeros((8, 8, 3), dtype=np.uint8))
            cv2.imwrite(str(scene_dir / f"{sample_id}_gtCoarse_labelIds.png"), np.zeros((8, 8), dtype=np.uint8))

    config = make_train_config(tmp_path / "prepared")
    config.dataset.raw_root_dir = raw_root
    config.dataset.images_dir = Path(".")
    config.dataset.masks_dir = Path(".")
    config.dataset.image_suffix = "_leftImg8bit.png"
    config.dataset.mask_suffix = "_gtCoarse_labelIds.png"

    samples = discover_raw_samples(config.dataset)
    train_ids, val_ids = create_or_load_split(
        samples,
        config.dataset.prepared_root_dir,
        config.dataset.train_ratio,
        config.dataset.split_seed,
        train_split_names=config.dataset.raw_train_splits,
        val_split_names=config.dataset.raw_val_splits,
    )

    assert train_ids == ["train_000", "train_001"]
    assert val_ids == ["test_000", "test_001"]


def test_copy_sample_can_remap_source_mask_values(tmp_path: Path, make_train_config) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    image_path = raw_root / "sample_leftImg8bit.png"
    mask_path = raw_root / "sample_gtCoarse_labelIds.png"
    cv2.imwrite(str(image_path), np.zeros((2, 3, 3), dtype=np.uint8))
    cv2.imwrite(str(mask_path), np.array([[0, 1, 6], [1, 6, 0]], dtype=np.uint8))

    config = make_train_config(tmp_path / "prepared")
    config.dataset.image_suffix = "_leftImg8bit.png"
    config.dataset.mask_suffix = "_gtCoarse_labelIds.png"
    config.dataset.mask_background_values = [1]
    config.dataset.mask_ignore_values = [0]
    config.dataset.mask_unmapped_value = config.dataset.obstacle_value

    split_dirs = {
        "images": tmp_path / "out" / "images",
        "masks": tmp_path / "out" / "masks",
        "teacher_heatmaps": tmp_path / "out" / "teacher_heatmaps",
    }
    for path in split_dirs.values():
        path.mkdir(parents=True)

    _copy_sample(
        RawStudentSample(sample_id="sample", image_path=image_path, mask_path=mask_path),
        split_dirs,
        config.dataset,
    )

    remapped = cv2.imread(str(split_dirs["masks"] / "sample_gtCoarse_labelIds.png"), cv2.IMREAD_UNCHANGED)

    assert remapped.tolist() == [[255, 0, 1], [0, 1, 255]]
