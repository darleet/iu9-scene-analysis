from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scene_analysis.config import load_config
from scene_analysis.student.artifacts import save_json
import scene_analysis.student.teacher_prepare as teacher_prepare


def test_large_teacher_config_uses_metric_outdoor_depth() -> None:
    config = load_config(Path("configs/teacher_depth_anything_large.yaml"))

    assert config.depth.model == "depth-anything/Depth-Anything-V2-Metric-Outdoor-Large-hf"
    assert config.depth.use_fp16 is True
    assert config.obstacle_heatmap.near_score.use_relative_depth is False
    assert config.obstacle_heatmap.near_score.invert_depth is False
    assert config.obstacle_heatmap.road_suppression.enabled is False


def test_build_teacher_pipeline_uses_student_input_geometry(
    tmp_path: Path,
    make_train_config,
    monkeypatch,
) -> None:
    config = make_train_config(tmp_path / "prepared")

    monkeypatch.setattr(teacher_prepare, "create_depth_estimator", lambda _: object())
    monkeypatch.setattr(teacher_prepare, "create_obstacle_heatmap_builder", lambda _: object())

    pipeline = teacher_prepare._build_teacher_pipeline(config.teacher.config_path, config.input)
    preprocessing = pipeline.preprocessor.config

    assert preprocessing.roi.enabled is False
    assert preprocessing.resize_width == config.input.width
    assert preprocessing.resize_height == config.input.height


def test_prepare_student_data_dispatches_multiple_dataset_configs(
    tmp_path: Path,
    make_train_config,
    monkeypatch,
) -> None:
    config = make_train_config(tmp_path / "prepared")
    assert config.dataset is not None
    first_dataset = config.dataset.model_copy(
        update={"name": "first", "prepared_root_dir": tmp_path / "prepared_first"}
    )
    second_dataset = config.dataset.model_copy(
        update={"name": "second", "prepared_root_dir": tmp_path / "prepared_second"}
    )
    config.dataset = first_dataset
    config.datasets = [first_dataset, second_dataset]
    seen_names: list[str] = []

    def fake_prepare(single_config, *, overwrite_teacher_heatmaps: bool, limit: int | None):
        seen_names.append(single_config.dataset.name)
        return {
            "status": "ok",
            "dataset_name": single_config.dataset.name,
            "raw_root_dir": str(single_config.dataset.raw_root_dir),
            "prepared_root_dir": str(single_config.dataset.prepared_root_dir),
            "teacher_metadata_path": str(single_config.dataset.prepared_root_dir / "teacher_metadata.json"),
            "train_samples": 2,
            "val_samples": 1,
            "copied": {"train": 2, "val": 1},
            "teacher_heatmaps_generated": {"train": 0, "val": 0},
            "teacher_heatmaps_skipped_existing": {"train": 2, "val": 1},
            "overwrite_teacher_heatmaps": overwrite_teacher_heatmaps,
            "limit": limit,
            "preview_paths": [],
            "split_train": str(single_config.dataset.prepared_root_dir / "split" / "train.txt"),
            "split_val": str(single_config.dataset.prepared_root_dir / "split" / "val.txt"),
        }

    monkeypatch.setattr(teacher_prepare, "_prepare_single_student_data", fake_prepare)

    summary = teacher_prepare.prepare_student_data(config, overwrite_teacher_heatmaps=True, limit=5)

    assert seen_names == ["first", "second"]
    assert summary["dataset_count"] == 2
    assert summary["train_samples"] == 4
    assert summary["val_samples"] == 2


def test_teacher_heatmap_shape_squeezes_singleton_channel(tmp_path: Path) -> None:
    heatmap_path = tmp_path / "teacher.npy"
    np.save(heatmap_path, np.zeros((1, 64, 96), dtype=np.float32))

    assert teacher_prepare._teacher_heatmap_shape(heatmap_path) == (64, 96)


def test_teacher_metadata_requires_explicit_overwrite_for_missing_metadata(
    tmp_path: Path,
    make_train_config,
) -> None:
    config = make_train_config(tmp_path / "prepared")
    heatmap_dir = config.dataset.prepared_root_dir / "train" / "teacher_heatmaps"
    heatmap_dir.mkdir(parents=True)
    np.save(heatmap_dir / "sample.npy", np.zeros((64, 96), dtype=np.float32))
    expected_metadata = teacher_prepare._build_teacher_metadata(config)

    with pytest.raises(ValueError, match="--overwrite-teacher-heatmaps"):
        teacher_prepare._validate_existing_teacher_metadata(
            config.dataset.prepared_root_dir,
            expected_metadata,
        )


def test_teacher_metadata_accepts_matching_fingerprint(
    tmp_path: Path,
    make_train_config,
) -> None:
    config = make_train_config(tmp_path / "prepared")
    heatmap_dir = config.dataset.prepared_root_dir / "train" / "teacher_heatmaps"
    heatmap_dir.mkdir(parents=True)
    np.save(heatmap_dir / "sample.npy", np.zeros((64, 96), dtype=np.float32))
    expected_metadata = teacher_prepare._build_teacher_metadata(config)
    save_json(teacher_prepare._teacher_metadata_path(config.dataset.prepared_root_dir), expected_metadata)

    teacher_prepare._validate_existing_teacher_metadata(
        config.dataset.prepared_root_dir,
        expected_metadata,
    )


def test_teacher_metadata_rejects_changed_fingerprint(
    tmp_path: Path,
    make_train_config,
) -> None:
    config = make_train_config(tmp_path / "prepared")
    heatmap_dir = config.dataset.prepared_root_dir / "train" / "teacher_heatmaps"
    heatmap_dir.mkdir(parents=True)
    np.save(heatmap_dir / "sample.npy", np.zeros((64, 96), dtype=np.float32))
    expected_metadata = teacher_prepare._build_teacher_metadata(config)
    stale_metadata = dict(expected_metadata)
    stale_metadata["fingerprint"] = "stale"
    save_json(teacher_prepare._teacher_metadata_path(config.dataset.prepared_root_dir), stale_metadata)

    with pytest.raises(ValueError, match="different or unknown teacher configuration"):
        teacher_prepare._validate_existing_teacher_metadata(
            config.dataset.prepared_root_dir,
            expected_metadata,
        )


def test_teacher_heatmap_shape_error_requires_explicit_overwrite(tmp_path: Path) -> None:
    heatmap_path = tmp_path / "teacher.npy"

    with pytest.raises(ValueError, match="--overwrite-teacher-heatmaps"):
        teacher_prepare._raise_teacher_heatmap_shape_error(
            sample_id="sample",
            heatmap_path=heatmap_path,
            actual_shape=(360, 640),
            expected_shape=(64, 96),
        )


def test_invalidate_resized_cache_removes_stale_sample_cache(
    tmp_path: Path,
    make_train_config,
) -> None:
    config = make_train_config(tmp_path / "prepared")
    config.dataset.use_resized_cache = True
    cache_dir = (
        config.dataset.prepared_root_dir
        / "train"
        / f"cache_{config.input.height}x{config.input.width}"
    )
    cache_dir.mkdir(parents=True)
    stale_cache = cache_dir / "sample.npz"
    other_cache = cache_dir / "other.npz"
    stale_cache.write_bytes(b"stale")
    other_cache.write_bytes(b"keep")

    teacher_prepare._invalidate_resized_cache(
        config.dataset,
        config.input,
        "train",
        "sample",
    )

    assert not stale_cache.exists()
    assert other_cache.exists()
