from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from loguru import logger

from scene_analysis.config import load_config
from scene_analysis.depth.base import create_depth_estimator
from scene_analysis.obstacle_map.base import create_obstacle_heatmap_builder
from scene_analysis.pipeline.mvp_pipeline import MVPSceneAnalysisPipeline
from scene_analysis.preprocessing.frame_preprocessor import FramePreprocessor
from scene_analysis.student.artifacts import save_json
from scene_analysis.student.config import StudentDatasetConfig, StudentInputConfig, StudentTrainConfig
from scene_analysis.student.split import RawStudentSample, create_or_load_split, discover_raw_samples
from scene_analysis.student.visualization import save_heatmap_preview
from scene_analysis.types import FrameData
from scene_analysis.utils import safe_mkdir


def prepare_student_data(
    config: StudentTrainConfig,
    *,
    overwrite_teacher_heatmaps: bool = False,
    limit: int | None = None,
) -> dict[str, Any]:
    """Подготовка train/val выборок и тепловой карты учиеля"""
    effective_overwrite = overwrite_teacher_heatmaps or config.teacher.overwrite_teacher_heatmaps
    samples = discover_raw_samples(config.dataset, limit=limit)
    train_ids, val_ids = create_or_load_split(
        samples=samples,
        prepared_root_dir=config.dataset.prepared_root_dir,
        train_ratio=config.dataset.train_ratio,
        seed=config.dataset.split_seed,
        train_split_names=config.dataset.raw_train_splits,
        val_split_names=config.dataset.raw_val_splits,
    )
    sample_by_id = {sample.sample_id: sample for sample in samples}

    copied = {"train": 0, "val": 0}
    generated = {"train": 0, "val": 0}
    skipped_existing = {"train": 0, "val": 0}
    preview_paths: list[str] = []
    pipeline: MVPSceneAnalysisPipeline | None = None

    for split_name, split_ids in (("train", train_ids), ("val", val_ids)):
        split_samples = [sample_by_id[sample_id] for sample_id in split_ids if sample_id in sample_by_id]
        split_dirs = _prepare_split_dirs(config.dataset.prepared_root_dir, split_name)
        for sample in split_samples:
            prepared_image, prepared_mask = _copy_sample(sample, split_dirs, config.dataset)
            copied[split_name] += 1

            heatmap_path = split_dirs["teacher_heatmaps"] / f"{sample.sample_id}{config.dataset.teacher_suffix}"
            heatmap_exists = heatmap_path.exists()
            expected_shape = _expected_teacher_heatmap_shape(config.input)
            if heatmap_exists and not effective_overwrite:
                heatmap_shape = _teacher_heatmap_shape(heatmap_path)
                if heatmap_shape != expected_shape:
                    _raise_teacher_heatmap_shape_error(
                        sample_id=sample.sample_id,
                        heatmap_path=heatmap_path,
                        actual_shape=heatmap_shape,
                        expected_shape=expected_shape,
                    )
                skipped_existing[split_name] += 1
            else:
                if pipeline is None:
                    pipeline = _build_teacher_pipeline(config.teacher.config_path, config.input)
                _generate_teacher_heatmap(
                    pipeline=pipeline,
                    image_path=prepared_image,
                    sample_id=sample.sample_id,
                    output_path=heatmap_path,
                )
                _invalidate_resized_cache(config.dataset, config.input, split_name, sample.sample_id)
                generated[split_name] += 1

            if len(preview_paths) < 8 and heatmap_path.exists():
                preview_path = config.dataset.prepared_root_dir / "teacher_previews" / f"{split_name}_{sample.sample_id}.png"
                save_heatmap_preview(np.load(heatmap_path), preview_path)
                preview_paths.append(str(preview_path))

    summary = {
        "status": "ok",
        "raw_root_dir": str(config.dataset.raw_root_dir),
        "prepared_root_dir": str(config.dataset.prepared_root_dir),
        "train_samples": len(train_ids),
        "val_samples": len(val_ids),
        "copied": copied,
        "teacher_heatmaps_generated": generated,
        "teacher_heatmaps_skipped_existing": skipped_existing,
        "overwrite_teacher_heatmaps": effective_overwrite,
        "limit": limit,
        "preview_paths": preview_paths,
        "split_train": str(config.dataset.prepared_root_dir / "split" / "train.txt"),
        "split_val": str(config.dataset.prepared_root_dir / "split" / "val.txt"),
    }
    save_json(config.dataset.prepared_root_dir / "prepare_summary.json", summary)
    logger.info(
        "Prepared student data: train={} val={} generated={} skipped_existing={}",
        len(train_ids),
        len(val_ids),
        sum(generated.values()),
        sum(skipped_existing.values()),
    )
    return summary


def _prepare_split_dirs(prepared_root_dir: Path, split_name: str) -> dict[str, Path]:
    split_root = safe_mkdir(prepared_root_dir.expanduser() / split_name)
    return {
        "images": safe_mkdir(split_root / "images"),
        "masks": safe_mkdir(split_root / "masks"),
        "teacher_heatmaps": safe_mkdir(split_root / "teacher_heatmaps"),
    }


def _copy_sample(
    sample: RawStudentSample,
    split_dirs: dict[str, Path],
    dataset_config: StudentDatasetConfig,
) -> tuple[Path, Path]:
    image_output = split_dirs["images"] / sample.image_path.name
    mask_output = split_dirs["masks"] / f"{sample.sample_id}{dataset_config.mask_suffix}"
    if sample.image_path.resolve() != image_output.resolve():
        shutil.copy2(sample.image_path, image_output)
    if _should_remap_mask(dataset_config):
        _write_remapped_mask(sample.mask_path, mask_output, dataset_config)
    elif sample.mask_path.resolve() != mask_output.resolve():
        shutil.copy2(sample.mask_path, mask_output)
    return image_output, mask_output


def _should_remap_mask(dataset_config: StudentDatasetConfig) -> bool:
    return any(
        (
            dataset_config.mask_obstacle_values,
            dataset_config.mask_background_values,
            dataset_config.mask_ignore_values,
            dataset_config.mask_unmapped_value is not None,
        )
    )


def _write_remapped_mask(
    input_path: Path,
    output_path: Path,
    dataset_config: StudentDatasetConfig,
) -> None:
    mask = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        raise RuntimeError(f"Failed to read raw mask for remapping: {input_path}")
    if mask.ndim == 3:
        mask = mask[:, :, 0]

    default_value = dataset_config.mask_unmapped_value
    if default_value is None:
        default_value = dataset_config.ignore_value
    remapped = np.full(mask.shape, int(default_value), dtype=np.uint8)
    _assign_mask_values(
        remapped,
        mask,
        dataset_config.mask_background_values,
        dataset_config.background_value,
    )
    _assign_mask_values(
        remapped,
        mask,
        dataset_config.mask_obstacle_values,
        dataset_config.obstacle_value,
    )
    _assign_mask_values(
        remapped,
        mask,
        dataset_config.mask_ignore_values,
        dataset_config.ignore_value,
    )
    safe_mkdir(output_path.parent)
    if not cv2.imwrite(str(output_path), remapped):
        raise RuntimeError(f"Failed to write remapped mask: {output_path}")


def _assign_mask_values(
    output_mask: np.ndarray,
    source_mask: np.ndarray,
    source_values: list[int],
    target_value: int,
) -> None:
    if not source_values:
        return
    output_mask[np.isin(source_mask, source_values)] = int(target_value)


def _build_teacher_pipeline(config_path: Path, input_config: StudentInputConfig) -> MVPSceneAnalysisPipeline:
    logger.info("Loading teacher pipeline from {}", config_path)
    teacher_config = load_config(config_path)
    teacher_config.preprocessing.roi.enabled = False
    teacher_config.preprocessing.resize_width = input_config.width
    teacher_config.preprocessing.resize_height = input_config.height
    logger.info(
        "Teacher preprocessing: roi disabled, resize={}x{}",
        input_config.width,
        input_config.height,
    )
    preprocessor = FramePreprocessor(teacher_config.preprocessing)
    depth_estimator = create_depth_estimator(teacher_config.depth)
    obstacle_builder = create_obstacle_heatmap_builder(teacher_config.obstacle_heatmap)
    return MVPSceneAnalysisPipeline(
        preprocessor=preprocessor,
        depth_estimator=depth_estimator,
        obstacle_heatmap_builder=obstacle_builder,
    )


def _generate_teacher_heatmap(
    pipeline: MVPSceneAnalysisPipeline,
    image_path: Path,
    sample_id: str,
    output_path: Path,
) -> None:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"Failed to read prepared image for teacher heatmap: {image_path}")
    height, width = image_bgr.shape[:2]
    frame = FrameData(
        frame_index=0,
        timestamp_ms=0.0,
        image=image_bgr,
        source_path=str(image_path),
        width=width,
        height=height,
    )
    result = pipeline.process_frame(frame)
    if result.obstacle_heatmap.heatmap is None:
        raise RuntimeError(f"Teacher heatmap is unavailable for sample '{sample_id}'")
    safe_mkdir(output_path.parent)

    np.save(output_path, result.obstacle_heatmap.heatmap.astype("float32", copy=False))


def _teacher_heatmap_shape(path: Path) -> tuple[int, ...] | None:
    try:
        heatmap = np.load(path, mmap_mode="r")
    except (OSError, ValueError) as exc:
        logger.warning("Existing teacher heatmap is unreadable and must be regenerated: {} ({})", path, exc)
        return None
    shape = tuple(int(dimension) for dimension in heatmap.shape)
    if len(shape) == 3 and 1 in shape:
        return tuple(dimension for dimension in shape if dimension != 1)
    return shape


def _expected_teacher_heatmap_shape(input_config: StudentInputConfig) -> tuple[int, int]:
    return (input_config.height, input_config.width)


def _raise_teacher_heatmap_shape_error(
    *,
    sample_id: str,
    heatmap_path: Path,
    actual_shape: tuple[int, ...] | None,
    expected_shape: tuple[int, int],
) -> None:
    raise ValueError(
        "Existing teacher heatmap has an incompatible shape. "
        f"sample='{sample_id}' path='{heatmap_path}' shape={actual_shape} expected={expected_shape}. "
        "Regenerate teacher heatmaps explicitly with: "
        "poetry run scene-analysis prepare-student-data --config <student_train.yaml> --overwrite-teacher-heatmaps"
    )


def _invalidate_resized_cache(
    dataset_config: StudentDatasetConfig,
    input_config: StudentInputConfig,
    split_name: str,
    sample_id: str,
) -> None:
    if not dataset_config.use_resized_cache:
        return
    cache_path = (
        dataset_config.prepared_root_dir.expanduser()
        / split_name
        / f"cache_{input_config.height}x{input_config.width}"
        / f"{sample_id}.npz"
    )
    if cache_path.exists():
        cache_path.unlink()
