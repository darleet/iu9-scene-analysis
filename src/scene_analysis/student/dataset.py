from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from scene_analysis.student.config import (
    StudentAugmentationConfig,
    StudentDatasetConfig,
    StudentInputConfig,
)


@dataclass(frozen=True)
class PreparedStudentSample:
    sample_id: str
    image_path: Path
    mask_path: Path
    teacher_heatmap_path: Path
    cache_path: Path | None = None


class StudentHeatmapDataset(Dataset[dict[str, Any]]):
    """Датасет с масками и с тепловой картой учителя"""

    def __init__(
        self,
        prepared_root_dir: Path,
        split: str,
        dataset_config: StudentDatasetConfig,
        input_config: StudentInputConfig,
        augmentation_config: StudentAugmentationConfig | None = None,
        *,
        training: bool = False,
    ) -> None:
        self.prepared_root_dir = prepared_root_dir.expanduser()
        self.split = split
        self.dataset_config = dataset_config
        self.input_config = input_config
        self.augmentation_config = augmentation_config
        self.training = training
        self.samples = self._discover_samples()
        self.transform = self._build_transform()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        cached_inputs = self._read_resized_cache(sample.cache_path)
        if cached_inputs is None:
            image_rgb = self._read_image_rgb(sample.image_path)
            mask = self._read_mask(sample.mask_path)
            teacher_heatmap = self._read_teacher_heatmap(sample.teacher_heatmap_path)
            image_rgb, mask, teacher_heatmap = self._resize_inputs(image_rgb, mask, teacher_heatmap)
        else:
            image_rgb, mask, teacher_heatmap = cached_inputs

        if self.transform is not None:
            transformed = self.transform(image=image_rgb, mask=mask, teacher_heatmap=teacher_heatmap)
            image_rgb = transformed["image"]
            mask = transformed["mask"]
            teacher_heatmap = transformed["teacher_heatmap"]

        teacher_heatmap = np.clip(teacher_heatmap.astype(np.float32), 0.0, 1.0)
        obstacle_target = (mask == self.dataset_config.obstacle_value).astype(np.float32)
        valid_mask = (mask != self.dataset_config.ignore_value).astype(np.float32)
        ignore_mask = (mask == self.dataset_config.ignore_value).astype(np.float32)

        image_tensor = self._normalize_image(image_rgb)
        return {
            "image": image_tensor,
            "obstacle_target": torch.from_numpy(obstacle_target[None, ...]),
            "valid_mask": torch.from_numpy(valid_mask[None, ...]),
            "ignore_mask": torch.from_numpy(ignore_mask[None, ...]),
            "teacher_heatmap": torch.from_numpy(teacher_heatmap[None, ...]),
            "sample_id": sample.sample_id,
        }

    def _discover_samples(self) -> list[PreparedStudentSample]:
        split_dir = self.prepared_root_dir / self.split
        images_dir = split_dir / "images"
        masks_dir = split_dir / "masks"
        teacher_dir = split_dir / "teacher_heatmaps"
        if not images_dir.exists():
            raise FileNotFoundError(f"Prepared images directory not found: {images_dir}")
        if not masks_dir.exists():
            raise FileNotFoundError(f"Prepared masks directory not found: {masks_dir}")
        if not teacher_dir.exists():
            raise FileNotFoundError(f"Prepared teacher heatmap directory not found: {teacher_dir}")

        samples: list[PreparedStudentSample] = []
        for image_path in sorted(images_dir.glob(f"*{self.dataset_config.image_suffix}")):
            sample_id = _sample_id_from_filename(image_path.name, self.dataset_config.image_suffix)
            if sample_id is None:
                continue
            mask_path = masks_dir / f"{sample_id}{self.dataset_config.mask_suffix}"
            teacher_path = teacher_dir / f"{sample_id}{self.dataset_config.teacher_suffix}"
            if not mask_path.exists():
                raise FileNotFoundError(f"Prepared mask is missing for sample '{sample_id}': {mask_path}")
            if not teacher_path.exists():
                raise FileNotFoundError(
                    f"Prepared teacher heatmap is missing for sample '{sample_id}': {teacher_path}"
                )
            samples.append(
                PreparedStudentSample(
                    sample_id=sample_id,
                    image_path=image_path,
                    mask_path=mask_path,
                    teacher_heatmap_path=teacher_path,
                    cache_path=_resized_cache_path(
                        self.prepared_root_dir,
                        self.split,
                        sample_id,
                        self.input_config,
                    )
                    if self.dataset_config.use_resized_cache
                    else None,
                )
            )
        if not samples:
            raise FileNotFoundError(f"No prepared student samples found in {split_dir}")
        return samples

    def _build_transform(self) -> A.Compose | None:
        config = self.augmentation_config
        if not self.training or config is None or not config.enabled:
            return None
        transforms: list[A.BasicTransform] = []
        if config.horizontal_flip_p > 0.0:
            transforms.append(A.HorizontalFlip(p=config.horizontal_flip_p))
        if config.brightness_contrast_p > 0.0:
            transforms.append(A.RandomBrightnessContrast(p=config.brightness_contrast_p))
        if config.blur_p > 0.0:
            transforms.append(A.Blur(blur_limit=3, p=config.blur_p))
        if config.noise_p > 0.0:
            transforms.append(A.GaussNoise(p=config.noise_p))
        if not transforms:
            return None
        return A.Compose(transforms, additional_targets={"teacher_heatmap": "mask"})

    def _resize_inputs(
        self,
        image_rgb: np.ndarray,
        mask: np.ndarray,
        teacher_heatmap: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        size = (self.input_config.width, self.input_config.height)
        image_resized = cv2.resize(image_rgb, size, interpolation=cv2.INTER_LINEAR)
        mask_resized = cv2.resize(mask, size, interpolation=cv2.INTER_NEAREST)
        teacher_resized = cv2.resize(teacher_heatmap, size, interpolation=cv2.INTER_LINEAR)
        return image_resized, mask_resized, teacher_resized.astype(np.float32, copy=False)

    def _read_resized_cache(self, path: Path | None) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        if path is None or not path.exists():
            return None
        with np.load(path) as cache:
            image_rgb = np.ascontiguousarray(cache["image_rgb"].astype(np.uint8, copy=False))
            mask = np.ascontiguousarray(cache["mask"].astype(np.uint8, copy=False))
            teacher_heatmap = np.ascontiguousarray(cache["teacher_heatmap"].astype(np.float32, copy=False))
        self._validate_cached_shapes(path, image_rgb, mask, teacher_heatmap)
        return image_rgb, mask, teacher_heatmap

    def _validate_cached_shapes(
        self,
        path: Path,
        image_rgb: np.ndarray,
        mask: np.ndarray,
        teacher_heatmap: np.ndarray,
    ) -> None:
        expected_image_shape = (self.input_config.height, self.input_config.width, 3)
        expected_map_shape = (self.input_config.height, self.input_config.width)
        if image_rgb.shape != expected_image_shape:
            raise ValueError(f"Cached image has shape {image_rgb.shape}, expected {expected_image_shape}: {path}")
        if mask.shape != expected_map_shape:
            raise ValueError(f"Cached mask has shape {mask.shape}, expected {expected_map_shape}: {path}")
        if teacher_heatmap.shape != expected_map_shape:
            raise ValueError(
                f"Cached teacher heatmap has shape {teacher_heatmap.shape}, expected {expected_map_shape}: {path}"
            )

    def _normalize_image(self, image_rgb: np.ndarray) -> torch.Tensor:
        image = image_rgb.astype(np.float32) / 255.0
        mean = np.asarray(self.input_config.normalize_mean, dtype=np.float32)
        std = np.asarray(self.input_config.normalize_std, dtype=np.float32)
        image = (image - mean) / std
        image = np.transpose(image, (2, 0, 1)).astype(np.float32, copy=False)
        return torch.from_numpy(image)

    @staticmethod
    def _read_image_rgb(path: Path) -> np.ndarray:
        image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise RuntimeError(f"Failed to read image: {path}")
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    @staticmethod
    def _read_mask(path: Path) -> np.ndarray:
        mask = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if mask is None:
            raise RuntimeError(f"Failed to read mask: {path}")
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        return mask

    @staticmethod
    def _read_teacher_heatmap(path: Path) -> np.ndarray:
        heatmap = np.load(path)
        if heatmap.ndim == 3 and 1 in heatmap.shape:
            heatmap = np.squeeze(heatmap)
        if heatmap.ndim != 2:
            raise ValueError(f"Teacher heatmap must be 2D, got shape {heatmap.shape} for {path}")
        return np.nan_to_num(heatmap.astype(np.float32), nan=0.0, posinf=1.0, neginf=0.0)


def build_resized_student_cache(
    prepared_root_dir: Path,
    split: str,
    dataset_config: StudentDatasetConfig,
    input_config: StudentInputConfig,
) -> dict[str, int | str]:
    dataset = StudentHeatmapDataset(
        prepared_root_dir,
        split,
        dataset_config,
        input_config,
        augmentation_config=None,
        training=False,
    )
    if not dataset_config.use_resized_cache:
        return {
            "split": split,
            "cache_dir": "",
            "total": len(dataset.samples),
            "created": 0,
            "skipped": len(dataset.samples),
        }

    cache_dir = _resized_cache_dir(dataset.prepared_root_dir, split, input_config)
    cache_dir.mkdir(parents=True, exist_ok=True)
    created = 0
    skipped = 0
    for sample in dataset.samples:
        if sample.cache_path is None:
            skipped += 1
            continue
        if sample.cache_path.exists() and not dataset_config.overwrite_resized_cache:
            skipped += 1
            continue

        image_rgb = StudentHeatmapDataset._read_image_rgb(sample.image_path)
        mask = StudentHeatmapDataset._read_mask(sample.mask_path)
        teacher_heatmap = StudentHeatmapDataset._read_teacher_heatmap(sample.teacher_heatmap_path)
        image_rgb, mask, teacher_heatmap = dataset._resize_inputs(image_rgb, mask, teacher_heatmap)

        tmp_path = sample.cache_path.with_name(f"{sample.cache_path.name}.tmp")
        with tmp_path.open("wb") as file:
            np.savez(
                file,
                image_rgb=image_rgb.astype(np.uint8, copy=False),
                mask=mask.astype(np.uint8, copy=False),
                teacher_heatmap=teacher_heatmap.astype(np.float16, copy=False),
            )
        tmp_path.replace(sample.cache_path)
        created += 1

    return {
        "split": split,
        "cache_dir": str(cache_dir),
        "total": len(dataset.samples),
        "created": created,
        "skipped": skipped,
    }


def _resized_cache_path(
    prepared_root_dir: Path,
    split: str,
    sample_id: str,
    input_config: StudentInputConfig,
) -> Path:
    return _resized_cache_dir(prepared_root_dir, split, input_config) / f"{sample_id}.npz"


def _resized_cache_dir(prepared_root_dir: Path, split: str, input_config: StudentInputConfig) -> Path:
    return prepared_root_dir / split / f"cache_{input_config.height}x{input_config.width}"


def _sample_id_from_filename(filename: str, suffix: str) -> str | None:
    if not filename.endswith(suffix):
        return None
    sample_id = filename[: -len(suffix)]
    return sample_id or None
