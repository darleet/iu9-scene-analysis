from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from scene_analysis.student.config import StudentDatasetConfig
from scene_analysis.utils import safe_mkdir


@dataclass(frozen=True)
class RawStudentSample:
    sample_id: str
    image_path: Path
    mask_path: Path


def discover_raw_samples(config: StudentDatasetConfig, limit: int | None = None) -> list[RawStudentSample]:
    """Получение набора пар картника/маска из датасета"""
    raw_root = config.raw_root_dir.expanduser()
    images_dir = _resolve_under_root(raw_root, config.images_dir)
    masks_dir = _resolve_under_root(raw_root, config.masks_dir)

    if not images_dir.exists():
        raise FileNotFoundError(
            f"Raw images directory not found: {images_dir}. Expected structure: "
            f"{raw_root}/images and {raw_root}/masks"
        )
    if not masks_dir.exists():
        raise FileNotFoundError(
            f"Raw masks directory not found: {masks_dir}. Expected structure: "
            f"{raw_root}/images and {raw_root}/masks"
        )

    mask_index = _build_mask_index(masks_dir, config.mask_suffix)
    image_paths = sorted(path for path in images_dir.rglob(f"*{config.image_suffix}") if path.is_file())
    samples: list[RawStudentSample] = []
    for image_path in image_paths:
        sample_id = image_path.stem
        mask_path = mask_index.get(sample_id)
        if mask_path is None:
            logger.warning("Skipping raw sample '{}' because matching mask is missing", sample_id)
            continue
        samples.append(RawStudentSample(sample_id=sample_id, image_path=image_path, mask_path=mask_path))
        if limit is not None and len(samples) >= limit:
            break

    if not samples:
        raise FileNotFoundError(
            "No image/mask pairs found. Expected at least files like "
            f"{images_dir}/sample_001{config.image_suffix} and "
            f"{masks_dir}/sample_001{config.mask_suffix}"
        )

    logger.info("Discovered {} raw student sample(s)", len(samples))
    return samples


def create_or_load_split(
    samples: list[RawStudentSample],
    prepared_root_dir: Path,
    train_ratio: float,
    seed: int,
) -> tuple[list[str], list[str]]:
    """Разбиение датасета на train/val выборки"""
    if not samples:
        raise ValueError("Cannot create train/val split for an empty sample list")

    split_dir = safe_mkdir(prepared_root_dir.expanduser() / "split")
    train_path = split_dir / "train.txt"
    val_path = split_dir / "val.txt"
    sample_ids = [sample.sample_id for sample in sorted(samples, key=lambda item: item.sample_id)]
    sample_id_set = set(sample_ids)

    if train_path.exists() and val_path.exists():
        train_ids = _read_split_file(train_path, sample_id_set)
        val_ids = _read_split_file(val_path, sample_id_set)
        if train_ids and val_ids:
            logger.info("Loaded existing student split from {}", split_dir)
            return train_ids, val_ids
        logger.warning("Existing split is empty after filtering current samples; recreating {}", split_dir)

    shuffled = sample_ids[:]
    random.Random(seed).shuffle(shuffled)
    train_count = int(round(len(shuffled) * train_ratio))
    train_count = min(max(train_count, 1), len(shuffled) - 1) if len(shuffled) > 1 else len(shuffled)
    train_ids = sorted(shuffled[:train_count])
    val_ids = sorted(shuffled[train_count:])
    if not val_ids and train_ids:
        val_ids = [train_ids.pop()]

    _write_split_file(train_path, train_ids)
    _write_split_file(val_path, val_ids)
    logger.info(
        "Created student split: train={} val={} seed={} ratio={}",
        len(train_ids),
        len(val_ids),
        seed,
        train_ratio,
    )
    return train_ids, val_ids


def _resolve_under_root(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def _build_mask_index(masks_dir: Path, mask_suffix: str) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for mask_path in sorted(path for path in masks_dir.rglob(f"*{Path(mask_suffix).suffix}") if path.is_file()):
        sample_id = _sample_id_from_mask(mask_path.name, mask_suffix)
        if sample_id is None:
            continue
        if sample_id in index:
            logger.warning("Duplicate mask for sample '{}' ignored: {}", sample_id, mask_path)
            continue
        index[sample_id] = mask_path
    return index


def _sample_id_from_mask(filename: str, mask_suffix: str) -> str | None:
    if not filename.endswith(mask_suffix):
        return None
    sample_id = filename[: -len(mask_suffix)]
    return sample_id or None


def _read_split_file(path: Path, allowed_ids: set[str]) -> list[str]:
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    filtered = [sample_id for sample_id in ids if sample_id in allowed_ids]
    skipped = len(ids) - len(filtered)
    if skipped:
        logger.warning("Ignored {} split id(s) from {} that are absent in current dataset", skipped, path)
    return filtered


def _write_split_file(path: Path, sample_ids: list[str]) -> None:
    path.write_text("\n".join(sample_ids) + ("\n" if sample_ids else ""), encoding="utf-8")
