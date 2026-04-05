from __future__ import annotations

from pathlib import Path

from loguru import logger

from scene_analysis.config import EvaluationDatasetConfig
from scene_analysis.evaluation.types import EvaluationSample
from scene_analysis.utils import list_files_by_extension


class RoadObstacle21Dataset:
    def __init__(self, config: EvaluationDatasetConfig) -> None:
        self.config = config
        self.root_dir = config.root_dir
        self.images_dir = self._resolve_dir(config.images_dir)
        self.masks_dir = self._resolve_dir(config.masks_dir)
        self.predictions_dir = self._resolve_dir(config.predictions_dir)
        self.split_file = self._resolve_optional_path(config.split_file)

    def discover_samples(self) -> list[EvaluationSample]:
        if not self.predictions_dir.exists():
            raise FileNotFoundError(f"Predictions directory not found: {self.predictions_dir}")
        if not self.masks_dir.exists():
            raise FileNotFoundError(f"Masks directory not found: {self.masks_dir}")

        split_ids = self._load_split_ids()
        mask_index = self._build_index(self.masks_dir, self.config.file_extension_masks)
        image_index = (
            self._build_index(self.images_dir, self.config.file_extension_images)
            if self.images_dir.exists()
            else {}
        )

        samples: list[EvaluationSample] = []
        prediction_files = list_files_by_extension(
            self.predictions_dir,
            self.config.file_extension_predictions,
        )
        for prediction_path in prediction_files:
            sample_id = prediction_path.stem
            if split_ids is not None and sample_id not in split_ids:
                continue

            mask_path = mask_index.get(sample_id)
            if mask_path is None:
                logger.warning("Skipping sample '{}' because mask file is missing", sample_id)
                continue

            samples.append(
                EvaluationSample(
                    sample_id=sample_id,
                    image_path=image_index.get(sample_id),
                    mask_path=mask_path,
                    prediction_path=prediction_path,
                )
            )

        samples.sort(key=lambda item: item.sample_id)
        logger.info("Discovered {} evaluation sample(s) in {}", len(samples), self.predictions_dir)
        return samples

    def _resolve_dir(self, path: Path) -> Path:
        return path if path.is_absolute() else self.root_dir / path

    def _resolve_optional_path(self, path: Path | None) -> Path | None:
        if path is None:
            return None
        return path if path.is_absolute() else self.root_dir / path

    def _load_split_ids(self) -> set[str] | None:
        if self.split_file is None:
            return None
        if not self.split_file.exists():
            raise FileNotFoundError(f"Split file not found: {self.split_file}")

        sample_ids = {
            line.strip()
            for line in self.split_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        logger.info("Loaded {} sample id(s) from split file {}", len(sample_ids), self.split_file)
        return sample_ids

    @staticmethod
    def _build_index(directory: Path, extension: str) -> dict[str, Path]:
        index: dict[str, Path] = {}
        for path in list_files_by_extension(directory, extension):
            sample_id = path.stem
            if sample_id in index:
                logger.warning(
                    "Duplicate file for sample '{}' detected: '{}' will be ignored in favor of '{}'",
                    sample_id,
                    path,
                    index[sample_id],
                )
                continue
            index[sample_id] = path
        return index
