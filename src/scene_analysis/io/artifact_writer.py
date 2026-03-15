"""Сохранение артефактов обработанных кадров"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from scene_analysis.config import OutputConfig
from scene_analysis.types import FrameData, SceneAnalysisResult
from scene_analysis.utils import ensure_uint8_image, safe_mkdir, timestamp_to_str


class ArtifactWriter:
    def __init__(self, config: OutputConfig) -> None:
        self.config = config
        self.output_dir = safe_mkdir(config.output_dir)
        self.original_dir = self.output_dir / "original_frames"
        self.preprocessed_dir = self.output_dir / "preprocessed_frames"
        self.overlay_dir = self.output_dir / "overlay_frames"
        self.results_path = self.output_dir / "results.jsonl"
        self._jsonl_file = None

        if config.save_original_frames:
            safe_mkdir(self.original_dir)
        if config.save_preprocessed_frames:
            safe_mkdir(self.preprocessed_dir)
        if config.save_overlay_frames:
            safe_mkdir(self.overlay_dir)
        if config.save_jsonl:
            self._jsonl_file = self.results_path.open("a", encoding="utf-8")

    def save_original(self, frame: FrameData) -> None:
        """Сохранить исходный кадр в PNG"""
        if not self.config.save_original_frames:
            return
        self._write_image(self.original_dir / self._frame_filename(frame.frame_index), frame.image)

    def save_preprocessed(self, frame_index: int, image: np.ndarray) -> None:
        """Сохранить препроцессированный кадр в PNG"""
        if not self.config.save_preprocessed_frames:
            return
        self._write_image(self.preprocessed_dir / self._frame_filename(frame_index), image)

    def save_overlay(self, frame_index: int, image: np.ndarray) -> None:
        """Сохранить overlay-кадр в PNG"""
        if not self.config.save_overlay_frames:
            return
        self._write_image(self.overlay_dir / self._frame_filename(frame_index), image)

    def append_result(self, result: SceneAnalysisResult) -> None:
        """Добавить одну JSONL-запись с метаданными"""
        if not self.config.save_jsonl or self._jsonl_file is None:
            return

        record = {
            "frame": {
                "frame_index": result.frame.frame_index,
                "timestamp_ms": result.frame.timestamp_ms,
                "timestamp": timestamp_to_str(result.frame.timestamp_ms),
                "source_path": result.frame.source_path,
                "width": result.frame.width,
                "height": result.frame.height,
            },
            "preprocessed_image": self._describe_array(result.preprocessed_image),
            "overlay_image": self._describe_array(result.overlay_image),
            "depth": {
                "depth_map": self._describe_array(result.depth.depth_map),
                "confidence_map": self._describe_array(result.depth.confidence_map),
                "metadata": self._to_jsonable(result.depth.metadata),
            },
            "obstacle_map": {
                "obstacle_mask": self._describe_array(result.obstacle_map.obstacle_mask),
                "occupancy_grid": self._describe_array(result.obstacle_map.occupancy_grid),
                "costmap": self._describe_array(result.obstacle_map.costmap),
                "metadata": self._to_jsonable(result.obstacle_map.metadata),
            },
            "dynamic_objects": [
                {
                    "track_id": obj.track_id,
                    "label": obj.label,
                    "confidence": obj.confidence,
                    "bbox": list(obj.bbox) if obj.bbox is not None else None,
                }
                for obj in result.dynamic_objects
            ],
            "metadata": self._to_jsonable(result.metadata),
        }
        self._jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._jsonl_file.flush()

    def close(self) -> None:
        """Закрыть открытые файлы"""
        if self._jsonl_file is not None:
            self._jsonl_file.close()
            self._jsonl_file = None

    @staticmethod
    def _frame_filename(frame_index: int) -> str:
        return f"frame_{frame_index:06d}.png"

    def _write_image(self, path: Path, image: np.ndarray) -> None:
        prepared = self._prepare_image_for_saving(image)
        if not cv2.imwrite(str(path), prepared):
            raise IOError(f"Failed to save image: {path}")

    @staticmethod
    def _prepare_image_for_saving(image: np.ndarray) -> np.ndarray:
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("Image must be a non-empty numpy array.")

        prepared = ensure_uint8_image(image)
        if prepared.ndim == 3 and prepared.shape[2] == 1:
            return prepared[:, :, 0]
        return prepared

    @staticmethod
    def _describe_array(value: np.ndarray | None) -> dict[str, Any] | None:
        if value is None:
            return None
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }

    @classmethod
    def _to_jsonable(cls, value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return cls._describe_array(value)
        if isinstance(value, dict):
            return {str(key): cls._to_jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._to_jsonable(item) for item in value]
        return str(value)
