from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from scene_analysis.config import DepthConfig, OutputConfig
from scene_analysis.depth.visualization import colorize_depth_map
from scene_analysis.types import FrameData, SceneAnalysisResult
from scene_analysis.utils import (
    ensure_float32_array,
    ensure_uint8_image,
    safe_mkdir,
    timestamp_to_str,
    to_serializable_metadata,
)


class ArtifactWriter:
    def __init__(self, output_config: OutputConfig, depth_config: DepthConfig) -> None:
        self.output_config = output_config
        self.depth_config = depth_config
        self.output_dir = safe_mkdir(output_config.output_dir)
        self.original_dir = self.output_dir / "original_frames"
        self.preprocessed_dir = self.output_dir / "preprocessed_frames"
        self.depth_npy_dir = self.output_dir / "depth_npy"
        self.depth_colormap_dir = self.output_dir / "depth_colormap"
        self.overlay_dir = self.output_dir / "overlay_frames"
        self.results_path = self.output_dir / "results.jsonl"
        self._jsonl_file = None

        if output_config.save_original_frames:
            safe_mkdir(self.original_dir)
        if output_config.save_preprocessed_frames:
            safe_mkdir(self.preprocessed_dir)
        if depth_config.save_raw_depth_npy:
            safe_mkdir(self.depth_npy_dir)
        if depth_config.save_depth_colormap:
            safe_mkdir(self.depth_colormap_dir)
        if output_config.save_overlay_frames:
            safe_mkdir(self.overlay_dir)
        if output_config.save_jsonl:
            self._jsonl_file = self.results_path.open("a", encoding="utf-8")

    def save_original(self, frame: FrameData) -> None:
        """Сохранить исходный кадр в PNG"""
        if not self.output_config.save_original_frames:
            return
        self._write_image(self.original_dir / self._frame_filename(frame.frame_index), frame.image)

    def save_preprocessed(self, frame_index: int, image: np.ndarray) -> None:
        """Сохранить препроцессированный кадр в PNG"""
        if not self.output_config.save_preprocessed_frames:
            return
        self._write_image(self.preprocessed_dir / self._frame_filename(frame_index), image)

    def save_depth_npy(self, frame_index: int, depth_map: np.ndarray) -> None:
        """Сохранить raw depth map в формате NumPy"""
        if not self.depth_config.save_raw_depth_npy:
            return

        prepared = ensure_float32_array(depth_map)
        np.save(self.depth_npy_dir / self._depth_filename(frame_index, ".npy"), prepared)

    def save_depth_colormap(self, frame_index: int, image: np.ndarray) -> None:
        """Сохранить colorized depth map в PNG"""
        if not self.depth_config.save_depth_colormap:
            return
        self._write_image(self.depth_colormap_dir / self._depth_filename(frame_index, ".png"), image)

    def save_overlay(self, frame_index: int, image: np.ndarray) -> None:
        """Сохранить overlay-кадр в PNG"""
        if not self.output_config.save_overlay_frames:
            return
        self._write_image(self.overlay_dir / self._frame_filename(frame_index), image)

    def append_result(self, result: SceneAnalysisResult) -> None:
        """Добавить одну JSONL-запись с метаданными"""
        if result.depth.depth_map is not None:
            if self.depth_config.save_raw_depth_npy:
                self.save_depth_npy(result.frame.frame_index, result.depth.depth_map)
            if self.depth_config.save_depth_colormap:
                self.save_depth_colormap(
                    result.frame.frame_index,
                    self._build_depth_colormap(result.depth.depth_map),
                )

        if not self.output_config.save_jsonl or self._jsonl_file is None:
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
                "status": result.depth.metadata.get("status"),
                "provider": result.depth.metadata.get("provider"),
                "model": result.depth.metadata.get("model"),
                "device": result.depth.metadata.get("device"),
                "inference_ms": result.depth.metadata.get("inference_ms"),
                "scale_type": result.depth.metadata.get("scale_type"),
                "depth_min": result.depth.metadata.get("depth_min"),
                "depth_max": result.depth.metadata.get("depth_max"),
                "depth_mean": result.depth.metadata.get("depth_mean"),
                "depth_map_shape": list(result.depth.depth_map.shape) if result.depth.depth_map is not None else None,
                "confidence_map_shape": (
                    list(result.depth.confidence_map.shape) if result.depth.confidence_map is not None else None
                ),
            },
            "obstacle_map": {
                "status": result.obstacle_map.metadata.get("status"),
                "obstacle_mask": self._describe_array(result.obstacle_map.obstacle_mask),
                "occupancy_grid": self._describe_array(result.obstacle_map.occupancy_grid),
                "costmap": self._describe_array(result.obstacle_map.costmap),
                "metadata": to_serializable_metadata(result.obstacle_map.metadata),
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
            "metadata": to_serializable_metadata(result.metadata),
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

    @staticmethod
    def _depth_filename(frame_index: int, suffix: str) -> str:
        return f"frame_{frame_index:06d}{suffix}"

    def _write_image(self, path: Path, image: np.ndarray) -> None:
        prepared = self._prepare_image_for_saving(image)
        if not cv2.imwrite(str(path), prepared):
            raise IOError(f"Failed to save image: {path}")

    @staticmethod
    def _prepare_image_for_saving(image: np.ndarray) -> np.ndarray:
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("Image must be a non-empty numpy array")

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

    def _build_depth_colormap(self, depth_map: np.ndarray) -> np.ndarray:
        min_percentile = self.depth_config.clip_percentiles.min if self.depth_config.normalize_depth_for_viz else 0.0
        max_percentile = self.depth_config.clip_percentiles.max if self.depth_config.normalize_depth_for_viz else 100.0
        return colorize_depth_map(
            depth_map=depth_map,
            min_percentile=min_percentile,
            max_percentile=max_percentile,
        )
