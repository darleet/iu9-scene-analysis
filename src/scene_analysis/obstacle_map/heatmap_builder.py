from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from scene_analysis.config import ObstacleHeatmapConfig
from scene_analysis.obstacle_map.base import ObstacleHeatmapBuilder
from scene_analysis.obstacle_map.visualization import heatmap_to_bgr, overlay_heatmap_on_image
from scene_analysis.types import DepthResult, ObstacleHeatmapResult
from scene_analysis.utils import (
    apply_binary_roi,
    ensure_float32_array,
    ensure_uint8_image,
    make_odd_kernel,
    normalize_to_unit_range,
    safe_percentile,
)


class DepthToObstacleHeatmapBuilder(ObstacleHeatmapBuilder):
    """Базовый Depth -> Obstacle Heatmap builder с подавлением дороги"""

    def __init__(self, config: ObstacleHeatmapConfig) -> None:
        self.config = config

    def build(self, depth: DepthResult, image: np.ndarray) -> ObstacleHeatmapResult:
        depth_map, valid_mask, status = self._validate_depth(depth)
        if depth_map is None or valid_mask is None:
            metadata = self._collect_metadata(
                depth=depth,
                heatmap=None,
                status=status,
            )
            return ObstacleHeatmapResult(
                heatmap=None,
                heatmap_visualization=None,
                overlay_image=None,
                metadata=metadata,
            )

        near_score = self._compute_near_score(depth_map, valid_mask)
        roi_mask = self._build_roi_mask(depth_map.shape)
        row_baseline = self._estimate_row_baseline(near_score, roi_mask)
        suppressed = self._suppress_road(near_score, row_baseline, image)
        heatmap = self._postprocess_heatmap(suppressed, roi_mask, valid_mask)
        visualization = self._build_visualization(heatmap)
        overlay = self._build_overlay(image, heatmap)
        metadata = self._collect_metadata(depth=depth, heatmap=heatmap, status="ok")
        return ObstacleHeatmapResult(
            heatmap=heatmap,
            heatmap_visualization=visualization,
            overlay_image=overlay,
            metadata=metadata,
        )

    def _validate_depth(
        self,
        depth: DepthResult,
    ) -> tuple[np.ndarray | None, np.ndarray | None, str]:
        if depth.depth_map is None:
            return None, None, "no_depth"
        if not isinstance(depth.depth_map, np.ndarray) or depth.depth_map.size == 0:
            return None, None, "invalid_depth"
        if depth.depth_map.ndim != 2:
            return None, None, "invalid_depth"

        depth_map = ensure_float32_array(depth.depth_map)
        finite_mask = np.isfinite(depth_map)
        if not np.any(finite_mask):
            return None, None, "invalid_depth"

        finite_values = depth_map[finite_mask]
        fill_value = float(np.median(finite_values))
        sanitized = np.nan_to_num(
            depth_map,
            nan=fill_value,
            posinf=fill_value,
            neginf=fill_value,
        ).astype(np.float32, copy=False)
        return sanitized, finite_mask.astype(np.float32), "ok"

    def _compute_near_score(self, depth_map: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
        near_config = self.config.near_score
        valid_values = depth_map[valid_mask > 0]
        lower = safe_percentile(
            valid_values,
            near_config.clip_min_percentile,
            fallback=float(np.min(valid_values)) if valid_values.size else 0.0,
        )
        upper = safe_percentile(
            valid_values,
            near_config.clip_max_percentile,
            fallback=float(np.max(valid_values)) if valid_values.size else 1.0,
        )
        normalized = normalize_to_unit_range(depth_map, lower=lower, upper=upper, clip=True)

        near_score = normalized if near_config.use_relative_depth else 1.0 - normalized
        if near_config.invert_depth:
            near_score = 1.0 - near_score

        near_score = np.power(np.clip(near_score, 0.0, 1.0), near_config.gamma).astype(np.float32, copy=False)
        near_score[valid_mask <= 0] = 0.0
        return near_score

    def _build_roi_mask(self, shape: tuple[int, int]) -> np.ndarray:
        height, width = shape
        mask = np.ones((height, width), dtype=np.float32)
        roi_config = self.config.roi
        if not roi_config.enabled:
            return mask

        top_ignore = int(round(height * roi_config.top_ignore_ratio))
        left_ignore = int(round(width * roi_config.left_ignore_ratio))
        right_ignore = int(round(width * roi_config.right_ignore_ratio))
        keep_bottom = int(round(height * roi_config.bottom_keep_ratio))

        if top_ignore > 0:
            mask[:top_ignore, :] = 0.0
        if left_ignore > 0:
            mask[:, :left_ignore] = 0.0
        if right_ignore > 0:
            mask[:, width - right_ignore :] = 0.0
        if 0 < keep_bottom < height:
            mask[: height - keep_bottom, :] = 0.0
        return mask

    def _estimate_row_baseline(self, near_score: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:
        height = near_score.shape[0]
        baseline = np.zeros(height, dtype=np.float32)
        suppression_config = self.config.road_suppression
        if not suppression_config.enabled:
            return baseline

        strip_height = max(1, int(round(height * suppression_config.bottom_strip_ratio)))
        strip_start = max(0, height - strip_height)
        percentile = suppression_config.baseline_quantile * 100.0
        for row_index in range(height):
            row_mask = roi_mask[row_index] > 0.0
            if not np.any(row_mask):
                baseline[row_index] = baseline[row_index - 1] if row_index > 0 else 0.0
                continue

            row_values = near_score[row_index][row_mask]
            fallback = float(np.mean(row_values)) if row_values.size else 0.0
            baseline[row_index] = safe_percentile(row_values, percentile, fallback=fallback)

        smooth_kernel = make_odd_kernel(suppression_config.row_smooth_kernel)
        baseline_column = baseline.reshape(-1, 1)
        if baseline_column.shape[0] > 1 and smooth_kernel > 1:
            baseline_column = cv2.GaussianBlur(
                baseline_column,
                (1, smooth_kernel),
                sigmaX=0,
                sigmaY=0,
                borderType=cv2.BORDER_REPLICATE,
            )
        baseline = baseline_column.reshape(-1)

        baseline = np.maximum.accumulate(baseline).astype(np.float32, copy=False)
        if strip_start > 0:
            upper_weights = np.linspace(0.0, 1.0, num=strip_start, endpoint=False, dtype=np.float32)
            baseline[:strip_start] *= upper_weights

        baseline[baseline < suppression_config.min_row_activation] = 0.0
        return baseline

    def _suppress_road(
        self,
        near_score: np.ndarray,
        row_baseline: np.ndarray,
        image: np.ndarray,
    ) -> np.ndarray:
        suppression_config = self.config.road_suppression
        if not suppression_config.enabled:
            return near_score.copy()

        baseline_map = np.repeat(row_baseline[:, None], near_score.shape[1], axis=1)
        suppressed = np.clip(
            near_score - suppression_config.suppression_strength * baseline_map,
            0.0,
            1.0,
        ).astype(np.float32, copy=False)

        if suppression_config.preserve_vertical_edges:
            edge_strength = self._compute_vertical_edge_strength(image, near_score)
            suppressed = np.clip(
                suppressed + suppression_config.edge_weight * edge_strength * near_score,
                0.0,
                1.0,
            ).astype(np.float32, copy=False)

        return suppressed

    def _postprocess_heatmap(
        self,
        heatmap: np.ndarray,
        roi_mask: np.ndarray,
        valid_mask: np.ndarray,
    ) -> np.ndarray:
        postprocess_config = self.config.postprocess
        processed = apply_binary_roi(heatmap, roi_mask)
        processed = apply_binary_roi(processed, valid_mask)

        blur_kernel = make_odd_kernel(postprocess_config.blur_kernel_size)
        if blur_kernel > 1:
            processed = cv2.GaussianBlur(
                processed,
                (blur_kernel, blur_kernel),
                sigmaX=0,
                sigmaY=0,
                borderType=cv2.BORDER_REPLICATE,
            )

        morph_kernel = make_odd_kernel(postprocess_config.morph_kernel_size)
        if morph_kernel > 1:
            kernel = np.ones((morph_kernel, morph_kernel), dtype=np.uint8)
            morph_input = np.clip(processed * 255.0, 0.0, 255.0).astype(np.uint8)
            morph_input = cv2.morphologyEx(morph_input, cv2.MORPH_OPEN, kernel)
            morph_input = cv2.morphologyEx(morph_input, cv2.MORPH_CLOSE, kernel)
            processed = (morph_input.astype(np.float32) / 255.0).astype(np.float32, copy=False)

        processed[processed < postprocess_config.min_activation] = 0.0
        processed = np.clip(processed, 0.0, 1.0).astype(np.float32, copy=False)

        if postprocess_config.normalize_output and np.any(processed > 0.0):
            active_values = processed[processed > 0.0]
            upper_bound = safe_percentile(
                active_values,
                99.5,
                fallback=float(np.max(active_values)) if active_values.size else 1.0,
            )
            processed = normalize_to_unit_range(
                processed,
                lower=postprocess_config.min_activation,
                upper=upper_bound,
                clip=True,
            )

        return processed.astype(np.float32, copy=False)

    def _build_visualization(self, heatmap: np.ndarray) -> np.ndarray:
        return heatmap_to_bgr(
            heatmap,
            colormap=self.config.visualization.colormap,
        )

    def _build_overlay(self, image: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
        return overlay_heatmap_on_image(
            image=image,
            heatmap=heatmap,
            alpha=0.45,
            colormap=self.config.visualization.colormap,
        )

    def _collect_metadata(
        self,
        depth: DepthResult,
        heatmap: np.ndarray | None,
        status: str,
    ) -> dict[str, Any]:
        if heatmap is None:
            heatmap_min = None
            heatmap_max = None
            heatmap_mean = None
            heatmap_nonzero_ratio = None
        else:
            prepared = np.clip(ensure_float32_array(heatmap), 0.0, 1.0)
            heatmap_min = float(np.min(prepared))
            heatmap_max = float(np.max(prepared))
            heatmap_mean = float(np.mean(prepared))
            heatmap_nonzero_ratio = float(np.mean(prepared > 0.0))

        return {
            "status": status,
            "depth_available": depth.depth_map is not None,
            "scale_type": depth.metadata.get("scale_type", "unknown"),
            "heatmap_min": heatmap_min,
            "heatmap_max": heatmap_max,
            "heatmap_mean": heatmap_mean,
            "heatmap_nonzero_ratio": heatmap_nonzero_ratio,
            "roi_enabled": self.config.roi.enabled,
            "road_suppression_enabled": self.config.road_suppression.enabled,
            "suppression_mode": self.config.road_suppression.mode,
            "suppression_strength": self.config.road_suppression.suppression_strength,
            "clip_min_percentile": self.config.near_score.clip_min_percentile,
            "clip_max_percentile": self.config.near_score.clip_max_percentile,
            "gamma": self.config.near_score.gamma,
        }

    @staticmethod
    def _compute_vertical_edge_strength(image: np.ndarray, near_score: np.ndarray) -> np.ndarray:
        prepared_image = ensure_uint8_image(image)
        if prepared_image.ndim == 2:
            grayscale = prepared_image
        elif prepared_image.ndim == 3 and prepared_image.shape[2] == 1:
            grayscale = prepared_image[:, :, 0]
        elif prepared_image.ndim == 3 and prepared_image.shape[2] == 3:
            grayscale = cv2.cvtColor(prepared_image, cv2.COLOR_BGR2GRAY)
        else:
            raise ValueError("Input image must have shape HxW or HxWx3")

        image_edges = np.abs(cv2.Sobel(grayscale.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3))
        near_edges = np.abs(cv2.Sobel(near_score.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3))
        image_edges = normalize_to_unit_range(image_edges, clip=True)
        near_edges = normalize_to_unit_range(near_edges, clip=True)
        return np.maximum(image_edges, near_edges).astype(np.float32, copy=False)
