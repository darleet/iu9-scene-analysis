"""Препроцессинг кадров для MVP-пайплайна"""

from __future__ import annotations

import cv2
import numpy as np

from scene_analysis.config import PreprocessingConfig
from scene_analysis.utils import clamp_roi, ensure_uint8_image


class FramePreprocessor:
    """Покадровый препроцессор"""

    def __init__(self, config: PreprocessingConfig) -> None:
        self.config = config

    def process(self, image: np.ndarray) -> np.ndarray:
        """Обрезать, изменить размер и при необходимости нормализовать изображение"""
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("Input image must be a non-empty numpy array.")
        if image.ndim not in (2, 3):
            raise ValueError("Input image must be 2D or 3D.")

        working_image = image
        if self.config.roi.enabled:
            image_height, image_width = working_image.shape[:2]
            x, y, width, height = clamp_roi(
                x=self.config.roi.x,
                y=self.config.roi.y,
                width=self.config.roi.width,
                height=self.config.roi.height,
                image_width=image_width,
                image_height=image_height,
            )
            working_image = working_image[y : y + height, x : x + width]

        resized = cv2.resize(
            working_image,
            (self.config.resize_width, self.config.resize_height),
            interpolation=cv2.INTER_LINEAR,
        )

        if self.config.normalize_to_float:
            processed = resized.astype(np.float32)
            min_value = float(np.min(processed))
            max_value = float(np.max(processed))
            if min_value < 0.0 or max_value > 1.0:
                processed = np.clip(processed / 255.0, 0.0, 1.0)
            else:
                processed = np.clip(processed, 0.0, 1.0)
        else:
            processed = ensure_uint8_image(resized)

        if processed.shape[0] != self.config.resize_height or processed.shape[1] != self.config.resize_width:
            raise RuntimeError("Preprocessing produced an image with unexpected size.")

        return processed
