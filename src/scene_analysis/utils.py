from __future__ import annotations

from pathlib import Path

import numpy as np


def safe_mkdir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def clamp_roi(
    x: int,
    y: int,
    width: int,
    height: int,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    """Ограничить ROI границами изображения"""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive.")

    clamped_x = min(max(x, 0), image_width - 1)
    clamped_y = min(max(y, 0), image_height - 1)
    clamped_width = min(max(width, 1), image_width - clamped_x)
    clamped_height = min(max(height, 1), image_height - clamped_y)
    return clamped_x, clamped_y, clamped_width, clamped_height


def ensure_uint8_image(image: np.ndarray) -> np.ndarray:
    """Безопасно привести изображение к типу uint8"""
    if not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("Image must be a non-empty numpy array.")

    if image.dtype == np.uint8:
        return image.copy()

    if np.issubdtype(image.dtype, np.bool_):
        return image.astype(np.uint8) * 255

    converted = image.astype(np.float32, copy=False)
    min_value = float(np.min(converted))
    max_value = float(np.max(converted))
    if 0.0 <= min_value and max_value <= 1.0:
        converted = converted * 255.0

    return np.clip(converted, 0.0, 255.0).astype(np.uint8)


def maybe_colorize_mask(
    mask: np.ndarray | None,
    color: tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray | None:
    """Преобразовать бинарную маску в простое цветное наложение"""
    if mask is None:
        return None

    mask_uint8 = ensure_uint8_image(mask)
    if mask_uint8.ndim == 3 and mask_uint8.shape[2] == 3:
        return mask_uint8

    if mask_uint8.ndim == 3 and mask_uint8.shape[2] == 1:
        mask_uint8 = mask_uint8[:, :, 0]

    if mask_uint8.ndim != 2:
        raise ValueError("Mask must be 2D or 3-channel.")

    colored = np.zeros((*mask_uint8.shape, 3), dtype=np.uint8)
    active_pixels = mask_uint8 > 0
    colored[active_pixels] = color
    return colored


def timestamp_to_str(timestamp_ms: float) -> str:
    """Отформатировать миллисекунды как HH:MM:SS.mmm"""
    total_ms = max(int(round(timestamp_ms)), 0)
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    seconds, milliseconds = divmod(remainder, 1_000)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"
