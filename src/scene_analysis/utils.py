from __future__ import annotations

import math
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import cv2
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


def ensure_float32_array(value: np.ndarray) -> np.ndarray:
    """Безопасно привести массив к типу float32"""
    if not isinstance(value, np.ndarray) or value.size == 0:
        raise ValueError("Value must be a non-empty numpy array")
    return np.asarray(value, dtype=np.float32)


def safe_percentile(values: np.ndarray, percentile: float, fallback: float = 0.0) -> float:
    """Вычислить процентиль по конечным значениям или вернуть fallback"""
    array = ensure_float32_array(np.asarray(values))
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return float(fallback)
    return float(np.percentile(finite, float(percentile)))


def normalize_to_unit_range(
    values: np.ndarray,
    lower: float | None = None,
    upper: float | None = None,
    *,
    clip: bool = True,
) -> np.ndarray:
    """Нормализовать массив к диапазону [0, 1] без падений на константных картах"""
    array = ensure_float32_array(values)
    finite_mask = np.isfinite(array)
    if not np.any(finite_mask):
        return np.zeros_like(array, dtype=np.float32)

    finite_values = array[finite_mask]
    lower_bound = float(np.min(finite_values) if lower is None else lower)
    upper_bound = float(np.max(finite_values) if upper is None else upper)

    if not np.isfinite(lower_bound) or not np.isfinite(upper_bound):
        return np.zeros_like(array, dtype=np.float32)
    if upper_bound <= lower_bound + 1e-8:
        return np.zeros_like(array, dtype=np.float32)

    normalized = (array - lower_bound) / (upper_bound - lower_bound)
    if clip:
        normalized = np.clip(normalized, 0.0, 1.0)
    normalized = normalized.astype(np.float32, copy=False)
    normalized[~finite_mask] = 0.0
    return normalized


def make_odd_kernel(size: int) -> int:
    """Проверить, что размер ядра положительный и нечетный"""
    kernel = int(size)
    if kernel <= 0:
        raise ValueError("Kernel size must be greater than 0")
    if kernel % 2 == 0:
        raise ValueError("Kernel size must be odd")
    return kernel


def apply_binary_roi(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Применить бинарную float32-маску к карте признаков"""
    array = ensure_float32_array(values)
    roi_mask = ensure_float32_array(mask)
    if array.shape != roi_mask.shape:
        raise ValueError("ROI mask shape must match the input array shape")
    return (array * (roi_mask > 0).astype(np.float32)).astype(np.float32, copy=False)


def maybe_resize_float_map(values: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    """Масштабировать 2D float-карту до нужной формы"""
    array = ensure_float32_array(values)
    if array.ndim != 2:
        raise ValueError("Float map must be 2D for resizing")
    target_height, target_width = output_shape
    if target_height <= 0 or target_width <= 0:
        raise ValueError("Output shape must contain positive dimensions")
    if array.shape == (target_height, target_width):
        return array.copy()
    resized = cv2.resize(array, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    return resized.astype(np.float32, copy=False)


def list_files_by_extension(directory: Path, extension: str) -> list[Path]:
    """Рекурсивно собрать файлы с указанным расширением"""
    normalized_directory = Path(directory).expanduser()
    normalized_extension = extension.lower()
    if not normalized_extension.startswith("."):
        normalized_extension = f".{normalized_extension}"
    if not normalized_directory.exists():
        return []
    return sorted(
        path
        for path in normalized_directory.rglob(f"*{normalized_extension}")
        if path.is_file()
    )


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


def shorten_model_name(model_name: str | None) -> str:
    """Вернуть короткое имя модели"""
    if model_name is None:
        return "n/a"
    normalized = model_name.strip()
    if not normalized:
        return "n/a"
    return normalized.rsplit("/", maxsplit=1)[-1]


def to_serializable(value: Any) -> Any:
    """Подготовить произвольный объект к JSON/CSV сериализации"""
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return to_serializable(value.item())
    if isinstance(value, np.ndarray):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    if is_dataclass(value):
        return to_serializable(asdict(value))
    if isinstance(value, dict):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    return str(value)


def to_serializable_metadata(value: Any) -> Any:
    return to_serializable(value)
