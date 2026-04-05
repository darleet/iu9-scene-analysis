from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from scene_analysis.utils import ensure_float32_array


def load_mask(path: Path) -> np.ndarray:
    """Загрузить ground truth mask как 2D uint8"""
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Failed to read mask file: {path}")
    return mask.astype(np.uint8, copy=False)


def load_prediction(path: Path) -> np.ndarray:
    """Загрузить prediction heatmap из .npy или .png"""
    suffix = path.suffix.lower()
    if suffix == ".npy":
        prediction = np.load(path)
    elif suffix == ".png":
        prediction = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if prediction is None:
            raise FileNotFoundError(f"Failed to read prediction PNG file: {path}")
        prediction = prediction.astype(np.float32) / 255.0
    else:
        raise ValueError(f"Unsupported prediction format: {path.suffix}")

    prediction_array = ensure_float32_array(np.asarray(prediction))
    if prediction_array.ndim != 2:
        raise ValueError(f"Prediction map must be 2D, got shape {prediction_array.shape} for {path}")
    return prediction_array


def load_image_if_exists(path: Path | None) -> np.ndarray | None:
    """Загрузить RGB/BGR изображение, если путь существует"""
    if path is None:
        return None
    if not path.exists():
        return None
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image file: {path}")
    return image
