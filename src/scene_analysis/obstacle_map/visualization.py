from __future__ import annotations

import cv2
import numpy as np

from scene_analysis.utils import ensure_float32_array, ensure_uint8_image


_COLORMAP_MAP = {
    "inferno": getattr(cv2, "COLORMAP_INFERNO", cv2.COLORMAP_JET),
    "magma": getattr(cv2, "COLORMAP_MAGMA", cv2.COLORMAP_JET),
    "turbo": getattr(cv2, "COLORMAP_TURBO", cv2.COLORMAP_JET),
    "jet": cv2.COLORMAP_JET,
}


def _prepare_heatmap(heatmap: np.ndarray) -> np.ndarray:
    if not isinstance(heatmap, np.ndarray) or heatmap.size == 0:
        raise ValueError("Heatmap must be a non-empty numpy array")
    if heatmap.ndim != 2:
        raise ValueError("Heatmap must be a 2D array")

    prepared = ensure_float32_array(heatmap)
    prepared = np.nan_to_num(prepared, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32, copy=False)
    return np.clip(prepared, 0.0, 1.0)


def _resolve_colormap(colormap: str) -> int:
    return _COLORMAP_MAP.get(colormap.strip().lower(), _COLORMAP_MAP["inferno"])


def heatmap_to_bgr(
    heatmap: np.ndarray,
    colormap: str = "inferno",
) -> np.ndarray:
    """Преобразовать obstacle heatmap [0, 1] в BGR uint8"""
    prepared = _prepare_heatmap(heatmap)
    grayscale = np.clip(prepared * 255.0, 0.0, 255.0).astype(np.uint8)
    return cv2.applyColorMap(grayscale, _resolve_colormap(colormap))


def overlay_heatmap_on_image(
    image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: str = "inferno",
) -> np.ndarray:
    """Наложить obstacle heatmap на изображение"""
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("Alpha must be in the range [0, 1]")

    base_image = ensure_uint8_image(image)
    if base_image.ndim == 2:
        base_image = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
    elif base_image.ndim == 3 and base_image.shape[2] == 1:
        base_image = cv2.cvtColor(base_image[:, :, 0], cv2.COLOR_GRAY2BGR)
    elif base_image.ndim != 3 or base_image.shape[2] != 3:
        raise ValueError("Overlay image must have shape HxW or HxWx3")

    prepared_heatmap = _prepare_heatmap(heatmap)
    if prepared_heatmap.shape != base_image.shape[:2]:
        prepared_heatmap = cv2.resize(
            prepared_heatmap,
            (base_image.shape[1], base_image.shape[0]),
            interpolation=cv2.INTER_LINEAR,
        )
        prepared_heatmap = np.clip(prepared_heatmap, 0.0, 1.0).astype(np.float32, copy=False)

    colorized = heatmap_to_bgr(prepared_heatmap, colormap=colormap).astype(np.float32)
    alpha_mask = (prepared_heatmap[..., None] * float(alpha)).astype(np.float32)
    blended = base_image.astype(np.float32) * (1.0 - alpha_mask) + colorized * alpha_mask
    return np.clip(blended, 0.0, 255.0).astype(np.uint8)
