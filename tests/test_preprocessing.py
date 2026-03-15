from __future__ import annotations

import numpy as np

from scene_analysis.config import PreprocessingConfig, RoiConfig
from scene_analysis.preprocessing.frame_preprocessor import FramePreprocessor


def test_resize_changes_output_shape() -> None:
    image = np.zeros((4, 8, 3), dtype=np.uint8)
    config = PreprocessingConfig(
        resize_width=6,
        resize_height=3,
        normalize_to_float=False,
        roi=RoiConfig(),
    )
    preprocessor = FramePreprocessor(config)

    result = preprocessor.process(image)

    assert result.shape == (3, 6, 3)
    assert result.dtype == np.uint8


def test_roi_crop_uses_requested_region() -> None:
    image = np.arange(16, dtype=np.uint8).reshape(4, 4)
    config = PreprocessingConfig(
        resize_width=2,
        resize_height=2,
        normalize_to_float=False,
        roi=RoiConfig(enabled=True, x=1, y=1, width=2, height=2),
    )
    preprocessor = FramePreprocessor(config)

    result = preprocessor.process(image)

    expected = np.array([[5, 6], [9, 10]], dtype=np.uint8)
    assert np.array_equal(result, expected)


def test_normalization_returns_float_image() -> None:
    image = np.full((2, 2, 3), 255, dtype=np.uint8)
    config = PreprocessingConfig(
        resize_width=2,
        resize_height=2,
        normalize_to_float=True,
        roi=RoiConfig(),
    )
    preprocessor = FramePreprocessor(config)

    result = preprocessor.process(image)

    assert result.dtype == np.float32
    assert np.allclose(result, 1.0)
