from __future__ import annotations

from pathlib import Path

from scene_analysis.app.cli import _apply_image_prediction_overrides
from scene_analysis.config import load_config


def test_image_prediction_overrides_disable_preprocessing_roi_by_default() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"
    config = load_config(config_path)

    assert config.preprocessing.roi.enabled is True

    updated = _apply_image_prediction_overrides(
        config=config.model_copy(deep=True),
        keep_preprocessing_roi=False,
    )

    assert updated.preprocessing.roi.enabled is False


def test_image_prediction_overrides_can_keep_preprocessing_roi() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"
    config = load_config(config_path)

    updated = _apply_image_prediction_overrides(
        config=config.model_copy(deep=True),
        keep_preprocessing_roi=True,
    )

    assert updated.preprocessing.roi.enabled is True
