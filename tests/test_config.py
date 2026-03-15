from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from scene_analysis.config import load_config


def test_load_config_from_base_yaml() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"
    config = load_config(config_path)

    assert config.app.name == "scene-analysis"
    assert config.preprocessing.resize_width == 640
    assert config.preprocessing.resize_height == 360


def test_roi_validation_rejects_negative_origin(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid_roi.yaml"
    config_path.write_text(
        """
app:
  name: "scene-analysis"
  debug: true
input:
  source_path: "data/raw/sample.mp4"
  max_frames: 10
  sample_every_n: 1
preprocessing:
  resize_width: 640
  resize_height: 360
  normalize_to_float: false
  roi:
    enabled: true
    x: -1
    y: 0
    width: 100
    height: 100
output:
  output_dir: "data/artifacts/test"
  save_original_frames: true
  save_preprocessed_frames: true
  save_overlay_frames: true
  save_jsonl: true
runtime:
  log_level: "INFO"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_config(config_path)


def test_invalid_resize_dimensions_raise_error(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid_resize.yaml"
    config_path.write_text(
        """
app:
  name: "scene-analysis"
  debug: true
input:
  source_path: "data/raw/sample.mp4"
  max_frames: 10
  sample_every_n: 1
preprocessing:
  resize_width: 0
  resize_height: 360
  normalize_to_float: false
  roi:
    enabled: false
    x: 0
    y: 0
    width: 100
    height: 100
output:
  output_dir: "data/artifacts/test"
  save_original_frames: true
  save_preprocessed_frames: true
  save_overlay_frames: true
  save_jsonl: true
runtime:
  log_level: "INFO"
""".strip(),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_config(config_path)
