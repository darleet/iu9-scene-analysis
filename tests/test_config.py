from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from scene_analysis.config import load_config


def _config_text(*, extra_depth_block: str = "") -> str:
    depth_block = """
depth:
  enabled: true
  provider: "depth_anything_v2"
  model: "depth-anything/Depth-Anything-V2-Small-hf"
  device: "auto"
  cache_dir: null
  use_fp16: false
  compile_model: false
  save_raw_depth_npy: true
  save_depth_colormap: true
  normalize_depth_for_viz: true
  clip_percentiles:
    min: 2.0
    max: 98.0
""".strip()
    if extra_depth_block:
        depth_block = extra_depth_block.strip()

    return f"""
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
    enabled: false
    x: 0
    y: 0
    width: 100
    height: 100
{depth_block}
output:
  output_dir: "data/artifacts/test"
  save_original_frames: true
  save_preprocessed_frames: true
  save_overlay_frames: true
  save_jsonl: true
runtime:
  log_level: "INFO"
""".strip()


def test_load_config_from_base_yaml() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "base.yaml"
    config = load_config(config_path)

    assert config.app.name == "scene-analysis"
    assert config.preprocessing.resize_width == 640
    assert config.preprocessing.resize_height == 360
    assert config.depth.enabled is True
    assert config.depth.provider == "depth_anything_v2"
    assert config.depth.model == "depth-anything/Depth-Anything-V2-Small-hf"


def test_roi_validation_rejects_negative_origin(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid_roi.yaml"
    config_path.write_text(_config_text().replace("enabled: false", "enabled: true", 1).replace("x: 0", "x: -1", 1), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_config(config_path)


def test_invalid_resize_dimensions_raise_error(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid_resize.yaml"
    config_path.write_text(_config_text().replace("resize_width: 640", "resize_width: 0", 1), encoding="utf-8")

    with pytest.raises(ValidationError):
        load_config(config_path)


def test_invalid_depth_provider_raises_error(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid_provider.yaml"
    config_path.write_text(
        _config_text(
            extra_depth_block="""
depth:
  enabled: true
  provider: "dummy"
  model: "depth-anything/Depth-Anything-V2-Small-hf"
  device: "auto"
  cache_dir: null
  use_fp16: false
  compile_model: false
  save_raw_depth_npy: true
  save_depth_colormap: true
  normalize_depth_for_viz: true
  clip_percentiles:
    min: 2.0
    max: 98.0
"""
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_config(config_path)


def test_invalid_depth_percentiles_raise_error(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid_percentiles.yaml"
    config_path.write_text(
        _config_text(
            extra_depth_block="""
depth:
  enabled: true
  provider: "depth_anything_v2"
  model: "depth-anything/Depth-Anything-V2-Small-hf"
  device: "auto"
  cache_dir: null
  use_fp16: false
  compile_model: false
  save_raw_depth_npy: true
  save_depth_colormap: true
  normalize_depth_for_viz: true
  clip_percentiles:
    min: 98.0
    max: 2.0
"""
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_config(config_path)


def test_invalid_depth_device_raises_error(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid_device.yaml"
    config_path.write_text(
        _config_text(
            extra_depth_block="""
depth:
  enabled: true
  provider: "depth_anything_v2"
  model: "depth-anything/Depth-Anything-V2-Small-hf"
  device: "tpu"
  cache_dir: null
  use_fp16: false
  compile_model: false
  save_raw_depth_npy: true
  save_depth_colormap: true
  normalize_depth_for_viz: true
  clip_percentiles:
    min: 2.0
    max: 98.0
"""
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_config(config_path)


def test_depth_can_be_disabled_in_config(tmp_path: Path) -> None:
    config_path = tmp_path / "depth_disabled.yaml"
    config_path.write_text(
        _config_text(
            extra_depth_block="""
depth:
  enabled: false
  provider: "depth_anything_v2"
  model: "depth-anything/Depth-Anything-V2-Small-hf"
  device: "auto"
  cache_dir: null
  use_fp16: false
  compile_model: false
  save_raw_depth_npy: true
  save_depth_colormap: true
  normalize_depth_for_viz: true
  clip_percentiles:
    min: 2.0
    max: 98.0
"""
        ),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config.depth.enabled is False
