from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from scene_analysis.config import load_config


def _config_text(
    *,
    extra_depth_block: str = "",
    extra_obstacle_block: str = "",
    extra_evaluation_block: str = "",
) -> str:
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
    obstacle_block = """
obstacle_heatmap:
  enabled: true
  near_score:
    use_relative_depth: true
    invert_depth: false
    clip_min_percentile: 2.0
    clip_max_percentile: 98.0
    gamma: 1.0
  roi:
    enabled: true
    top_ignore_ratio: 0.22
    left_ignore_ratio: 0.0
    right_ignore_ratio: 0.0
    bottom_keep_ratio: 1.0
  road_suppression:
    enabled: true
    mode: "row_baseline"
    bottom_strip_ratio: 0.30
    row_smooth_kernel: 11
    baseline_quantile: 0.60
    suppression_strength: 1.0
    min_row_activation: 0.03
    preserve_vertical_edges: true
    edge_weight: 0.35
  postprocess:
    blur_kernel_size: 5
    morph_kernel_size: 5
    min_activation: 0.05
    normalize_output: true
  visualization:
    save_heatmap_npy: true
    save_heatmap_png: true
    save_overlay_png: true
    colormap: "inferno"
""".strip()
    evaluation_block = """
evaluation:
  enabled: true
  dataset:
    name: "road_obstacle_21"
    root_dir: "data/datasets/road_obstacle_21"
    images_dir: "images"
    masks_dir: "masks"
    predictions_dir: "predictions"
    split_file: null
    file_extension_images: ".png"
    file_extension_masks: ".png"
    file_extension_predictions: ".npy"
  labels:
    obstacle_values: [1]
    background_values: [0]
    ignore_values: [255]
  prediction:
    resize_to_gt: true
    clip_to_unit_range: true
    allow_png_heatmaps: false
  metrics:
    average_precision: true
  outputs:
    output_dir: "data/artifacts/eval_run_001"
    save_pr_curve_png: true
    save_per_sample_csv: true
    save_summary_json: true
    save_hard_examples: true
    hard_examples_top_k: 20
""".strip()
    if extra_depth_block:
        depth_block = extra_depth_block.strip()
    if extra_obstacle_block:
        obstacle_block = extra_obstacle_block.strip()
    if extra_evaluation_block:
        evaluation_block = extra_evaluation_block.strip()

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
{obstacle_block}
{evaluation_block}
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
    assert config.obstacle_heatmap.enabled is True
    assert config.obstacle_heatmap.road_suppression.mode == "row_baseline"
    assert config.obstacle_heatmap.visualization.colormap == "inferno"
    assert config.evaluation.enabled is True
    assert config.evaluation.dataset.name == "road_obstacle_21"
    assert config.evaluation.dataset.file_extension_predictions == ".npy"


def test_roi_validation_rejects_negative_origin(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid_roi.yaml"
    config_path.write_text(
        _config_text().replace("enabled: false", "enabled: true", 1).replace("x: 0", "x: -1", 1),
        encoding="utf-8",
    )

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


def test_invalid_obstacle_heatmap_kernel_raises_error(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid_heatmap_kernel.yaml"
    config_path.write_text(
        _config_text(
            extra_obstacle_block="""
obstacle_heatmap:
  enabled: true
  near_score:
    use_relative_depth: true
    invert_depth: false
    clip_min_percentile: 2.0
    clip_max_percentile: 98.0
    gamma: 1.0
  roi:
    enabled: true
    top_ignore_ratio: 0.22
    left_ignore_ratio: 0.0
    right_ignore_ratio: 0.0
    bottom_keep_ratio: 1.0
  road_suppression:
    enabled: true
    mode: "row_baseline"
    bottom_strip_ratio: 0.30
    row_smooth_kernel: 10
    baseline_quantile: 0.60
    suppression_strength: 1.0
    min_row_activation: 0.03
    preserve_vertical_edges: true
    edge_weight: 0.35
  postprocess:
    blur_kernel_size: 5
    morph_kernel_size: 5
    min_activation: 0.05
    normalize_output: true
  visualization:
    save_heatmap_npy: true
    save_heatmap_png: true
    save_overlay_png: true
    colormap: "inferno"
"""
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_config(config_path)


def test_invalid_evaluation_prediction_extension_raises_error(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid_evaluation_extension.yaml"
    config_path.write_text(
        _config_text(
            extra_evaluation_block="""
evaluation:
  enabled: true
  dataset:
    name: "road_obstacle_21"
    root_dir: "data/datasets/road_obstacle_21"
    images_dir: "images"
    masks_dir: "masks"
    predictions_dir: "predictions"
    split_file: null
    file_extension_images: ".png"
    file_extension_masks: ".png"
    file_extension_predictions: ".png"
  labels:
    obstacle_values: [1]
    background_values: [0]
    ignore_values: [255]
  prediction:
    resize_to_gt: true
    clip_to_unit_range: true
    allow_png_heatmaps: false
  metrics:
    average_precision: true
  outputs:
    output_dir: "data/artifacts/eval_run_001"
    save_pr_curve_png: true
    save_per_sample_csv: true
    save_summary_json: true
    save_hard_examples: true
    hard_examples_top_k: 20
"""
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_config(config_path)
