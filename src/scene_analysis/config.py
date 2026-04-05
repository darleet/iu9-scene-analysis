from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


def _validate_percentile_value(value: float, field_name: str) -> float:
    normalized = float(value)
    if not 0.0 <= normalized <= 100.0:
        raise ValueError(f"{field_name} must be in the range [0, 100]")
    return normalized


def _validate_ratio_value(value: float, field_name: str) -> float:
    normalized = float(value)
    if not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{field_name} must be in the range [0, 1]")
    return normalized


def _validate_probability_value(value: float, field_name: str) -> float:
    return _validate_ratio_value(value, field_name)


def _validate_odd_kernel(value: int, field_name: str) -> int:
    normalized = int(value)
    if normalized <= 0:
        raise ValueError(f"{field_name} must be greater than 0")
    if normalized % 2 == 0:
        raise ValueError(f"{field_name} must be an odd positive integer")
    return normalized


def _normalize_path(value: Path | str | None) -> Path | None:
    if value is None:
        return None
    return Path(value).expanduser()


def _normalize_extension(value: str) -> str:
    normalized = value.strip().lower()
    if not normalized:
        raise ValueError("File extension must not be empty")
    if not normalized.startswith("."):
        normalized = f".{normalized}"
    return normalized


class BaseConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class RoiConfig(BaseConfigModel):
    """Настройки области интереса для препроцессинга."""

    enabled: bool = False
    x: int = 0
    y: int = 0
    width: int = 640
    height: int = 360

    @field_validator("x", "y")
    @classmethod
    def validate_non_negative_origin(cls, value: int) -> int:
        if value < 0:
            raise ValueError("ROI origin values must be non-negative.")
        return value

    @field_validator("width", "height")
    @classmethod
    def validate_positive_size(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("ROI width and height must be positive.")
        return value


class PreprocessingConfig(BaseConfigModel):
    """Параметры препроцессинга"""

    resize_width: int = Field(default=640, gt=0)
    resize_height: int = Field(default=360, gt=0)
    normalize_to_float: bool = False
    roi: RoiConfig = Field(default_factory=RoiConfig)


class InputConfig(BaseConfigModel):
    """Настройки входного источника и сэмплирования"""

    source_path: Path
    max_frames: int | None = Field(default=100, gt=0)
    sample_every_n: int = Field(default=1, gt=0)

    @field_validator("source_path", mode="before")
    @classmethod
    def normalize_source_path(cls, value: Path | str) -> Path:
        return Path(value).expanduser()


class PercentileClipConfig(BaseConfigModel):
    """Параметры обрезки процентилей для визуализации карты глубины"""

    min: float = 2.0
    max: float = 98.0

    @field_validator("min", "max")
    @classmethod
    def validate_percentile_range(cls, value: float, info: Any) -> float:
        return _validate_percentile_value(value, f"Depth clip percentile '{info.field_name}'")

    @model_validator(mode="after")
    def validate_percentile_order(self) -> PercentileClipConfig:
        if self.min >= self.max:
            raise ValueError("Depth clip percentile minimum must be smaller than maximum")
        return self


class DepthConfig(BaseConfigModel):
    """Настройки оценки глубины"""

    enabled: bool = True
    provider: str = "depth_anything_v2"
    model: str = "depth-anything/Depth-Anything-V2-Small-hf"
    device: str = "auto"
    cache_dir: str | None = None
    use_fp16: bool = False
    compile_model: bool = False
    save_raw_depth_npy: bool = True
    save_depth_colormap: bool = True
    normalize_depth_for_viz: bool = True
    clip_percentiles: PercentileClipConfig = Field(default_factory=PercentileClipConfig)

    @field_validator("provider")
    @classmethod
    def validate_provider(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized != "depth_anything_v2":
            raise ValueError("Only provider 'depth_anything_v2' is supported at this stage")
        return normalized

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Depth model must not be empty")
        return normalized

    @field_validator("device")
    @classmethod
    def validate_device(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized in {"auto", "cpu", "cuda", "mps"} or normalized.startswith("cuda:"):
            return normalized
        raise ValueError(
            "Depth device must be one of {'auto', 'cpu', 'cuda', 'mps'} or start with 'cuda:'"
        )

    @field_validator("cache_dir")
    @classmethod
    def validate_cache_dir(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


class NearScoreConfig(BaseConfigModel):
    """Параметры преобразования depth map в near-score"""

    use_relative_depth: bool = True
    invert_depth: bool = False
    clip_min_percentile: float = 2.0
    clip_max_percentile: float = 98.0
    gamma: float = Field(default=1.0, gt=0.0)

    @field_validator("clip_min_percentile", "clip_max_percentile")
    @classmethod
    def validate_percentile(cls, value: float, info: Any) -> float:
        return _validate_percentile_value(value, f"Near-score percentile '{info.field_name}'")

    @model_validator(mode="after")
    def validate_percentile_order(self) -> NearScoreConfig:
        if self.clip_min_percentile >= self.clip_max_percentile:
            raise ValueError("Near-score clip_min_percentile must be smaller than clip_max_percentile")
        return self


class HeatmapRoiConfig(BaseConfigModel):
    """Параметры ROI для obstacle heatmap"""

    enabled: bool = True
    top_ignore_ratio: float = 0.22
    left_ignore_ratio: float = 0.0
    right_ignore_ratio: float = 0.0
    bottom_keep_ratio: float = 1.0

    @field_validator(
        "top_ignore_ratio",
        "left_ignore_ratio",
        "right_ignore_ratio",
        "bottom_keep_ratio",
    )
    @classmethod
    def validate_ratio(cls, value: float, info: Any) -> float:
        return _validate_ratio_value(value, f"Heatmap ROI '{info.field_name}'")


class RoadSuppressionConfig(BaseConfigModel):
    """Параметры подавления дороги"""

    enabled: bool = True
    mode: str = "row_baseline"
    bottom_strip_ratio: float = 0.30
    row_smooth_kernel: int = 11
    baseline_quantile: float = 0.60
    suppression_strength: float = 1.0
    min_row_activation: float = 0.03
    preserve_vertical_edges: bool = True
    edge_weight: float = 0.35

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized != "row_baseline":
            raise ValueError("Only road_suppression mode 'row_baseline' is supported")
        return normalized

    @field_validator(
        "bottom_strip_ratio",
        "baseline_quantile",
        "min_row_activation",
        "suppression_strength",
        "edge_weight",
    )
    @classmethod
    def validate_unit_range(cls, value: float, info: Any) -> float:
        return _validate_probability_value(value, f"Road suppression '{info.field_name}'")

    @field_validator("row_smooth_kernel")
    @classmethod
    def validate_row_smooth_kernel(cls, value: int) -> int:
        return _validate_odd_kernel(value, "road_suppression.row_smooth_kernel")


class HeatmapPostprocessConfig(BaseConfigModel):
    """Параметры постобработки obstacle heatmap"""

    blur_kernel_size: int = 5
    morph_kernel_size: int = 5
    min_activation: float = 0.05
    normalize_output: bool = True

    @field_validator("blur_kernel_size")
    @classmethod
    def validate_blur_kernel(cls, value: int) -> int:
        return _validate_odd_kernel(value, "postprocess.blur_kernel_size")

    @field_validator("morph_kernel_size")
    @classmethod
    def validate_morph_kernel(cls, value: int) -> int:
        return _validate_odd_kernel(value, "postprocess.morph_kernel_size")

    @field_validator("min_activation")
    @classmethod
    def validate_min_activation(cls, value: float) -> float:
        return _validate_probability_value(value, "postprocess.min_activation")


class HeatmapVisualizationConfig(BaseConfigModel):
    """Параметры сохранения obstacle heatmap артефактов"""

    save_heatmap_npy: bool = True
    save_heatmap_png: bool = True
    save_overlay_png: bool = True
    colormap: str = "inferno"

    @field_validator("colormap")
    @classmethod
    def validate_colormap(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not normalized:
            raise ValueError("Visualization colormap must not be empty")
        return normalized


class ObstacleHeatmapConfig(BaseConfigModel):
    """Полная конфигурация построения obstacle heatmap"""

    enabled: bool = True
    near_score: NearScoreConfig = Field(default_factory=NearScoreConfig)
    roi: HeatmapRoiConfig = Field(default_factory=HeatmapRoiConfig)
    road_suppression: RoadSuppressionConfig = Field(default_factory=RoadSuppressionConfig)
    postprocess: HeatmapPostprocessConfig = Field(default_factory=HeatmapPostprocessConfig)
    visualization: HeatmapVisualizationConfig = Field(default_factory=HeatmapVisualizationConfig)


class EvaluationDatasetConfig(BaseConfigModel):
    """Настройки локального layout датасета для evaluation."""

    name: str = "road_obstacle_21"
    root_dir: Path = Path("data/datasets/road_obstacle_21")
    images_dir: Path = Path("images")
    masks_dir: Path = Path("masks")
    predictions_dir: Path = Path("predictions")
    split_file: Path | None = None
    file_extension_images: str = ".png"
    file_extension_masks: str = ".png"
    file_extension_predictions: str = ".npy"

    @field_validator("name")
    @classmethod
    def validate_dataset_name(cls, value: str) -> str:
        normalized = value.strip().lower()
        if not normalized:
            raise ValueError("Evaluation dataset name must not be empty")
        return normalized

    @field_validator("root_dir", "images_dir", "masks_dir", "predictions_dir", "split_file", mode="before")
    @classmethod
    def normalize_paths(cls, value: Path | str | None) -> Path | None:
        return _normalize_path(value)

    @field_validator("file_extension_images", "file_extension_masks", "file_extension_predictions")
    @classmethod
    def normalize_extensions(cls, value: str) -> str:
        return _normalize_extension(value)


class EvaluationLabelsConfig(BaseConfigModel):
    """Настройки кодов разметки obstacle/background/ignore"""

    obstacle_values: list[int] = Field(default_factory=lambda: [1])
    background_values: list[int] = Field(default_factory=lambda: [0])
    ignore_values: list[int] = Field(default_factory=lambda: [255])

    @field_validator("obstacle_values", "background_values", "ignore_values")
    @classmethod
    def validate_non_empty_values(cls, value: list[int], info: Any) -> list[int]:
        normalized = [int(item) for item in value]
        if not normalized:
            raise ValueError(f"Evaluation labels '{info.field_name}' must not be empty")
        return normalized


class EvaluationPredictionConfig(BaseConfigModel):
    """Настройки подготовки prediction heatmap перед подсчетом метрики"""

    resize_to_gt: bool = True
    clip_to_unit_range: bool = True
    allow_png_heatmaps: bool = False


class EvaluationMetricsConfig(BaseConfigModel):
    """Настройки метрик evaluation"""

    average_precision: bool = True


class EvaluationOutputsConfig(BaseConfigModel):
    """Настройки сохранения evaluation артефактов"""

    output_dir: Path = Path("data/artifacts/eval_run_001")
    save_pr_curve_png: bool = True
    save_per_sample_csv: bool = True
    save_summary_json: bool = True
    save_hard_examples: bool = True
    hard_examples_top_k: int = Field(default=20, gt=0)

    @field_validator("output_dir", mode="before")
    @classmethod
    def normalize_output_dir(cls, value: Path | str) -> Path:
        return Path(value).expanduser()


class EvaluationConfig(BaseConfigModel):
    """Полная конфигурация evaluation obstacle heatmap"""

    enabled: bool = True
    dataset: EvaluationDatasetConfig = Field(default_factory=EvaluationDatasetConfig)
    labels: EvaluationLabelsConfig = Field(default_factory=EvaluationLabelsConfig)
    prediction: EvaluationPredictionConfig = Field(default_factory=EvaluationPredictionConfig)
    metrics: EvaluationMetricsConfig = Field(default_factory=EvaluationMetricsConfig)
    outputs: EvaluationOutputsConfig = Field(default_factory=EvaluationOutputsConfig)

    @model_validator(mode="after")
    def validate_prediction_extension(self) -> EvaluationConfig:
        prediction_extension = self.dataset.file_extension_predictions
        if prediction_extension == ".npy":
            return self
        if prediction_extension == ".png" and self.prediction.allow_png_heatmaps:
            return self
        raise ValueError(
            "Evaluation dataset.file_extension_predictions must be '.npy' or '.png' with "
            "prediction.allow_png_heatmaps=true"
        )


class OutputConfig(BaseConfigModel):
    """Настройки сохранения артефактов"""

    output_dir: Path
    save_original_frames: bool = True
    save_preprocessed_frames: bool = True
    save_overlay_frames: bool = True
    save_jsonl: bool = True

    @field_validator("output_dir", mode="before")
    @classmethod
    def normalize_output_dir(cls, value: Path | str) -> Path:
        return Path(value).expanduser()


class RuntimeConfig(BaseConfigModel):
    """Настройки выполнения и логирования"""

    log_level: str = "INFO"

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        normalized = value.upper()
        allowed_levels = {
            "TRACE",
            "DEBUG",
            "INFO",
            "SUCCESS",
            "WARNING",
            "ERROR",
            "CRITICAL",
        }
        if normalized not in allowed_levels:
            raise ValueError(f"Unsupported log level: {value}")
        return normalized


class AppMetaConfig(BaseConfigModel):
    """Верхнеуровневые метаданные приложения"""

    name: str = "scene-analysis"
    debug: bool = True

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Application name must not be empty")
        return value


class AppConfig(BaseConfigModel):
    app: AppMetaConfig
    input: InputConfig
    preprocessing: PreprocessingConfig
    depth: DepthConfig
    obstacle_heatmap: ObstacleHeatmapConfig = Field(default_factory=ObstacleHeatmapConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    output: OutputConfig
    runtime: RuntimeConfig


def load_config(path: Path) -> AppConfig:
    config_path = path.expanduser()
    if not config_path.exists() or not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        raw_data: Any = yaml.safe_load(file) or {}

    if not isinstance(raw_data, dict):
        raise ValueError("Configuration file must contain a YAML mapping")

    return AppConfig.model_validate(raw_data)
