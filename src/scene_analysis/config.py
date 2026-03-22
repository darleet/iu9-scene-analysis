from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class BaseConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class RoiConfig(BaseConfigModel):
    """Настройки области интереса для препроцессинга"""

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


class PercentileClipConfig(BaseConfigModel):
    """Параметры обрезки процентилей для визуализации карты глубины"""

    min: float = 2.0
    max: float = 98.0

    @field_validator("min", "max")
    @classmethod
    def validate_percentile_range(cls, value: float) -> float:
        if not 0.0 <= value <= 100.0:
            raise ValueError("Depth clip percentiles must be in the range [0, 100]")
        return float(value)

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


class OutputConfig(BaseConfigModel):
    """Настройки сохранения артефактов"""

    output_dir: Path
    save_original_frames: bool = True
    save_preprocessed_frames: bool = True
    save_overlay_frames: bool = True
    save_jsonl: bool = True


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
