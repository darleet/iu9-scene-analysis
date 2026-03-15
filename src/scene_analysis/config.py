from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator


class BaseConfigModel(BaseModel):
    """Базовая модель для строгой валидации конфигурации"""

    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class RoiConfig(BaseConfigModel):
    """Настройки области интереса для препроцессинга"""

    enabled: bool = False
    x: int = 0
    y: int = 0
    width: int = 640
    height: int = 360

    @classmethod
    @field_validator("x", "y")
    def validate_non_negative_origin(cls, value: int) -> int:
        if value < 0:
            raise ValueError("ROI origin values must be non-negative.")
        return value

    @classmethod
    @field_validator("width", "height")
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

    @classmethod
    @field_validator("log_level")
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

    @classmethod
    @field_validator("name")
    def validate_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Application name must not be empty.")
        return value


class AppConfig(BaseConfigModel):
    app: AppMetaConfig
    input: InputConfig
    preprocessing: PreprocessingConfig
    output: OutputConfig
    runtime: RuntimeConfig


def load_config(path: Path) -> AppConfig:
    config_path = path.expanduser()
    if not config_path.exists() or not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as file:
        raw_data: Any = yaml.safe_load(file) or {}

    if not isinstance(raw_data, dict):
        raise ValueError("Configuration file must contain a YAML mapping.")

    return AppConfig.model_validate(raw_data)
