from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

STUDENT_NAMES: set[str] = {"student_s", "student_m", "student_q"}
BACKBONE_NAMES: set[str] = {
    "mobilenet_v3_small",
    "shufflenet_v2_x1_0",
    "efficientnet_b0",
}


class StudentBaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


def _normalize_path(value: Path | str | None) -> Path | None:
    if value is None:
        return None
    return Path(value).expanduser()


def _validate_device(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"auto", "cpu", "cuda", "mps"} or normalized.startswith("cuda:"):
        return normalized
    raise ValueError("device must be one of {'auto', 'cpu', 'cuda', 'mps'} or start with 'cuda:'")


def _normalize_extension(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("File suffix must not be empty")
    return normalized


class StudentExperimentConfig(StudentBaseConfig):
    name: str = "student_heatmap_distillation"
    seed: int = 1667

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("experiment.name must not be empty")
        return normalized


class StudentDatasetConfig(StudentBaseConfig):
    raw_root_dir: Path = Path("data/datasets/lost_and_found_raw")
    prepared_root_dir: Path = Path("data/datasets/lost_and_found_prepared")
    images_dir: Path = Path(".")
    masks_dir: Path = Path(".")
    image_suffix: str = "_leftImg8bit.png"
    mask_suffix: str = "_gtCoarse_labelIds.png"
    teacher_suffix: str = ".npy"
    train_ratio: float = Field(default=0.8, gt=0.0, lt=1.0)
    split_seed: int = 1667
    raw_train_splits: list[str] = Field(default_factory=lambda: ["train"])
    raw_val_splits: list[str] = Field(default_factory=lambda: ["val", "test"])
    obstacle_value: int = 1
    background_value: int = 0
    ignore_value: int = 255
    mask_obstacle_values: list[int] = Field(default_factory=list)
    mask_background_values: list[int] = Field(default_factory=lambda: [1])
    mask_ignore_values: list[int] = Field(default_factory=lambda: [0, 255])
    mask_unmapped_value: int | None = 1
    use_resized_cache: bool = True
    overwrite_resized_cache: bool = False

    @field_validator("raw_root_dir", "prepared_root_dir", "images_dir", "masks_dir", mode="before")
    @classmethod
    def normalize_paths(cls, value: Path | str) -> Path:
        path = _normalize_path(value)
        assert path is not None
        return path

    @field_validator("image_suffix", "mask_suffix", "teacher_suffix")
    @classmethod
    def normalize_suffixes(cls, value: str) -> str:
        return _normalize_extension(value)

    @field_validator("mask_obstacle_values", "mask_background_values", "mask_ignore_values")
    @classmethod
    def normalize_mask_source_values(cls, value: list[int]) -> list[int]:
        return [int(item) for item in value]

    @field_validator("raw_train_splits", "raw_val_splits")
    @classmethod
    def normalize_split_names(cls, value: list[str]) -> list[str]:
        return [item.strip().lower() for item in value if item.strip()]

    @model_validator(mode="after")
    def validate_mask_values(self) -> StudentDatasetConfig:
        values = {self.obstacle_value, self.background_value, self.ignore_value}
        if len(values) != 3:
            raise ValueError("obstacle_value, background_value and ignore_value must be different")
        return self


class StudentTeacherConfig(StudentBaseConfig):
    config_path: Path = Path("configs/base.yaml")
    overwrite_teacher_heatmaps: bool = False

    @field_validator("config_path", mode="before")
    @classmethod
    def normalize_config_path(cls, value: Path | str) -> Path:
        path = _normalize_path(value)
        assert path is not None
        return path


class StudentInputConfig(StudentBaseConfig):
    height: int = Field(default=256, gt=0)
    width: int = Field(default=512, gt=0)
    normalize_mean: list[float] = Field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalize_std: list[float] = Field(default_factory=lambda: [0.229, 0.224, 0.225])

    @field_validator("normalize_mean", "normalize_std")
    @classmethod
    def validate_normalization(cls, value: list[float]) -> list[float]:
        if len(value) != 3:
            raise ValueError("normalize_mean and normalize_std must contain 3 values")
        return [float(item) for item in value]


class StudentAugmentationConfig(StudentBaseConfig):
    enabled: bool = True
    horizontal_flip_p: float = Field(default=0.5, ge=0.0, le=1.0)
    brightness_contrast_p: float = Field(default=0.3, ge=0.0, le=1.0)
    blur_p: float = Field(default=0.1, ge=0.0, le=1.0)
    noise_p: float = Field(default=0.1, ge=0.0, le=1.0)


class StudentModelConfig(StudentBaseConfig):
    backbone: str
    pretrained_backbone: bool = True
    decoder_channels: list[int] = Field(default_factory=lambda: [128, 64, 32])
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)
    use_roi_head_in_heatmap: bool = False

    @field_validator("backbone")
    @classmethod
    def validate_backbone(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in BACKBONE_NAMES:
            raise ValueError(f"Unsupported student backbone: {value}")
        return normalized

    @field_validator("decoder_channels")
    @classmethod
    def validate_decoder_channels(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("decoder_channels must not be empty")
        normalized = [int(item) for item in value]
        if any(item <= 0 for item in normalized):
            raise ValueError("decoder_channels must contain positive integers")
        return normalized


class StudentModelsConfig(StudentBaseConfig):
    train_students: list[str] = Field(default_factory=lambda: ["student_s", "student_m", "student_q"])
    student_s: StudentModelConfig
    student_m: StudentModelConfig
    student_q: StudentModelConfig

    @field_validator("train_students")
    @classmethod
    def validate_train_students(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("models.train_students must not be empty")
        normalized = [item.strip() for item in value]
        unknown = [item for item in normalized if item not in STUDENT_NAMES]
        if unknown:
            raise ValueError(f"Unsupported student name(s): {unknown}")
        return normalized

    def get(self, student_name: str) -> StudentModelConfig:
        if student_name not in STUDENT_NAMES:
            raise ValueError(f"Unsupported student name: {student_name}")
        return getattr(self, student_name)


class StudentLossConfig(StudentBaseConfig):
    bce_weight: float = Field(default=1.0, ge=0.0)
    dice_weight: float = Field(default=0.5, ge=0.0)
    roi_bce_weight: float = Field(default=0.4, ge=0.0)
    distill_mse_weight: float = Field(default=0.2, ge=0.0)
    offroad_weight: float = Field(default=0.1, ge=0.0)
    positive_class_weight: float = Field(default=6.0, gt=0.0)
    eps: float = Field(default=0.000001, gt=0.0)


class StudentOptimizerConfig(StudentBaseConfig):
    name: str = "adamw"
    lr: float = Field(default=0.0003, gt=0.0)
    weight_decay: float = Field(default=0.0001, ge=0.0)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized != "adamw":
            raise ValueError("Only AdamW optimizer is supported")
        return normalized


class StudentSchedulerConfig(StudentBaseConfig):
    name: str = "cosine"
    min_lr: float = Field(default=0.000001, ge=0.0)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized != "cosine":
            raise ValueError("Only cosine scheduler is supported")
        return normalized


class StudentTrainingRuntimeConfig(StudentBaseConfig):
    device: str = "auto"
    batch_size: int = Field(default=8, gt=0)
    num_workers: int = Field(default=4, ge=0)
    epochs: int = Field(default=40, gt=0)
    use_amp: bool = True
    grad_clip_norm: float | None = Field(default=1.0, gt=0.0)
    log_every_n_steps: int = Field(default=20, gt=0)
    save_every_n_epochs: int = Field(default=5, gt=0)
    max_train_batches: int | None = Field(default=None, gt=0)
    max_val_batches: int | None = Field(default=None, gt=0)

    @field_validator("device")
    @classmethod
    def validate_device(cls, value: str) -> str:
        return _validate_device(value)


class StudentValidationConfig(StudentBaseConfig):
    compute_average_precision: bool = True
    save_best_by: str = "val_ap"
    threshold_preview: float = Field(default=0.5, ge=0.0, le=1.0)
    save_visual_examples: bool = True
    num_visual_examples: int = Field(default=12, gt=0)
    save_visual_every_n_epochs: int = Field(default=5, gt=0)

    @field_validator("save_best_by")
    @classmethod
    def validate_save_best_by(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"val_ap", "val_loss"}:
            raise ValueError("validation.save_best_by must be 'val_ap' or 'val_loss'")
        return normalized


class StudentOutputConfig(StudentBaseConfig):
    root_dir: Path = Path("data/artifacts/student_runs")
    save_checkpoints: bool = True
    save_history_csv: bool = True
    save_summary_json: bool = True
    save_pr_curve_png: bool = True
    save_visual_previews: bool = True

    @field_validator("root_dir", mode="before")
    @classmethod
    def normalize_root_dir(cls, value: Path | str) -> Path:
        path = _normalize_path(value)
        assert path is not None
        return path


class StudentTrainConfig(StudentBaseConfig):
    experiment: StudentExperimentConfig
    dataset: StudentDatasetConfig
    teacher: StudentTeacherConfig
    input: StudentInputConfig
    augmentations: StudentAugmentationConfig
    models: StudentModelsConfig
    loss: StudentLossConfig
    optimizer: StudentOptimizerConfig
    scheduler: StudentSchedulerConfig
    training: StudentTrainingRuntimeConfig
    validation: StudentValidationConfig
    outputs: StudentOutputConfig


class StudentSmokeConfig(StudentTrainConfig):
    """Конфиг для smoke-тестирования"""


class StudentInferenceRuntimeConfig(StudentBaseConfig):
    device: str = "auto"
    student: str = "student_s"
    checkpoint_path: Path = Path("data/artifacts/student_runs/student_s/checkpoints/best.pt")

    @field_validator("device")
    @classmethod
    def validate_device(cls, value: str) -> str:
        return _validate_device(value)

    @field_validator("student")
    @classmethod
    def validate_student(cls, value: str) -> str:
        normalized = value.strip()
        if normalized not in STUDENT_NAMES:
            raise ValueError(f"Unsupported student name: {value}")
        return normalized

    @field_validator("checkpoint_path", mode="before")
    @classmethod
    def normalize_checkpoint_path(cls, value: Path | str) -> Path:
        path = _normalize_path(value)
        assert path is not None
        return path


class StudentVideoFolderConfig(StudentBaseConfig):
    input_dir: Path = Path("data/input_videos")
    output_dir: Path = Path("data/artifacts/student_video_folder")
    video_extensions: list[str] = Field(default_factory=lambda: [".mp4", ".avi", ".mov", ".mkv"])
    max_frames: int | None = Field(default=None, gt=0)
    sample_every_n: int = Field(default=1, gt=0)

    @field_validator("input_dir", "output_dir", mode="before")
    @classmethod
    def normalize_paths(cls, value: Path | str) -> Path:
        path = _normalize_path(value)
        assert path is not None
        return path

    @field_validator("video_extensions")
    @classmethod
    def normalize_extensions(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("video_extensions must not be empty")
        return [item if item.startswith(".") else f".{item}" for item in value]


class StudentCameraConfig(StudentBaseConfig):
    camera_index: int = Field(default=0, ge=0)
    width: int = Field(default=640, gt=0)
    height: int = Field(default=480, gt=0)
    fps: int = Field(default=15, gt=0)
    display: bool = True
    save_video: bool = True
    output_dir: Path = Path("data/artifacts/student_camera")

    @field_validator("output_dir", mode="before")
    @classmethod
    def normalize_output_dir(cls, value: Path | str) -> Path:
        path = _normalize_path(value)
        assert path is not None
        return path


class StudentVisualizationConfig(StudentBaseConfig):
    colormap: str = "inferno"
    alpha: float = Field(default=0.45, ge=0.0, le=1.0)
    show_binary_mask: bool = True
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    draw_stats: bool = True
    draw_model_name: bool = True


class StudentInferenceOutputConfig(StudentBaseConfig):
    save_overlay_video: bool = True
    save_frames: bool = False
    save_heatmap_npy: bool = False
    save_heatmap_png: bool = True
    save_jsonl: bool = True


class StudentInferenceConfig(StudentBaseConfig):
    inference: StudentInferenceRuntimeConfig
    input: StudentInputConfig
    video_folder: StudentVideoFolderConfig
    camera: StudentCameraConfig
    visualization: StudentVisualizationConfig
    output: StudentInferenceOutputConfig


def _load_yaml_mapping(path: Path) -> dict[str, Any]:
    config_path = path.expanduser()
    if not config_path.exists() or not config_path.is_file():
        raise FileNotFoundError(f"Student config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as file:
        raw_data: Any = yaml.safe_load(file) or {}
    if not isinstance(raw_data, dict):
        raise ValueError("Student configuration file must contain a YAML mapping")
    return raw_data


def load_student_train_config(path: Path) -> StudentTrainConfig:
    return StudentTrainConfig.model_validate(_load_yaml_mapping(path))


def load_student_inference_config(path: Path) -> StudentInferenceConfig:
    return StudentInferenceConfig.model_validate(_load_yaml_mapping(path))
