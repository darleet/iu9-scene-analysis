from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pytest
import torch

from scene_analysis.student.config import StudentInferenceConfig, StudentTrainConfig
from scene_analysis.student.model import count_parameters
from scene_analysis.student.model_registry import create_student_model


def write_sample(root: Path, split: str, sample_id: str, *, positive: bool = True) -> None:
    split_root = root / split
    images_dir = split_root / "images"
    masks_dir = split_root / "masks"
    teacher_dir = split_root / "teacher_heatmaps"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    teacher_dir.mkdir(parents=True, exist_ok=True)

    image = np.zeros((48, 64, 3), dtype=np.uint8)
    image[:, :, 0] = 40
    image[:, :, 1] = np.linspace(0, 255, 64, dtype=np.uint8)[None, :]
    image[:, :, 2] = 160
    mask = np.zeros((48, 64), dtype=np.uint8)
    if positive:
        mask[16:28, 24:36] = 1
    mask[:6, :] = 255
    teacher = np.zeros((48, 64), dtype=np.float32)
    teacher[16:28, 24:36] = 0.9
    teacher += np.linspace(0.0, 0.2, 48, dtype=np.float32)[:, None]
    teacher = np.clip(teacher, 0.0, 1.0)

    cv2.imwrite(str(images_dir / f"{sample_id}.png"), image)
    cv2.imwrite(str(masks_dir / f"{sample_id}_labels_semantic.png"), mask)
    np.save(teacher_dir / f"{sample_id}.npy", teacher)


@pytest.fixture
def prepared_dataset(tmp_path: Path) -> Path:
    root = tmp_path / "prepared"
    for index in range(4):
        write_sample(root, "train", f"train_{index}", positive=True)
    for index in range(2):
        write_sample(root, "val", f"val_{index}", positive=True)
    return root


@pytest.fixture
def make_train_config(tmp_path: Path) -> Callable[[Path], StudentTrainConfig]:
    def _make(prepared_root: Path) -> StudentTrainConfig:
        return StudentTrainConfig.model_validate(
            {
                "experiment": {"name": "test_student", "seed": 7},
                "dataset": {
                    "raw_root_dir": str(tmp_path / "raw"),
                    "prepared_root_dir": str(prepared_root),
                    "images_dir": "images",
                    "masks_dir": "masks",
                    "image_suffix": ".png",
                    "mask_suffix": "_labels_semantic.png",
                    "teacher_suffix": ".npy",
                    "train_ratio": 0.8,
                    "split_seed": 1667,
                    "obstacle_value": 1,
                    "background_value": 0,
                    "ignore_value": 255,
                    "use_resized_cache": False,
                    "overwrite_resized_cache": False,
                },
                "teacher": {"config_path": "configs/base.yaml", "overwrite_teacher_heatmaps": False},
                "input": {
                    "height": 64,
                    "width": 96,
                    "normalize_mean": [0.485, 0.456, 0.406],
                    "normalize_std": [0.229, 0.224, 0.225],
                },
                "augmentations": {
                    "enabled": False,
                    "horizontal_flip_p": 0.0,
                    "brightness_contrast_p": 0.0,
                    "blur_p": 0.0,
                    "noise_p": 0.0,
                },
                "models": {
                    "train_students": ["student_s"],
                    "student_s": {
                        "backbone": "mobilenet_v3_small",
                        "pretrained_backbone": False,
                        "decoder_channels": [16],
                        "dropout": 0.0,
                    },
                    "student_m": {
                        "backbone": "shufflenet_v2_x1_0",
                        "pretrained_backbone": False,
                        "decoder_channels": [16],
                        "dropout": 0.0,
                    },
                    "student_q": {
                        "backbone": "efficientnet_b0",
                        "pretrained_backbone": False,
                        "decoder_channels": [16],
                        "dropout": 0.0,
                    },
                },
                "loss": {
                    "bce_weight": 1.0,
                    "dice_weight": 0.5,
                    "roi_bce_weight": 0.4,
                    "distill_mse_weight": 0.2,
                    "offroad_weight": 0.1,
                    "positive_class_weight": 2.0,
                    "eps": 0.000001,
                },
                "optimizer": {"name": "adamw", "lr": 0.001, "weight_decay": 0.0},
                "scheduler": {"name": "cosine", "min_lr": 0.000001},
                "training": {
                    "device": "cpu",
                    "batch_size": 2,
                    "num_workers": 0,
                    "epochs": 1,
                    "use_amp": False,
                    "grad_clip_norm": 1.0,
                    "log_every_n_steps": 1,
                    "save_every_n_epochs": 1,
                    "max_train_batches": 1,
                    "max_val_batches": 1,
                },
                "validation": {
                    "compute_average_precision": True,
                    "save_best_by": "val_ap",
                    "threshold_preview": 0.5,
                    "save_visual_examples": True,
                    "num_visual_examples": 2,
                    "save_visual_every_n_epochs": 1,
                },
                "outputs": {
                    "root_dir": str(tmp_path / "artifacts"),
                    "save_checkpoints": True,
                    "save_history_csv": True,
                    "save_summary_json": True,
                    "save_pr_curve_png": True,
                    "save_visual_previews": True,
                },
            }
        )

    return _make


@pytest.fixture
def train_config(prepared_dataset: Path, make_train_config: Callable[[Path], StudentTrainConfig]) -> StudentTrainConfig:
    return make_train_config(prepared_dataset)


@pytest.fixture
def student_checkpoint(tmp_path: Path, train_config: StudentTrainConfig) -> Path:
    model_config = train_config.models.student_s
    model = create_student_model("student_s", model_config)
    checkpoint_path = tmp_path / "student_s.pt"
    torch.save(
        {
            "student_name": "student_s",
            "backbone": model_config.backbone,
            "model_state_dict": model.state_dict(),
            "input_height": train_config.input.height,
            "input_width": train_config.input.width,
            "normalize_mean": train_config.input.normalize_mean,
            "normalize_std": train_config.input.normalize_std,
            "epoch": 0,
            "val_ap": 0.0,
            "parameter_count": count_parameters(model),
            "config": train_config.model_dump(mode="json"),
        },
        checkpoint_path,
    )
    return checkpoint_path


@pytest.fixture
def make_inference_config(tmp_path: Path) -> Callable[[Path], StudentInferenceConfig]:
    def _make(checkpoint_path: Path) -> StudentInferenceConfig:
        return StudentInferenceConfig.model_validate(
            {
                "inference": {"device": "cpu", "student": "student_s", "checkpoint_path": str(checkpoint_path)},
                "input": {
                    "height": 64,
                    "width": 96,
                    "normalize_mean": [0.485, 0.456, 0.406],
                    "normalize_std": [0.229, 0.224, 0.225],
                },
                "video_folder": {
                    "input_dir": str(tmp_path / "videos"),
                    "output_dir": str(tmp_path / "video_out"),
                    "video_extensions": [".avi"],
                    "max_frames": 2,
                    "sample_every_n": 1,
                },
                "camera": {
                    "camera_index": 0,
                    "width": 64,
                    "height": 48,
                    "fps": 5,
                    "display": False,
                    "save_video": False,
                    "output_dir": str(tmp_path / "camera_out"),
                },
                "visualization": {
                    "colormap": "inferno",
                    "alpha": 0.45,
                    "show_binary_mask": True,
                    "threshold": 0.5,
                    "draw_stats": True,
                    "draw_model_name": True,
                },
                "output": {
                    "save_overlay_video": True,
                    "save_frames": False,
                    "save_heatmap_npy": False,
                    "save_heatmap_png": True,
                    "save_jsonl": True,
                },
            }
        )

    return _make
