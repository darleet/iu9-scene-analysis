from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.amp import GradScaler

from scene_analysis.student.losses import StudentHeatmapLoss
from scene_analysis.student.model_registry import create_student_model
from scene_analysis.student.trainer import StudentTrainer


def _write_prepared_sample(root: Path, split: str, sample_id: str) -> None:
    split_root = root / split
    images_dir = split_root / "images"
    masks_dir = split_root / "masks"
    teacher_dir = split_root / "teacher_heatmaps"
    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    teacher_dir.mkdir(parents=True, exist_ok=True)

    image = np.zeros((48, 64, 3), dtype=np.uint8)
    mask = np.zeros((48, 64), dtype=np.uint8)
    mask[16:28, 24:36] = 1
    teacher = np.zeros((48, 64), dtype=np.float32)
    teacher[16:28, 24:36] = 0.9

    cv2.imwrite(str(images_dir / f"{sample_id}.png"), image)
    cv2.imwrite(str(masks_dir / f"{sample_id}_labels_semantic.png"), mask)
    np.save(teacher_dir / f"{sample_id}.npy", teacher)


def test_student_trainer_smoke_saves_artifacts(train_config) -> None:
    trainer = StudentTrainer(train_config, "student_s")
    summary = trainer.train()
    output_dir = Path(summary["summary"]).parent

    assert summary["status"] == "ok"
    assert (output_dir / "checkpoints" / "best.pt").exists()
    assert (output_dir / "checkpoints" / "last.pt").exists()
    assert (output_dir / "history.csv").exists()
    assert (output_dir / "summary.json").exists()
    assert (output_dir / "pr_curve.png").exists()
    assert (output_dir / "previews" / "epoch_001_sample_grid.png").exists()


def test_student_trainer_skips_zero_valid_train_batches(train_config) -> None:
    train_config.training.use_amp = False
    train_config.training.max_train_batches = None
    trainer = StudentTrainer(train_config, "student_s")
    trainer.model = create_student_model("student_s", train_config.models.student_s).to(trainer.device)
    criterion = StudentHeatmapLoss(train_config.loss)
    optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=train_config.optimizer.lr)
    scaler = GradScaler("cuda", enabled=False)
    height = train_config.input.height
    width = train_config.input.width

    image = torch.zeros(1, 3, height, width)
    zero = torch.zeros(1, 1, height, width)
    one = torch.ones(1, 1, height, width)
    empty_batch = {
        "image": image,
        "obstacle_target": zero,
        "valid_mask": zero,
        "ignore_mask": one,
        "teacher_heatmap": zero,
        "sample_id": ["empty"],
    }
    valid_batch = {
        "image": image,
        "obstacle_target": zero.clone(),
        "valid_mask": one,
        "ignore_mask": zero,
        "teacher_heatmap": zero,
        "sample_id": ["valid"],
    }
    valid_batch["obstacle_target"][:, :, 4:8, 4:8] = 1.0

    metrics = trainer.train_one_epoch(
        epoch=1,
        dataloader=[empty_batch, valid_batch],
        criterion=criterion,
        optimizer=optimizer,
        scaler=scaler,
    )

    assert metrics["train_loss"] > 0.0


def test_student_trainer_reuses_visual_preview_indices(train_config) -> None:
    trainer = StudentTrainer(train_config, "student_s")
    _, val_loader = trainer._create_dataloaders()

    first_batch = trainer._sample_visual_preview_batch(epoch=1, dataloader=val_loader)
    first_indices = list(trainer._visual_preview_indices or [])
    second_batch = trainer._sample_visual_preview_batch(epoch=5, dataloader=val_loader)
    second_indices = list(trainer._visual_preview_indices or [])

    assert first_indices
    assert second_indices == first_indices
    assert list(second_batch["sample_id"]) == list(first_batch["sample_id"])


def test_student_trainer_samples_up_to_eight_visual_preview_examples(tmp_path: Path, make_train_config) -> None:
    root = tmp_path / "prepared"
    for split in ("train", "val"):
        for index in range(10):
            _write_prepared_sample(root, split, f"{split}_{index}")

    config = make_train_config(root)
    config.validation.num_visual_examples = 12
    trainer = StudentTrainer(config, "student_s")
    _, val_loader = trainer._create_dataloaders()

    batch = trainer._sample_visual_preview_batch(epoch=1, dataloader=val_loader)

    assert len(trainer._visual_preview_indices or []) == 8
    assert len(batch["sample_id"]) == 8


def test_student_trainer_loads_multiple_prepared_datasets(tmp_path: Path, make_train_config) -> None:
    first_root = tmp_path / "prepared_a"
    second_root = tmp_path / "prepared_b"
    for root, prefix in ((first_root, "a"), (second_root, "b")):
        for split in ("train", "val"):
            for index in range(2):
                _write_prepared_sample(root, split, f"{prefix}_{split}_{index}")

    config = make_train_config(tmp_path / "unused")
    assert config.dataset is not None
    first_dataset = config.dataset.model_copy(update={"name": "first", "prepared_root_dir": first_root})
    second_dataset = config.dataset.model_copy(update={"name": "second", "prepared_root_dir": second_root})
    config.dataset = first_dataset
    config.datasets = [first_dataset, second_dataset]

    trainer = StudentTrainer(config, "student_s")
    train_loader, val_loader = trainer._create_dataloaders()

    assert len(train_loader.dataset) == 4
    assert len(val_loader.dataset) == 4
