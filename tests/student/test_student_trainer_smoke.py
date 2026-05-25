from __future__ import annotations

from pathlib import Path

import torch
from torch.amp import GradScaler

from scene_analysis.student.losses import StudentHeatmapLoss
from scene_analysis.student.model_registry import create_student_model
from scene_analysis.student.trainer import StudentTrainer


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
