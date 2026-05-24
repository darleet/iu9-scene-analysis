from __future__ import annotations

from pathlib import Path

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
