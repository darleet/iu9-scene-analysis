from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from scene_analysis.student.inference import StudentInferenceRunner
from scene_analysis.student.video_runtime import run_student_on_video_folder


def test_student_inference_loads_checkpoint_and_predicts_original_size(
    student_checkpoint: Path,
    make_inference_config,
) -> None:
    config = make_inference_config(student_checkpoint)
    runner = StudentInferenceRunner(config, "student_s", student_checkpoint)
    runner.load_checkpoint()
    frame = np.zeros((40, 70, 3), dtype=np.uint8)

    result = runner.predict_frame(frame)
    checkpoint = torch.load(student_checkpoint, map_location="cpu", weights_only=False)

    assert checkpoint["student_name"] == "student_s"
    assert result["heatmap"].shape == (40, 70)
    assert result["overlay"].shape == frame.shape
    assert 0.0 <= float(result["heatmap"].min()) <= 1.0
    assert 0.0 <= float(result["heatmap"].max()) <= 1.0
    assert result["stats"]["inference_ms"] >= 0.0


def test_student_video_folder_runtime_on_synthetic_video(
    tmp_path: Path,
    student_checkpoint: Path,
    make_inference_config,
) -> None:
    videos_dir = tmp_path / "videos"
    videos_dir.mkdir()
    video_path = videos_dir / "input.avi"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"MJPG"), 5.0, (64, 48))
    assert writer.isOpened()
    for index in range(3):
        frame = np.full((48, 64, 3), index * 30, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    config = make_inference_config(student_checkpoint)
    config.video_folder.input_dir = videos_dir
    config.video_folder.output_dir = tmp_path / "video_out"
    summary = run_student_on_video_folder(config, "student_s", student_checkpoint)
    output_dir = Path(summary["output_dir"])

    assert summary["videos_processed"] == 1
    assert summary["frames_processed"] == 2
    assert (output_dir / "videos" / "input_overlay.mp4").exists()
    assert (output_dir / "results.jsonl").exists()
    assert (output_dir / "summary.json").exists()
    assert any((output_dir / "heatmaps_png").glob("*.png"))
