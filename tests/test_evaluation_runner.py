from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scene_analysis.config import (
    EvaluationConfig,
    EvaluationDatasetConfig,
    EvaluationLabelsConfig,
    EvaluationMetricsConfig,
    EvaluationOutputsConfig,
    EvaluationPredictionConfig,
)
from scene_analysis.evaluation.runner import EvaluationRunner


def _write_png(path: Path, array: np.ndarray) -> None:
    import cv2

    if not cv2.imwrite(str(path), array):
        raise IOError(f"Failed to save test image: {path}")


def _prepare_dataset(root_dir: Path) -> None:
    images_dir = root_dir / "train" / "scene"
    masks_dir = root_dir / "train" / "scene"
    predictions_dir = root_dir / "predictions"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True)

    _write_png(images_dir / "sample_001_leftImg8bit.png", np.zeros((4, 4, 3), dtype=np.uint8))
    _write_png(images_dir / "sample_002_leftImg8bit.png", np.zeros((4, 4, 3), dtype=np.uint8))

    mask_1 = np.array(
        [
            [1, 2, 2, 1],
            [1, 2, 0, 1],
            [1, 1, 1, 1],
            [2, 2, 1, 1],
        ],
        dtype=np.uint8,
    )
    mask_2 = np.array(
        [
            [1, 1, 1, 1],
            [1, 2, 2, 1],
            [1, 2, 1, 1],
            [1, 1, 1, 0],
        ],
        dtype=np.uint8,
    )
    pred_1 = np.array(
        [
            [0.1, 0.9, 0.85, 0.2],
            [0.2, 0.95, 0.4, 0.1],
            [0.05, 0.1, 0.15, 0.05],
            [0.8, 0.75, 0.2, 0.1],
        ],
        dtype=np.float32,
    )
    pred_2 = np.array(
        [
            [0.05, 0.1, 0.05, 0.1],
            [0.1, 0.9, 0.85, 0.05],
            [0.2, 0.88, 0.1, 0.05],
            [0.05, 0.1, 0.05, 0.2],
        ],
        dtype=np.float32,
    )

    _write_png(masks_dir / "sample_001_gtCoarse_labelIds.png", mask_1)
    _write_png(masks_dir / "sample_002_gtCoarse_labelIds.png", mask_2)
    np.save(predictions_dir / "sample_001.npy", pred_1)
    np.save(predictions_dir / "sample_002.npy", pred_2)


def test_runner_computes_summary_and_saves_outputs(tmp_path: Path) -> None:
    dataset_root = tmp_path / "lost_and_found"
    outputs_dir = tmp_path / "eval_outputs"
    _prepare_dataset(dataset_root)

    runner = EvaluationRunner(
        EvaluationConfig(
            enabled=True,
            dataset=EvaluationDatasetConfig(
                name="lost_and_found",
                root_dir=dataset_root,
                images_dir=".",
                masks_dir=".",
                predictions_dir="predictions",
                split_file=None,
                file_extension_images="_leftImg8bit.png",
                file_extension_masks="_gtCoarse_labelIds.png",
                file_extension_predictions=".npy",
            ),
            labels=EvaluationLabelsConfig(
                obstacle_values=[],
                background_values=[1],
                ignore_values=[0, 255],
                unmapped_values="obstacle",
            ),
            prediction=EvaluationPredictionConfig(
                resize_to_gt=True,
                clip_to_unit_range=True,
                allow_png_heatmaps=False,
            ),
            metrics=EvaluationMetricsConfig(average_precision=True),
            outputs=EvaluationOutputsConfig(
                output_dir=outputs_dir,
                save_pr_curve_png=True,
                save_per_sample_csv=True,
                save_summary_json=True,
                save_hard_examples=True,
                hard_examples_top_k=5,
            ),
        )
    )

    summary = runner.run()

    assert summary.dataset_name == "lost_and_found"
    assert summary.num_samples == 2
    assert summary.num_valid_samples == 2
    assert summary.average_precision > 0.9

    summary_path = outputs_dir / "summary.json"
    per_sample_path = outputs_dir / "per_sample.csv"
    pr_curve_path = outputs_dir / "pr_curve.png"
    hard_examples_path = outputs_dir / "hard_examples.csv"

    assert summary_path.exists()
    assert per_sample_path.exists()
    assert pr_curve_path.exists()
    assert hard_examples_path.exists()

    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert payload["dataset_name"] == "lost_and_found"
    assert "average_precision" in payload
