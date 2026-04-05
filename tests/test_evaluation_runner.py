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
    images_dir = root_dir / "images"
    masks_dir = root_dir / "masks"
    predictions_dir = root_dir / "predictions"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)
    predictions_dir.mkdir(parents=True)

    _write_png(images_dir / "sample_001.png", np.zeros((4, 4, 3), dtype=np.uint8))
    _write_png(images_dir / "sample_002.png", np.zeros((4, 4, 3), dtype=np.uint8))

    mask_1 = np.array(
        [
            [0, 1, 1, 0],
            [0, 1, 255, 0],
            [0, 0, 0, 0],
            [1, 1, 0, 0],
        ],
        dtype=np.uint8,
    )
    mask_2 = np.array(
        [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 255],
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

    _write_png(masks_dir / "sample_001.png", mask_1)
    _write_png(masks_dir / "sample_002.png", mask_2)
    np.save(predictions_dir / "sample_001.npy", pred_1)
    np.save(predictions_dir / "sample_002.npy", pred_2)


def test_runner_computes_summary_and_saves_outputs(tmp_path: Path) -> None:
    dataset_root = tmp_path / "road_obstacle_21"
    outputs_dir = tmp_path / "eval_outputs"
    _prepare_dataset(dataset_root)

    runner = EvaluationRunner(
        EvaluationConfig(
            enabled=True,
            dataset=EvaluationDatasetConfig(
                name="road_obstacle_21",
                root_dir=dataset_root,
                images_dir="images",
                masks_dir="masks",
                predictions_dir="predictions",
                split_file=None,
                file_extension_images=".png",
                file_extension_masks=".png",
                file_extension_predictions=".npy",
            ),
            labels=EvaluationLabelsConfig(
                obstacle_values=[1],
                background_values=[0],
                ignore_values=[255],
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

    assert summary.dataset_name == "road_obstacle_21"
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
    assert payload["dataset_name"] == "road_obstacle_21"
    assert "average_precision" in payload
