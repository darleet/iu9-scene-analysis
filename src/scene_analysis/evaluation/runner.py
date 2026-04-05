from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from loguru import logger

from scene_analysis.config import EvaluationConfig
from scene_analysis.evaluation.dataset import RoadObstacle21Dataset
from scene_analysis.evaluation.io import load_mask, load_prediction
from scene_analysis.evaluation.metrics import (
    compute_average_precision,
    compute_dataset_ap,
    compute_precision_recall_curve_data,
)
from scene_analysis.evaluation.preparation import (
    build_valid_label_mask,
    flatten_scores_and_labels,
    normalize_prediction_map,
    resize_prediction_to_mask,
)
from scene_analysis.evaluation.types import EvaluationItemResult, EvaluationSample, EvaluationSummary
from scene_analysis.evaluation.visualization import plot_precision_recall_curve, save_hard_examples_report
from scene_analysis.utils import safe_mkdir, to_serializable

try:
    import pandas as pd
except ImportError:
    pd = None


class EvaluationRunner:
    def __init__(self, config: EvaluationConfig) -> None:
        self.config = config
        self.output_dir = safe_mkdir(config.outputs.output_dir)
        self._sample_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def run(self) -> EvaluationSummary:
        self._sample_arrays.clear()
        dataset = self._create_dataset()
        samples = dataset.discover_samples()
        item_results = [self._evaluate_sample(sample) for sample in samples]
        all_scores, all_labels = self._collect_global_arrays(item_results)

        if self.config.metrics.average_precision:
            global_ap = compute_dataset_ap(all_scores, all_labels)
            if all_scores and all_labels:
                global_scores = np.concatenate(all_scores)
                global_labels = np.concatenate(all_labels)
                precision, recall, thresholds = compute_precision_recall_curve_data(global_scores, global_labels)
            else:
                precision = np.empty(0, dtype=np.float32)
                recall = np.empty(0, dtype=np.float32)
                thresholds = np.empty(0, dtype=np.float32)
        else:
            global_ap = float("nan")
            precision = np.empty(0, dtype=np.float32)
            recall = np.empty(0, dtype=np.float32)
            thresholds = np.empty(0, dtype=np.float32)

        summary = EvaluationSummary(
            dataset_name=self.config.dataset.name,
            num_samples=len(samples),
            num_valid_samples=sum(item.valid_pixels > 0 for item in item_results),
            total_valid_pixels=int(sum(item.valid_pixels for item in item_results)),
            total_positive_pixels=int(sum(item.positive_pixels for item in item_results)),
            total_negative_pixels=int(sum(item.negative_pixels for item in item_results)),
            average_precision=float(global_ap),
            metadata={
                "dataset_root": str(dataset.root_dir),
                "images_dir": str(dataset.images_dir),
                "masks_dir": str(dataset.masks_dir),
                "predictions_dir": str(dataset.predictions_dir),
                "split_file": str(dataset.split_file) if dataset.split_file is not None else None,
                "status_counts": dict(Counter(item.status for item in item_results)),
                "pr_curve_points": int(len(precision)),
                "metrics_enabled": {
                    "average_precision": self.config.metrics.average_precision,
                },
            },
        )
        self._save_outputs(summary, item_results, precision, recall, thresholds)
        return summary

    def _evaluate_sample(self, sample: EvaluationSample) -> EvaluationItemResult:
        try:
            gt_mask = load_mask(sample.mask_path)
            prediction = load_prediction(sample.prediction_path)
            if prediction.shape != gt_mask.shape:
                if not self.config.prediction.resize_to_gt:
                    raise ValueError(
                        f"Prediction shape {prediction.shape} does not match GT shape {gt_mask.shape} "
                        f"for sample '{sample.sample_id}'"
                    )
                prediction = resize_prediction_to_mask(prediction, gt_mask.shape)

            prediction = normalize_prediction_map(
                prediction,
                clip_to_unit_range=self.config.prediction.clip_to_unit_range,
            )
            valid_mask, positive_mask = build_valid_label_mask(
                gt_mask=gt_mask,
                obstacle_values=self.config.labels.obstacle_values,
                background_values=self.config.labels.background_values,
                ignore_values=self.config.labels.ignore_values,
            )
            scores, labels = flatten_scores_and_labels(prediction, valid_mask, positive_mask)

            valid_pixels = int(scores.size)
            positive_pixels = int(labels.sum())
            negative_pixels = int(valid_pixels - positive_pixels)
            if valid_pixels > 0:
                self._sample_arrays[sample.sample_id] = (scores, labels)
            else:
                self._sample_arrays.pop(sample.sample_id, None)

            if valid_pixels == 0:
                status = "invalid_sample"
                ap_local_proxy = None
            elif positive_pixels == 0:
                status = "missing_positive"
                ap_local_proxy = None
            elif negative_pixels == 0:
                status = "missing_negative"
                ap_local_proxy = None
            else:
                status = "ok"
                ap_local_proxy = compute_average_precision(scores, labels)

            return EvaluationItemResult(
                sample_id=sample.sample_id,
                valid_pixels=valid_pixels,
                positive_pixels=positive_pixels,
                negative_pixels=negative_pixels,
                score_min=float(np.min(scores)) if valid_pixels > 0 else float("nan"),
                score_max=float(np.max(scores)) if valid_pixels > 0 else float("nan"),
                score_mean=float(np.mean(scores)) if valid_pixels > 0 else float("nan"),
                ap_local_proxy=float(ap_local_proxy) if ap_local_proxy is not None else None,
                status=status,
            )
        except Exception as error:
            logger.warning("Failed to evaluate sample '{}': {}", sample.sample_id, error)
            self._sample_arrays.pop(sample.sample_id, None)
            return EvaluationItemResult(
                sample_id=sample.sample_id,
                valid_pixels=0,
                positive_pixels=0,
                negative_pixels=0,
                score_min=float("nan"),
                score_max=float("nan"),
                score_mean=float("nan"),
                ap_local_proxy=None,
                status="error",
            )

    def _collect_global_arrays(
        self,
        item_results: list[EvaluationItemResult],
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        all_scores: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        for item in item_results:
            sample_arrays = self._sample_arrays.get(item.sample_id)
            if sample_arrays is None:
                continue
            scores, labels = sample_arrays
            if scores.size == 0:
                continue
            all_scores.append(scores)
            all_labels.append(labels)
        return all_scores, all_labels

    def _save_outputs(
        self,
        summary: EvaluationSummary,
        item_results: list[EvaluationItemResult],
        precision: np.ndarray,
        recall: np.ndarray,
        thresholds: np.ndarray,
    ) -> None:
        output_paths = {
            "summary_json": self.output_dir / "summary.json",
            "per_sample_csv": self.output_dir / "per_sample.csv",
            "pr_curve_png": self.output_dir / "pr_curve.png",
            "hard_examples_csv": self.output_dir / "hard_examples.csv",
        }
        summary.metadata["output_paths"] = {key: str(path) for key, path in output_paths.items()}
        summary.metadata["threshold_count"] = int(len(thresholds))

        if self.config.outputs.save_summary_json:
            summary_payload = {
                "dataset_name": summary.dataset_name,
                "num_samples": summary.num_samples,
                "num_valid_samples": summary.num_valid_samples,
                "total_valid_pixels": summary.total_valid_pixels,
                "total_positive_pixels": summary.total_positive_pixels,
                "total_negative_pixels": summary.total_negative_pixels,
                "average_precision": summary.average_precision,
                "metadata": summary.metadata,
            }
            with output_paths["summary_json"].open("w", encoding="utf-8") as file:
                json.dump(to_serializable(summary_payload), file, ensure_ascii=False, indent=2)

        if self.config.outputs.save_per_sample_csv:
            rows = [to_serializable(item) for item in item_results]
            self._write_csv(output_paths["per_sample_csv"], rows)

        if self.config.outputs.save_pr_curve_png:
            plot_precision_recall_curve(
                precision=precision,
                recall=recall,
                ap=summary.average_precision,
                output_path=output_paths["pr_curve_png"],
            )

        if self.config.outputs.save_hard_examples:
            save_hard_examples_report(
                item_results=item_results,
                output_path=output_paths["hard_examples_csv"],
                top_k=self.config.outputs.hard_examples_top_k,
            )

    def _create_dataset(self) -> RoadObstacle21Dataset:
        if self.config.dataset.name != "road_obstacle_21":
            raise ValueError(f"Unsupported evaluation dataset: {self.config.dataset.name}")
        return RoadObstacle21Dataset(self.config.dataset)

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
        if pd is not None:
            pd.DataFrame(rows).to_csv(path, index=False)
            return

        fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else []
        with path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
