from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from scene_analysis.evaluation.types import EvaluationItemResult
from scene_analysis.utils import safe_mkdir, to_serializable

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


MAX_PR_CURVE_POINTS = 5000


def plot_precision_recall_curve(
    precision: np.ndarray,
    recall: np.ndarray,
    ap: float,
    output_path: Path,
) -> None:
    safe_mkdir(output_path.parent)
    if precision.size == 0 or recall.size == 0:
        _save_empty_curve(output_path, ap)
        return

    plot_precision, plot_recall = _downsample_curve_for_plot(
        precision=precision,
        recall=recall,
        max_points=MAX_PR_CURVE_POINTS,
    )

    if plt is not None:
        figure, axis = plt.subplots(figsize=(7, 5))
        axis.plot(plot_recall, plot_precision, color="tab:red", linewidth=2)
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.set_xlabel("Recall")
        axis.set_ylabel("Precision")
        axis.set_title(f"Precision-Recall Curve (AP={ap:.4f})")
        axis.grid(True, alpha=0.3)
        figure.tight_layout()
        figure.savefig(output_path, dpi=150)
        plt.close(figure)
        return

    canvas = np.full((480, 640, 3), 255, dtype=np.uint8)
    origin = (70, 420)
    top_right = (580, 60)
    cv2.rectangle(canvas, origin, top_right, (0, 0, 0), 2)

    points = []
    for recall_value, precision_value in zip(plot_recall, plot_precision, strict=False):
        x_coord = int(origin[0] + float(recall_value) * (top_right[0] - origin[0]))
        y_coord = int(origin[1] - float(precision_value) * (origin[1] - top_right[1]))
        points.append((x_coord, y_coord))
    if len(points) >= 2:
        cv2.polylines(canvas, [np.array(points, dtype=np.int32)], isClosed=False, color=(0, 0, 255), thickness=2)

    cv2.putText(canvas, f"PR Curve AP={ap:.4f}", (70, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Recall", (280, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, "Precision", (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    if not cv2.imwrite(str(output_path), canvas):
        raise IOError(f"Failed to save PR curve image: {output_path}")


def save_hard_examples_report(
    item_results: list[EvaluationItemResult],
    output_path: Path,
    top_k: int,
) -> None:
    safe_mkdir(output_path.parent)
    sorted_items = sorted(item_results, key=_hard_example_sort_key)
    rows = [to_serializable(item) for item in sorted_items[:top_k]]
    if pd is not None:
        pd.DataFrame(rows).to_csv(output_path, index=False)
        return

    _write_csv_rows(output_path, rows)


def _save_empty_curve(output_path: Path, ap: float) -> None:
    canvas = np.full((320, 480, 3), 255, dtype=np.uint8)
    message = "PR curve unavailable"
    cv2.putText(canvas, message, (70, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"AP={ap}", (170, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
    if not cv2.imwrite(str(output_path), canvas):
        raise IOError(f"Failed to save PR curve placeholder: {output_path}")


def _downsample_curve_for_plot(
    precision: np.ndarray,
    recall: np.ndarray,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    if precision.shape != recall.shape:
        raise ValueError("Precision and recall must have the same shape")
    if max_points <= 0:
        raise ValueError("max_points must be positive")
    if precision.size <= max_points:
        return precision, recall

    indices = np.linspace(0, precision.size - 1, num=max_points, dtype=np.int64)
    indices = np.unique(np.concatenate(([0], indices, [precision.size - 1])))
    return precision[indices], recall[indices]


def _hard_example_sort_key(item: EvaluationItemResult) -> tuple[float, int, str]:
    proxy_score = item.ap_local_proxy if item.ap_local_proxy is not None else -1.0
    return (proxy_score, -item.positive_pixels, item.sample_id)


def _write_csv_rows(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()}) if rows else []
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
