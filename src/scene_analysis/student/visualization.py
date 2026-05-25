from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from scene_analysis.obstacle_map.visualization import heatmap_to_bgr, overlay_heatmap_on_image
from scene_analysis.utils import ensure_uint8_image, safe_mkdir


def render_training_preview(
    batch: dict[str, object],
    outputs: dict[str, torch.Tensor],
    output_path: Path,
    normalize_mean: list[float],
    normalize_std: list[float],
    *,
    max_samples: int = 4,
    alpha: float = 0.45,
    colormap: str = "inferno",
) -> Path:
    images = _to_numpy_batch(batch["image"])
    obstacle_target = _to_numpy_batch(batch["obstacle_target"])
    valid_mask = _to_numpy_batch(batch["valid_mask"])
    teacher_heatmap = _to_numpy_batch(batch["teacher_heatmap"])
    student_heatmap = _to_numpy_batch(outputs["final_heatmap"])

    sample_count = min(max_samples, images.shape[0])
    rows: list[np.ndarray] = []
    for index in range(sample_count):
        rgb = _unnormalize_image(images[index], normalize_mean, normalize_std)
        obstacle = obstacle_target[index, 0]
        valid = valid_mask[index, 0]
        teacher = teacher_heatmap[index, 0]
        student = student_heatmap[index, 0]

        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        overlay = overlay_heatmap_on_image(rgb_bgr, student, alpha=alpha, colormap=colormap)
        cells = [
            rgb_bgr,
            _mask_to_bgr(obstacle),
            _mask_to_bgr(valid),
            heatmap_to_bgr(teacher, colormap=colormap),
            heatmap_to_bgr(student, colormap=colormap),
            overlay,
        ]
        rows.append(np.concatenate([_label_cell(cell, label) for cell, label in zip(cells, _COLUMN_LABELS)], axis=1))

    grid = np.concatenate(rows, axis=0)
    safe_mkdir(output_path.parent)
    if not cv2.imwrite(str(output_path), grid):
        raise IOError(f"Failed to save training preview: {output_path}")
    return output_path


def save_heatmap_preview(
    heatmap: np.ndarray,
    output_path: Path,
    *,
    colormap: str = "inferno",
) -> Path:
    safe_mkdir(output_path.parent)
    image = heatmap_to_bgr(np.clip(heatmap.astype(np.float32), 0.0, 1.0), colormap=colormap)
    if not cv2.imwrite(str(output_path), image):
        raise IOError(f"Failed to save heatmap preview: {output_path}")
    return output_path


def draw_inference_overlay_text(
    overlay: np.ndarray,
    *,
    student_name: str,
    checkpoint_name: str,
    frame_index: int | None,
    stats: dict[str, float],
    threshold: float | None = None,
    draw_model_name: bool = True,
    draw_stats: bool = True,
) -> np.ndarray:
    image = ensure_uint8_image(overlay)
    lines: list[str] = []
    if draw_model_name:
        lines.append(f"Model: {student_name}")
        lines.append(f"Checkpoint: {checkpoint_name}")
    if frame_index is not None:
        lines.append(f"Frame: {frame_index}")
    if draw_stats:
        lines.append(f"Inference: {stats.get('inference_ms', 0.0):.1f} ms")
        lines.append(
            "Heatmap: "
            f"{stats.get('heatmap_min', 0.0):.3f} / "
            f"{stats.get('heatmap_mean', 0.0):.3f} / "
            f"{stats.get('heatmap_max', 0.0):.3f}"
        )
    if threshold is not None:
        lines.append(f"Threshold: {threshold:.2f}")

    y_position = 26
    for line in lines:
        cv2.putText(image, line, (12, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, line, (12, y_position), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 255, 255), 1, cv2.LINE_AA)
        y_position += 24
    return image


_COLUMN_LABELS = ["RGB", "GT", "Valid", "Teacher", "Student", "Overlay"]


def _to_numpy_batch(value: object) -> np.ndarray:
    if not isinstance(value, torch.Tensor):
        raise TypeError("Expected a torch.Tensor batch for preview rendering")
    return value.detach().cpu().float().numpy()


def _unnormalize_image(image_chw: np.ndarray, mean: list[float], std: list[float]) -> np.ndarray:
    image = np.transpose(image_chw, (1, 2, 0)).astype(np.float32)
    image = image * np.asarray(std, dtype=np.float32) + np.asarray(mean, dtype=np.float32)
    image = np.clip(image, 0.0, 1.0)
    return (image * 255.0).astype(np.uint8)


def _mask_to_bgr(mask: np.ndarray) -> np.ndarray:
    grayscale = np.clip(mask.astype(np.float32) * 255.0, 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(grayscale, cv2.COLOR_GRAY2BGR)


def _label_cell(image: np.ndarray, label: str) -> np.ndarray:
    cell = ensure_uint8_image(image)
    cv2.rectangle(cell, (0, 0), (cell.shape[1], 24), (0, 0, 0), thickness=-1)
    cv2.putText(cell, label, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return cell
