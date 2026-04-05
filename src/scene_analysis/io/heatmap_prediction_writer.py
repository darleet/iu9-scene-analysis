from __future__ import annotations

from pathlib import Path

import numpy as np

from scene_analysis.utils import ensure_float32_array, safe_mkdir


class HeatmapPredictionWriter:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = safe_mkdir(output_dir.expanduser())

    def save_prediction(self, sample_id: str, heatmap: np.ndarray) -> Path:
        """Сохранить prediction в .npy с именем sample_id"""
        normalized_sample_id = sample_id.strip()
        if not normalized_sample_id:
            raise ValueError("sample_id must not be empty")

        output_path = self.output_dir / f"{normalized_sample_id}.npy"
        np.save(output_path, ensure_float32_array(heatmap))
        return output_path
