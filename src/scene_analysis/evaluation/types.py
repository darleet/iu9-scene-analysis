from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class EvaluationSample:
    sample_id: str
    image_path: Path | None
    mask_path: Path
    prediction_path: Path


@dataclass(slots=True)
class EvaluationItemResult:
    sample_id: str
    valid_pixels: int
    positive_pixels: int
    negative_pixels: int
    score_min: float
    score_max: float
    score_mean: float
    ap_local_proxy: float | None
    status: str


@dataclass(slots=True)
class EvaluationSummary:
    dataset_name: str
    num_samples: int
    num_valid_samples: int
    total_valid_pixels: int
    total_positive_pixels: int
    total_negative_pixels: int
    average_precision: float
    metadata: dict[str, Any] = field(default_factory=dict)
