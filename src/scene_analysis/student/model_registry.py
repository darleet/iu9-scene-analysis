from __future__ import annotations

from scene_analysis.student.config import STUDENT_NAMES, StudentModelConfig
from scene_analysis.student.model import StudentHeatmapNet

STUDENT_REGISTRY: dict[str, dict[str, str]] = {
    "student_s": {
        "backbone": "mobilenet_v3_small",
        "description": "MobileNetV3 Small compact baseline",
    },
    "student_m": {
        "backbone": "shufflenet_v2_x1_0",
        "description": "ShuffleNetV2 x1.0 fast alternative backbone",
    },
    "student_q": {
        "backbone": "efficientnet_b0",
        "description": "EfficientNet-B0 stronger encoder",
    },
}


def validate_student_name(student_name: str) -> str:
    normalized = student_name.strip()
    if normalized not in STUDENT_NAMES:
        raise ValueError(f"Unsupported student name '{student_name}'. Expected one of {sorted(STUDENT_NAMES)}")
    return normalized


def create_student_model(student_name: str, config: StudentModelConfig) -> StudentHeatmapNet:
    """Создание student модели по конфигу"""
    normalized = validate_student_name(student_name)
    expected_backbone = STUDENT_REGISTRY[normalized]["backbone"]
    if config.backbone != expected_backbone:
        raise ValueError(
            f"Student '{normalized}' must use backbone '{expected_backbone}', got '{config.backbone}'"
        )
    return StudentHeatmapNet(
        backbone_name=config.backbone,
        pretrained_backbone=config.pretrained_backbone,
        decoder_channels=config.decoder_channels,
        dropout=config.dropout,
    )
