from __future__ import annotations

import torch

from scene_analysis.student.config import StudentModelConfig
from scene_analysis.student.model import count_parameters
from scene_analysis.student.model_registry import STUDENT_REGISTRY, create_student_model


def test_student_models_forward_shape_and_range() -> None:
    for student_name, metadata in STUDENT_REGISTRY.items():
        model = create_student_model(
            student_name,
            StudentModelConfig(
                backbone=metadata["backbone"],
                pretrained_backbone=False,
                decoder_channels=[16],
                dropout=0.0,
            ),
        )
        model.eval()
        with torch.no_grad():
            outputs = model(torch.randn(2, 3, 64, 96))

        assert outputs["obstacle_logits"].shape == (2, 1, 64, 96)
        assert outputs["roi_logits"].shape == (2, 1, 64, 96)
        assert outputs["final_heatmap"].shape == (2, 1, 64, 96)
        assert torch.equal(outputs["final_heatmap"], outputs["obstacle_prob"])
        assert torch.all(outputs["final_heatmap"] >= 0.0)
        assert torch.all(outputs["final_heatmap"] <= 1.0)
        assert count_parameters(model) > 0


def test_student_model_can_apply_roi_head_to_final_heatmap() -> None:
    model = create_student_model(
        "student_s",
        StudentModelConfig(
            backbone="mobilenet_v3_small",
            pretrained_backbone=False,
            decoder_channels=[16],
            dropout=0.0,
            use_roi_head_in_heatmap=True,
        ),
    )
    model.eval()

    with torch.no_grad():
        outputs = model(torch.randn(2, 3, 64, 96))

    assert torch.allclose(outputs["final_heatmap"], outputs["obstacle_prob"] * outputs["roi_prob"])
    assert torch.all(outputs["final_heatmap"] <= outputs["obstacle_prob"])
