from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from loguru import logger

from scene_analysis.obstacle_map.visualization import overlay_heatmap_on_image
from scene_analysis.student.config import StudentInferenceConfig, StudentModelConfig
from scene_analysis.student.model_registry import create_student_model, validate_student_name
from scene_analysis.student.visualization import draw_inference_overlay_text


class StudentInferenceRunner:
    def __init__(
        self,
        config: StudentInferenceConfig,
        student_name: str | None = None,
        checkpoint_path: Path | None = None,
    ) -> None:
        self.config = config
        self.student_name = validate_student_name(student_name or config.inference.student)
        self.checkpoint_path = (checkpoint_path or config.inference.checkpoint_path).expanduser()
        self.device = self._resolve_device(config.inference.device)
        self.model: torch.nn.Module | None = None
        self.input_height = config.input.height
        self.input_width = config.input.width
        self.normalize_mean = np.asarray(config.input.normalize_mean, dtype=np.float32)
        self.normalize_std = np.asarray(config.input.normalize_std, dtype=np.float32)
        self.checkpoint: dict[str, Any] = {}

    def load_checkpoint(self, checkpoint_path: Path | None = None) -> None:
        path = (checkpoint_path or self.checkpoint_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Student checkpoint not found: {path}")
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        checkpoint_student = checkpoint.get("student_name")
        if checkpoint_student is not None and checkpoint_student != self.student_name:
            raise ValueError(
                f"Checkpoint belongs to '{checkpoint_student}', but requested student is '{self.student_name}'"
            )

        self.input_height = int(checkpoint.get("input_height", self.input_height))
        self.input_width = int(checkpoint.get("input_width", self.input_width))
        self.normalize_mean = np.asarray(checkpoint.get("normalize_mean", self.normalize_mean), dtype=np.float32)
        self.normalize_std = np.asarray(checkpoint.get("normalize_std", self.normalize_std), dtype=np.float32)

        model_config = self._model_config_from_checkpoint(checkpoint)
        model = create_student_model(self.student_name, model_config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()
        self.model = model
        self.checkpoint = checkpoint
        self.checkpoint_path = path
        logger.info("Loaded student checkpoint {} on {}", path, self.device)

    def preprocess_frame(self, frame_bgr: np.ndarray) -> torch.Tensor:
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("frame_bgr must be a BGR image with shape HxWx3")
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(image_rgb, (self.input_width, self.input_height), interpolation=cv2.INTER_LINEAR)
        image = resized.astype(np.float32) / 255.0
        image = (image - self.normalize_mean) / self.normalize_std
        image = np.transpose(image, (2, 0, 1)).astype(np.float32, copy=False)
        return torch.from_numpy(image[None, ...]).to(self.device)

    def predict_frame(self, frame_bgr: np.ndarray) -> dict[str, Any]:
        if self.model is None:
            self.load_checkpoint(self.checkpoint_path)
        assert self.model is not None

        original_height, original_width = frame_bgr.shape[:2]
        input_tensor = self.preprocess_frame(frame_bgr)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        started = time.perf_counter()
        with torch.no_grad():
            outputs = self.model(input_tensor)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        inference_ms = (time.perf_counter() - started) * 1000.0

        heatmap_small = outputs["final_heatmap"][0, 0].detach().float().cpu().numpy()
        heatmap = cv2.resize(heatmap_small, (original_width, original_height), interpolation=cv2.INTER_LINEAR)
        heatmap = np.clip(heatmap.astype(np.float32), 0.0, 1.0)
        stats = {
            "inference_ms": float(inference_ms),
            "heatmap_min": float(np.min(heatmap)),
            "heatmap_max": float(np.max(heatmap)),
            "heatmap_mean": float(np.mean(heatmap)),
        }
        overlay = self.render_overlay(frame_bgr, heatmap, stats)
        return {"heatmap": heatmap, "overlay": overlay, "stats": stats}

    def render_overlay(
        self,
        frame_bgr: np.ndarray,
        heatmap: np.ndarray,
        stats: dict[str, float],
        frame_index: int | None = None,
    ) -> np.ndarray:
        overlay = overlay_heatmap_on_image(
            frame_bgr,
            heatmap,
            alpha=self.config.visualization.alpha,
            colormap=self.config.visualization.colormap,
        )
        if self.config.visualization.show_binary_mask:
            binary = (heatmap >= self.config.visualization.threshold).astype(np.uint8) * 255
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 255, 0), 1)
        return draw_inference_overlay_text(
            overlay,
            student_name=self.student_name,
            checkpoint_name=self.checkpoint_path.name,
            frame_index=frame_index,
            stats=stats,
            threshold=self.config.visualization.threshold if self.config.visualization.show_binary_mask else None,
            draw_model_name=self.config.visualization.draw_model_name,
            draw_stats=self.config.visualization.draw_stats,
        )

    def _model_config_from_checkpoint(self, checkpoint: dict[str, Any]) -> StudentModelConfig:
        config_payload = checkpoint.get("config") or {}
        model_payload = {}
        if isinstance(config_payload, dict):
            model_payload = ((config_payload.get("models") or {}).get(self.student_name) or {})
        backbone = checkpoint.get("backbone") or model_payload.get("backbone")
        if backbone is None:
            raise ValueError("Checkpoint does not contain student backbone metadata")
        return StudentModelConfig(
            backbone=backbone,
            pretrained_backbone=False,
            decoder_channels=model_payload.get("decoder_channels", [128, 64, 32]),
            dropout=float(model_payload.get("dropout", 0.0)),
        )

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        normalized = requested.strip().lower()
        if normalized == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")
        return torch.device(normalized)
