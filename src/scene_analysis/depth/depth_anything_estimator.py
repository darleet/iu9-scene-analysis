from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np
from loguru import logger

try:
    from PIL import Image
except ImportError as error:
    raise ImportError(
        "Pillow is required for Depth Anything V2 inference. Install project dependencies with 'poetry install'"
    ) from error

try:
    import torch
    import torch.nn.functional as F
except ImportError as error:
    raise ImportError(
        "PyTorch is required for Depth Anything V2 inference. Install project dependencies with 'poetry install'"
    ) from error

try:
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation
except ImportError as error:
    raise ImportError(
        "transformers is required for Depth Anything V2 inference. "
        "Install project dependencies with 'poetry install'"
    ) from error

from scene_analysis.depth.base import DepthEstimator
from scene_analysis.types import DepthResult
from scene_analysis.utils import ensure_float32_array, ensure_uint8_image


class DepthAnythingV2Estimator(DepthEstimator):
    """Depth estimator на базе Depth Anything V2"""

    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        cache_dir: str | None = None,
        use_fp16: bool = False,
        compile_model: bool = False,
    ) -> None:
        self.model_name = model_name.strip()
        self.requested_device = device
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir else None
        self.use_fp16 = use_fp16
        self.compile_model = compile_model
        self.scale_type = "metric" if "metric" in self.model_name.lower() else "relative"

        self.device = self._resolve_device()
        self.model_dtype = torch.float16 if self.use_fp16 and self.device.type == "cuda" else torch.float32
        self.image_processor: Any | None = None
        self.model: Any | None = None

        if self.use_fp16 and self.device.type != "cuda":
            logger.warning("fp16 inference is requested, but device '{}' does not support it safely. Using fp32", self.device)

        self._load_model()

    def _resolve_device(self) -> torch.device:
        requested = self.requested_device.strip().lower()
        if requested == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            return torch.device("cpu")

        if requested == "cpu":
            return torch.device("cpu")

        if requested == "mps":
            if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
                raise RuntimeError("Depth inference device 'mps' was requested, but MPS is not available")
            return torch.device("mps")

        if requested == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("Depth inference device 'cuda' was requested, but CUDA is not available")
            return torch.device("cuda")

        if requested.startswith("cuda:"):
            if not torch.cuda.is_available():
                raise RuntimeError(f"Depth inference device '{requested}' was requested, but CUDA is not available")
            resolved = torch.device(requested)
            device_index = resolved.index if resolved.index is not None else 0
            if device_index >= torch.cuda.device_count():
                raise RuntimeError(
                    f"Depth inference device '{requested}' is unavailable: "
                    f"CUDA device count is {torch.cuda.device_count()}."
                )
            return resolved

        raise ValueError(f"Unsupported depth inference device: {self.requested_device}")

    def _load_model(self) -> None:
        pretrained_kwargs: dict[str, Any] = {}
        if self.cache_dir is not None:
            pretrained_kwargs["cache_dir"] = str(self.cache_dir)

        model_kwargs = dict(pretrained_kwargs)
        if self.use_fp16 and self.device.type == "cuda":
            model_kwargs["torch_dtype"] = self.model_dtype

        try:
            self.image_processor = AutoImageProcessor.from_pretrained(self.model_name, **pretrained_kwargs)
            self.model = AutoModelForDepthEstimation.from_pretrained(self.model_name, **model_kwargs)
            self.model.to(self.device)
            self.model.eval()
        except Exception as error:
            raise RuntimeError(f"Failed to load depth model '{self.model_name}': {error}") from error

        if self.compile_model:
            if hasattr(torch, "compile") and self.device.type == "cuda":
                try:
                    self.model = torch.compile(self.model)
                except Exception as error:
                    logger.warning("torch.compile failed for model '{}': {}", self.model_name, error)
            else:
                logger.warning(
                    "Model compilation was requested for model '{}', but the current setup is not supported. "
                    "Skipping torch.compile",
                    self.model_name,
                )

    def predict(self, image: np.ndarray) -> DepthResult:
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("Input image must be a non-empty numpy array")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("DepthAnythingV2Estimator expects an OpenCV BGR image with shape HxWx3")
        if self.model is None or self.image_processor is None:
            raise RuntimeError("Depth model is not loaded.")

        prepared_image = ensure_uint8_image(image)
        original_height, original_width = prepared_image.shape[:2]
        rgb_image = cv2.cvtColor(prepared_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)

        inputs = self.image_processor(images=pil_image, return_tensors="pt")
        model_inputs = self._move_inputs_to_device(inputs)

        start_time = perf_counter()
        with torch.inference_mode():
            outputs = self.model(**model_inputs)
            predicted_depth = getattr(outputs, "predicted_depth", None)
            if predicted_depth is None or predicted_depth.numel() == 0:
                raise RuntimeError("Depth model returned an empty predicted_depth tensor.")

            if predicted_depth.ndim == 2:
                predicted_depth = predicted_depth.unsqueeze(0)
            if predicted_depth.ndim != 3:
                raise RuntimeError(
                    f"Unexpected predicted_depth shape from model '{self.model_name}': "
                    f"{tuple(predicted_depth.shape)}"
                )

            resized_depth = F.interpolate(
                predicted_depth.unsqueeze(1),
                size=(original_height, original_width),
                mode="bicubic",
                align_corners=False,
            ).squeeze(1)

        inference_ms = (perf_counter() - start_time) * 1000.0
        depth_map = ensure_float32_array(resized_depth[0].detach().float().cpu().numpy())
        depth_map = np.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        if depth_map.size == 0:
            raise RuntimeError("Depth model produced an empty depth map after post-processing")

        metadata = {
            "provider": "depth_anything_v2",
            "model": self.model_name,
            "device": str(self.device),
            "inference_ms": round(inference_ms, 3),
            "original_height": original_height,
            "original_width": original_width,
            "output_height": int(depth_map.shape[0]),
            "output_width": int(depth_map.shape[1]),
            "depth_min": float(depth_map.min()),
            "depth_max": float(depth_map.max()),
            "depth_mean": float(depth_map.mean()),
            "scale_type": self.scale_type,
            "status": "ok",
        }

        return DepthResult(
            depth_map=depth_map,
            confidence_map=None,
            metadata=metadata,
        )

    def _move_inputs_to_device(self, inputs: Any) -> dict[str, torch.Tensor]:
        prepared_inputs: dict[str, torch.Tensor] = {}
        for key, value in dict(inputs).items():
            if not isinstance(value, torch.Tensor):
                continue
            if value.is_floating_point() and self.use_fp16 and self.device.type == "cuda":
                prepared_inputs[key] = value.to(device=self.device, dtype=torch.float16)
            else:
                prepared_inputs[key] = value.to(self.device)
        return prepared_inputs
