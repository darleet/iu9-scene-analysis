from __future__ import annotations

import numpy as np
import pytest

from scene_analysis.config import DepthConfig, PercentileClipConfig
from scene_analysis.depth.base import DepthEstimator, DummyDepthEstimator, create_depth_estimator
from scene_analysis.types import DepthResult


class StubDepthAnythingEstimator(DepthEstimator):
    def __init__(
        self,
        model_name: str,
        device: str = "auto",
        cache_dir: str | None = None,
        use_fp16: bool = False,
        compile_model: bool = False,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self.use_fp16 = use_fp16
        self.compile_model = compile_model

    def predict(self, image: np.ndarray) -> DepthResult:
        return DepthResult(depth_map=None, confidence_map=None, metadata={"status": "stub"})


def test_create_depth_estimator_returns_depth_anything_estimator(monkeypatch: pytest.MonkeyPatch) -> None:
    def _stub_builder(depth_config: DepthConfig) -> DepthEstimator:
        return StubDepthAnythingEstimator(
            model_name=depth_config.model,
            device=depth_config.device,
            cache_dir=depth_config.cache_dir,
            use_fp16=depth_config.use_fp16,
            compile_model=depth_config.compile_model,
        )

    monkeypatch.setattr("scene_analysis.depth.base._create_depth_anything_v2_estimator", _stub_builder)

    config = DepthConfig(
        enabled=True,
        provider="depth_anything_v2",
        model="depth-anything/Depth-Anything-V2-Small-hf",
        device="cpu",
        cache_dir=None,
        use_fp16=False,
        compile_model=False,
        save_raw_depth_npy=True,
        save_depth_colormap=True,
        normalize_depth_for_viz=True,
        clip_percentiles=PercentileClipConfig(min=2.0, max=98.0),
    )

    estimator = create_depth_estimator(config)

    assert isinstance(estimator, StubDepthAnythingEstimator)
    assert estimator.model_name == "depth-anything/Depth-Anything-V2-Small-hf"
    assert estimator.device == "cpu"


def test_disabled_depth_returns_dummy_estimator() -> None:
    config = DepthConfig(
        enabled=False,
        provider="depth_anything_v2",
        model="depth-anything/Depth-Anything-V2-Small-hf",
        device="cpu",
        cache_dir=None,
        use_fp16=False,
        compile_model=False,
        save_raw_depth_npy=True,
        save_depth_colormap=True,
        normalize_depth_for_viz=True,
        clip_percentiles=PercentileClipConfig(min=2.0, max=98.0),
    )

    estimator = create_depth_estimator(config)

    assert isinstance(estimator, DummyDepthEstimator)


def test_unknown_provider_raises_value_error() -> None:
    config = DepthConfig.model_construct(
        enabled=True,
        provider="unknown_provider",
        model="depth-anything/Depth-Anything-V2-Small-hf",
        device="auto",
        cache_dir=None,
        use_fp16=False,
        compile_model=False,
        save_raw_depth_npy=True,
        save_depth_colormap=True,
        normalize_depth_for_viz=True,
        clip_percentiles=PercentileClipConfig(min=2.0, max=98.0),
    )

    with pytest.raises(ValueError):
        create_depth_estimator(config)
