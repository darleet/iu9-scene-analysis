from __future__ import annotations

import numpy as np

from scene_analysis.evaluation.visualization import _downsample_curve_for_plot


def test_downsample_curve_for_plot_limits_number_of_points() -> None:
    precision = np.linspace(1.0, 0.0, num=10000, dtype=np.float32)
    recall = np.linspace(0.0, 1.0, num=10000, dtype=np.float32)

    plot_precision, plot_recall = _downsample_curve_for_plot(
        precision=precision,
        recall=recall,
        max_points=500,
    )

    assert plot_precision.shape == plot_recall.shape
    assert plot_precision.size <= 500
    assert np.isclose(plot_precision[0], precision[0])
    assert np.isclose(plot_precision[-1], precision[-1])
    assert np.isclose(plot_recall[0], recall[0])
    assert np.isclose(plot_recall[-1], recall[-1])
