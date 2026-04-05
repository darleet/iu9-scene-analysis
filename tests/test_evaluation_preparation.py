from __future__ import annotations

import numpy as np

from scene_analysis.evaluation.preparation import (
    build_valid_label_mask,
    flatten_scores_and_labels,
    normalize_prediction_map,
)


def test_build_valid_label_mask_excludes_ignore_pixels() -> None:
    gt_mask = np.array(
        [
            [0, 1, 255],
            [1, 0, 255],
        ],
        dtype=np.uint8,
    )

    valid_mask, positive_mask = build_valid_label_mask(
        gt_mask=gt_mask,
        obstacle_values=[1],
        background_values=[0],
        ignore_values=[255],
    )

    assert valid_mask.tolist() == [
        [True, True, False],
        [True, True, False],
    ]
    assert positive_mask.tolist() == [
        [False, True, False],
        [True, False, False],
    ]


def test_flatten_scores_and_labels_returns_expected_sizes() -> None:
    prediction = normalize_prediction_map(
        np.array(
            [
                [0.1, 0.9, 0.7],
                [0.8, 0.2, 0.0],
            ],
            dtype=np.float32,
        )
    )
    valid_mask = np.array(
        [
            [True, True, False],
            [True, True, False],
        ],
        dtype=bool,
    )
    positive_mask = np.array(
        [
            [False, True, False],
            [True, False, False],
        ],
        dtype=bool,
    )

    scores, labels = flatten_scores_and_labels(prediction, valid_mask, positive_mask)

    assert scores.shape == (4,)
    assert labels.shape == (4,)
    assert labels.tolist() == [0, 1, 1, 0]
