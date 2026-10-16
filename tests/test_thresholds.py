import numpy as np
import pandas as pd

from policyjev.evaluation.thresholds import (
    choose_threshold,
    fit_thresholds,
    weighted_prf,
)

GRID = np.round(np.arange(1, 100) * 0.01, 10)


def test_weighted_prf_matches_hand_computation():
    y_true = np.array([1, 1, 0, 0, 1])
    y_pred = np.array([1, 0, 1, 0, 1])
    weights = np.array([2.0, 1.0, 3.0, 1.0, 4.0])

    precision, recall, f1 = weighted_prf(y_true, y_pred, weights)

    assert abs(precision - 6 / 9) < 1e-12
    assert abs(recall - 6 / 7) < 1e-12
    assert abs(f1 - 2 * (6 / 9) * (6 / 7) / (6 / 9 + 6 / 7)) < 1e-12


def test_weighted_prf_undefined_ratios_are_zero():
    precision, recall, f1 = weighted_prf(
        np.array([0, 0]), np.array([0, 0]), np.array([1.0, 1.0])
    )
    assert (precision, recall, f1) == (0.0, 0.0, 0.0)


def test_separable_data_gives_perfect_f1_and_separating_threshold():
    y = np.array([0, 0, 0, 1, 1, 1])
    probs = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    weights = np.ones(6)

    threshold, f1 = choose_threshold(y, probs, weights, GRID)

    assert f1 == 1.0
    assert (probs[y == 0] < threshold).all()
    assert (probs[y == 1] >= threshold).all()


def test_weighting_changes_the_best_threshold():
    y = np.array([1, 0, 1, 0])
    probs = np.array([0.9, 0.6, 0.4, 0.2])

    unweighted, _ = choose_threshold(y, probs, np.ones(4), GRID)
    heavy_negative, _ = choose_threshold(
        y, probs, np.array([1.0, 10.0, 1.0, 1.0]), GRID
    )

    assert unweighted != heavy_negative
    assert heavy_negative > 0.6


def test_all_negative_label_returns_zero_f1():
    y = np.zeros(5, dtype=int)
    probs = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

    threshold, f1 = choose_threshold(y, probs, np.ones(5), GRID)
    assert f1 == 0.0
    assert 0.0 < threshold < 1.0

    table = fit_thresholds(
        pd.DataFrame({"toxic": probs}),
        pd.DataFrame({"toxic": y}),
        np.ones(5),
        ["toxic"],
        GRID,
    )
    assert table.loc[0, "f1"] == 0.0
    assert table.loc[0, "positives"] == 0
