import numpy as np
import pytest

from policyjev.evaluation.bootstrap import (
    paired_bootstrap,
    percentile_interval,
    resample_indices,
)


def test_resample_indices_are_seeded_and_in_range():
    first = list(resample_indices(50, 5, seed=1))
    again = list(resample_indices(50, 5, seed=1))
    other = list(resample_indices(50, 5, seed=2))

    assert all((a == b).all() for a, b in zip(first, again))
    assert any((a != b).any() for a, b in zip(first, other))
    assert all(len(idx) == 50 and idx.min() >= 0 and idx.max() < 50 for idx in first)


def test_every_system_sees_the_same_rows_in_each_resample():
    rng = np.random.default_rng(0)
    values = rng.random(200)
    system_a = values
    system_b = values.copy()
    seen: list[np.ndarray] = []

    def statistic(idx: np.ndarray) -> np.ndarray:
        seen.append(idx)
        return np.array([system_a[idx].mean(), system_b[idx].mean()])

    samples = paired_bootstrap(200, statistic, n_resamples=30, seed=3)

    assert samples.shape == (30, 2)
    assert np.array_equal(samples[:, 0], samples[:, 1])
    assert len(seen) == 30
    assert all(len(idx) == 200 for idx in seen)


def test_paired_difference_is_tighter_than_unpaired_spread():
    rng = np.random.default_rng(1)
    base = rng.random(300)
    shifted = base + 0.1

    samples = paired_bootstrap(
        300, lambda idx: np.array([base[idx].mean(), shifted[idx].mean()]), 200, 5
    )
    difference = samples[:, 1] - samples[:, 0]

    assert np.allclose(difference, 0.1)
    assert samples[:, 0].std() > 0


def test_results_are_reproducible_and_seed_dependent():
    data = np.random.default_rng(2).random(100)

    def run(seed: int) -> np.ndarray:
        return paired_bootstrap(100, lambda idx: np.array([data[idx].mean()]), 40, seed)

    assert np.array_equal(run(7), run(7))
    assert not np.array_equal(run(7), run(8))


def test_percentile_interval_orders_bounds_and_ignores_nan():
    samples = np.column_stack([np.arange(101, dtype=float), np.full(101, np.nan)])
    samples[0, 1] = 1.0
    samples[1, 1] = 3.0

    lower, upper = percentile_interval(samples, confidence=0.9)

    assert lower[0] == pytest.approx(5.0)
    assert upper[0] == pytest.approx(95.0)
    assert lower[1] <= upper[1]
    assert not np.isnan(lower[1])
