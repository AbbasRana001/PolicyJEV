"""Paired bootstrap: every resample draws rows once and applies them to all systems."""

import warnings
from collections.abc import Callable, Iterator

import numpy as np


def resample_indices(n_rows: int, n_resamples: int, seed: int) -> Iterator[np.ndarray]:
    """Yield n_resamples seeded index arrays drawn with replacement."""
    rng = np.random.default_rng(seed)
    for _ in range(n_resamples):
        yield rng.integers(0, n_rows, size=n_rows)


def paired_bootstrap(
    n_rows: int,
    statistic: Callable[[np.ndarray], np.ndarray],
    n_resamples: int = 1000,
    seed: int = 42,
) -> np.ndarray:
    """Evaluate statistic on each resample; returns an array (n_resamples, n_values).

    The statistic receives one index array per resample and should evaluate every
    system on exactly those rows, so differences between systems are paired.
    """
    return np.array(
        [
            np.asarray(statistic(idx), dtype=float).ravel()
            for idx in resample_indices(n_rows, n_resamples, seed)
        ]
    )


def percentile_interval(
    samples: np.ndarray, confidence: float = 0.95
) -> tuple[np.ndarray, np.ndarray]:
    """Percentile interval per column; NaN resamples are ignored."""
    tail = (1.0 - confidence) / 2.0 * 100.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lower = np.nanpercentile(samples, tail, axis=0)
        upper = np.nanpercentile(samples, 100.0 - tail, axis=0)
    return lower, upper
