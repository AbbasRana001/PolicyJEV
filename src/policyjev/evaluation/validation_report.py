"""Full-validation comparison of the baselines and JEV with paired bootstrap intervals."""

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from policyjev.config import LABELS, REPO_ROOT, load_config
from policyjev.data.load import load_train
from policyjev.data.split import load_split_ids
from policyjev.evaluation.bootstrap import paired_bootstrap, percentile_interval
from policyjev.evaluation.thresholds import load_system_probabilities

SYSTEMS = ["logreg_word", "distilbert", "jev"]
DIFFERENCES = [("jev", "logreg_word"), ("jev", "distilbert")]

HEADER = (
    "Full-validation comparison (all validation rows, unweighted).\n"
    "AP uses the same implementation for every system. JEV probabilities take 101 "
    "discrete values, so AP involves more ties than for the trained models.\n"
    "Intervals are 95% percentile intervals from a paired bootstrap: each resample "
    "draws rows once and applies them to every system.\n"
    "F1 is measured at the validation-fitted threshold and is in-sample."
)


def average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Average precision for one label; NaN when there are no positives."""
    if y_true.sum() == 0:
        return float("nan")
    return float(average_precision_score(y_true, scores))


def _ap_matrix(
    y: np.ndarray,
    probs: dict[str, np.ndarray],
    idx: np.ndarray,
    systems: list[str] | None = None,
) -> np.ndarray:
    """AP per system and label on the given rows: shape (n_systems, n_labels)."""
    return np.array(
        [
            [average_precision(y[idx, j], probs[system][idx, j]) for j in range(y.shape[1])]
            for system in (systems or SYSTEMS)
        ]
    )


def _with_mean(values: np.ndarray) -> np.ndarray:
    """Append the across-label mean (NaN-aware) as a last column."""
    return np.concatenate([values, np.nanmean(values, axis=-1, keepdims=True)], axis=-1)


def build_comparison(
    y: np.ndarray,
    probs: dict[str, np.ndarray],
    f1_in_sample: dict[str, np.ndarray],
    n_resamples: int,
    seed: int,
    confidence: float,
    calibrated: dict[str, np.ndarray] | None = None,
    f1_prefix: str = "f1_in_sample",
    systems: list[str] | None = None,
    differences: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    """One row per label plus a mean row, with AP intervals and paired differences.

    AP is computed on the raw scores. Brier uses the calibrated probabilities when given
    (raw Brier is then added as brier_raw_<system>); F1 columns are named f1_prefix_<system>.
    Systems and contrasts default to the three primary systems.
    """
    systems = systems or SYSTEMS
    differences = differences if differences is not None else DIFFERENCES
    n_labels = y.shape[1]
    n_systems = len(systems)

    point_ap = _with_mean(_ap_matrix(y, probs, np.arange(len(y)), systems))

    def statistic(idx: np.ndarray) -> np.ndarray:
        return _ap_matrix(y, probs, idx, systems).ravel()

    samples = paired_bootstrap(len(y), statistic, n_resamples, seed)
    samples = _with_mean(samples.reshape(n_resamples, n_systems, n_labels))
    position = {system: i for i, system in enumerate(systems)}

    ap_low, ap_high = percentile_interval(samples, confidence)

    prevalence = y.mean(axis=0)
    rows = pd.DataFrame(
        {
            "label": LABELS + ["mean"],
            "positives": list(y.sum(axis=0)) + [float("nan")],
            "prevalence": list(prevalence) + [float(prevalence.mean())],
        }
    )
    for system in systems:
        i = position[system]
        rows[f"ap_{system}"] = point_ap[i]
        rows[f"ap_{system}_lo"] = ap_low[i]
        rows[f"ap_{system}_hi"] = ap_high[i]
    for a, b in differences:
        diff_samples = samples[:, position[a]] - samples[:, position[b]]
        low, high = percentile_interval(diff_samples, confidence)
        rows[f"diff_ap_{a}_minus_{b}"] = point_ap[position[a]] - point_ap[position[b]]
        rows[f"diff_ap_{a}_minus_{b}_lo"] = low
        rows[f"diff_ap_{a}_minus_{b}_hi"] = high
    brier_source = calibrated if calibrated is not None else probs
    for system in systems:
        brier = ((brier_source[system] - y) ** 2).mean(axis=0)
        rows[f"brier_{system}"] = list(brier) + [float(brier.mean())]
    if calibrated is not None:
        for system in systems:
            raw_brier = ((probs[system] - y) ** 2).mean(axis=0)
            rows[f"brier_raw_{system}"] = list(raw_brier) + [float(raw_brier.mean())]
    for system in systems:
        f1 = np.asarray(f1_in_sample[system], dtype=float)
        rows[f"{f1_prefix}_{system}"] = list(f1) + [float(f1.mean())]
    return rows


def main() -> None:
    """Write artifacts/metrics/validation_comparison.csv and print it."""
    config = load_config()
    metrics_dir = REPO_ROOT / "artifacts" / "metrics"
    thresholds_path = metrics_dir / "thresholds_full_validation.csv"
    if not thresholds_path.exists():
        raise SystemExit(
            "run python -m policyjev.evaluation.thresholds --split full_validation first"
        )
    thresholds = pd.read_csv(thresholds_path)

    _, validation_ids = load_split_ids(REPO_ROOT / "data" / "processed")
    y = load_train().set_index("id").loc[validation_ids, LABELS].to_numpy(dtype=int)
    jev_version = config["jev"]["primary_question_version"]
    probs = {
        system: load_system_probabilities(system, validation_ids, jev_version).to_numpy(
            dtype=float
        )
        for system in SYSTEMS
    }
    f1_in_sample = {
        system: thresholds[thresholds["system"] == system]
        .set_index("label")
        .loc[LABELS, "f1"]
        .to_numpy()
        for system in SYSTEMS
    }

    comparison = build_comparison(
        y,
        probs,
        f1_in_sample,
        config["bootstrap"]["n_resamples"],
        config["seed"],
        config["bootstrap"]["confidence_level"],
    )
    metrics_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(metrics_dir / "validation_comparison.csv", index=False)

    print(HEADER)
    print(f"JEV question set version: {jev_version}")
    print()
    print(comparison.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
