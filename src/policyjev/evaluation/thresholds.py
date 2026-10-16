"""Per-label F1-optimal decision thresholds fit on validation rows."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from policyjev.config import LABELS, REPO_ROOT, load_config
from policyjev.data.load import load_train
from policyjev.data.split import load_split_ids

FULL_VALIDATION_SYSTEMS = ["logreg_word", "distilbert", "jev"]


def weighted_prf(
    y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray
) -> tuple[float, float, float]:
    """Weighted precision, recall and F1; undefined ratios are 0.0."""
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    weights = np.asarray(weights, dtype=float)

    tp = float(weights[y_true & y_pred].sum())
    fp = float(weights[~y_true & y_pred].sum())
    fn = float(weights[y_true & ~y_pred].sum())

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f1


def choose_threshold(
    y_true: np.ndarray, probs: np.ndarray, weights: np.ndarray, grid: np.ndarray
) -> tuple[float, float]:
    """Pick the F1-maximizing threshold; ties resolve to the lower median."""
    probs = np.asarray(probs, dtype=float)
    f1_values = np.array(
        [weighted_prf(y_true, probs >= t, weights)[2] for t in grid]
    )
    best = f1_values.max()
    tied = np.sort(np.asarray(grid)[f1_values >= best - 1e-9])
    threshold = float(tied[(len(tied) - 1) // 2])
    return threshold, float(best)


def fit_thresholds(
    probs_df: pd.DataFrame,
    labels_df: pd.DataFrame,
    weights: np.ndarray,
    labels: list[str],
    grid: np.ndarray,
) -> pd.DataFrame:
    """Fit one threshold per label and report weighted metrics at that threshold."""
    weights = np.asarray(weights, dtype=float)
    rows = []
    for label in labels:
        y = labels_df[label].to_numpy(dtype=int)
        p = probs_df[label].to_numpy(dtype=float)
        threshold, _ = choose_threshold(y, p, weights, grid)
        precision, recall, f1 = weighted_prf(y, p >= threshold, weights)
        rows.append(
            {
                "label": label,
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "positives": int(y.sum()),
                "weighted_prevalence": float((weights * y).sum() / weights.sum()),
                "weighted_average_precision": (
                    float(average_precision_score(y, p, sample_weight=weights))
                    if y.sum() > 0
                    else float("nan")
                ),
            }
        )
    return pd.DataFrame(rows)


def fit_full_validation(
    probs_by_system: dict[str, pd.DataFrame], labels_df: pd.DataFrame, grid: np.ndarray
) -> pd.DataFrame:
    """Fit unweighted per-label thresholds for each system on all validation rows."""
    weights = np.ones(len(labels_df))
    tables = []
    for system, probs_df in probs_by_system.items():
        table = fit_thresholds(
            probs_df.reset_index(drop=True),
            labels_df.reset_index(drop=True),
            weights,
            LABELS,
            grid,
        )
        table.insert(0, "system", system)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def validation_prediction_path(system: str, jev_version: str | None = None) -> Path:
    """Path of a system's validation predictions; JEV files carry the question set version."""
    predictions_dir = REPO_ROOT / "artifacts" / "predictions"
    if system == "jev":
        if jev_version is None:
            raise ValueError("a question set version is required for the jev system")
        return predictions_dir / f"jev_validation_{jev_version}.csv"
    return predictions_dir / f"{system}_validation.csv"


def load_system_probabilities(
    system: str, ids: list[str], jev_version: str | None = None
) -> pd.DataFrame:
    """Read a system's validation predictions, aligned to ids (all ids required)."""
    path = validation_prediction_path(system, jev_version)
    predictions = pd.read_csv(path, dtype={"id": str}).set_index("id")
    if set(predictions.index) != set(ids):
        raise ValueError(f"{path.name} does not cover exactly the validation split")
    return predictions.loc[ids, LABELS]


def run_full_validation() -> None:
    """Fit thresholds on all validation rows for the baselines and JEV."""
    config = load_config()
    step = config["thresholds"]["grid_step"]
    grid = np.round(np.arange(1, round(1 / step)) * step, 10)

    _, validation_ids = load_split_ids(REPO_ROOT / "data" / "processed")
    labels_df = load_train().set_index("id").loc[validation_ids, LABELS]
    jev_version = config["jev"]["primary_question_version"]
    probs_by_system = {
        system: load_system_probabilities(system, validation_ids, jev_version)
        for system in FULL_VALIDATION_SYSTEMS
    }

    results = fit_full_validation(probs_by_system, labels_df, grid)
    for system in FULL_VALIDATION_SYSTEMS:
        print(system)
        print(
            results[results["system"] == system]
            .drop(columns="system")
            .round(4)
            .to_string(index=False)
        )
        print()

    metrics_dir = REPO_ROOT / "artifacts" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(metrics_dir / "thresholds_full_validation.csv", index=False)


def main() -> None:
    """Fit thresholds on all validation rows for the baselines and JEV."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["full_validation"], default="full_validation")
    parser.parse_args()
    run_full_validation()


if __name__ == "__main__":
    main()
