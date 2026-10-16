"""Exploratory validation-only analysis of how much headroom JEV has (never run on test).

(a) Prompt ensemble: the mean of the v1, v2 and v3 probabilities is training-free.
(b) Stacking: cross-validated logistic regression on JEV probabilities uses validation
    labels through cross-validation and is therefore not a JEV-only result.
"""

import numpy as np
import pandas as pd

from policyjev.config import LABELS, REPO_ROOT, load_config
from policyjev.evaluation.bootstrap import paired_bootstrap, percentile_interval
from policyjev.evaluation.splits import load_split_scores, validation_ids_and_labels
from policyjev.evaluation.stack import stacked_matrix, stacked_out_of_fold  # noqa: F401
from policyjev.evaluation.validation_report import average_precision

VERSIONS = ["v1", "v2", "v3"]
REFERENCES = ["v2", "logreg_word", "distilbert"]
KINDS = {
    "v1": "single",
    "v2": "single",
    "v3": "single",
    "ensemble_mean": "training-free",
    "stack_v2": "stacked (uses validation labels)",
    "stack_all": "stacked (uses validation labels)",
    "logreg_word": "reference",
    "distilbert": "reference",
}
HEADER = (
    "EXPLORATORY analysis on validation only; not a headline result and never run on test.\n"
    "ensemble_mean is training-free (mean of the v1, v2 and v3 probabilities).\n"
    "stack_v2 and stack_all fit a 5-fold cross-validated logistic regression on JEV "
    "probabilities (6 or 18 features); they use validation labels through cross-validation "
    "and are not JEV-only results.\n"
    "Intervals are 95% paired-bootstrap intervals; differences are paired on the same rows."
)


def build_candidates(
    y: np.ndarray,
    probs_by_version: dict[str, np.ndarray],
    baselines: dict[str, np.ndarray],
    seed: int,
) -> dict[str, np.ndarray]:
    """All scored candidates keyed by name, each of shape (n, n_labels)."""
    candidates = {v: probs_by_version[v] for v in VERSIONS}
    candidates["ensemble_mean"] = np.mean([probs_by_version[v] for v in VERSIONS], axis=0)
    candidates["stack_v2"] = stacked_matrix(probs_by_version["v2"], y, seed)
    candidates["stack_all"] = stacked_matrix(
        np.hstack([probs_by_version[v] for v in VERSIONS]), y, seed
    )
    candidates.update(baselines)
    return candidates


def build_headroom_table(
    y: np.ndarray,
    candidates: dict[str, np.ndarray],
    n_resamples: int,
    seed: int,
    confidence: float,
) -> pd.DataFrame:
    """AP with intervals per candidate and label, plus paired differences to the references."""
    names = list(candidates)
    n_labels = y.shape[1]

    def ap_matrix(idx: np.ndarray) -> np.ndarray:
        return np.array(
            [
                [average_precision(y[idx, j], candidates[name][idx, j]) for j in range(n_labels)]
                for name in names
            ]
        )

    def with_mean(values: np.ndarray) -> np.ndarray:
        return np.concatenate([values, np.nanmean(values, axis=-1, keepdims=True)], axis=-1)

    point = with_mean(ap_matrix(np.arange(len(y))))
    samples = paired_bootstrap(len(y), lambda idx: ap_matrix(idx).ravel(), n_resamples, seed)
    samples = with_mean(samples.reshape(n_resamples, len(names), n_labels))
    low, high = percentile_interval(samples, confidence)
    position = {name: i for i, name in enumerate(names)}

    rows = []
    for name in names:
        i = position[name]
        diffs = {}
        for ref in REFERENCES:
            if ref == name:
                nan = np.full(n_labels + 1, np.nan)
                diffs[ref] = (nan, nan, nan)
                continue
            draws = samples[:, i] - samples[:, position[ref]]
            d_low, d_high = percentile_interval(draws, confidence)
            diffs[ref] = (point[i] - point[position[ref]], d_low, d_high)
        for j, label in enumerate(LABELS + ["mean"]):
            row = {
                "candidate": name,
                "kind": KINDS.get(name, "other"),
                "label": label,
                "ap": point[i, j],
                "ap_lo": low[i, j],
                "ap_hi": high[i, j],
            }
            for ref in REFERENCES:
                value, d_low, d_high = diffs[ref]
                row[f"diff_vs_{ref}"] = value[j]
                row[f"diff_vs_{ref}_lo"] = d_low[j]
                row[f"diff_vs_{ref}_hi"] = d_high[j]
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    """Write artifacts/metrics/jev_headroom_validation.csv and print it."""
    config = load_config()
    ids, y = validation_ids_and_labels()
    probs = {v: load_split_scores("jev", "validation", ids, v) for v in VERSIONS}
    baselines = {
        system: load_split_scores(system, "validation", ids, config["jev"]["primary_question_version"])
        for system in ("logreg_word", "distilbert")
    }

    candidates = build_candidates(y, probs, baselines, config["seed"])
    table = build_headroom_table(
        y,
        candidates,
        config["bootstrap"]["n_resamples"],
        config["seed"],
        config["bootstrap"]["confidence_level"],
    )
    metrics_dir = REPO_ROOT / "artifacts" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(metrics_dir / "jev_headroom_validation.csv", index=False)

    print(HEADER)
    print()
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
