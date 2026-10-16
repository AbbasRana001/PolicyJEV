"""Compare JEV question-set versions on the validation split and apply the selection rule."""

import re
from pathlib import Path

import numpy as np
import pandas as pd

from policyjev.config import LABELS, REPO_ROOT, load_config
from policyjev.data.load import load_train
from policyjev.data.split import load_split_ids
from policyjev.evaluation.bootstrap import paired_bootstrap, percentile_interval
from policyjev.evaluation.thresholds import load_system_probabilities
from policyjev.evaluation.validation_report import average_precision

BASELINE_VERSION = "v1"
SELECTION_MARGIN = 0.005
_FILE_PATTERN = re.compile(r"^jev_validation_(.+)\.csv$")


def discover_versions(predictions_dir: Path) -> list[str]:
    """Question set versions that have a validation score file, sorted."""
    versions = []
    for path in predictions_dir.glob("jev_validation_*.csv"):
        match = _FILE_PATTERN.match(path.name)
        if match:
            versions.append(match.group(1))
    return sorted(versions)


def select_question_set(
    mean_ap: dict[str, float],
    baseline: str = BASELINE_VERSION,
    margin: float = SELECTION_MARGIN,
) -> tuple[str, str]:
    """Pre-specified rule: keep the baseline unless an alternative beats it by the margin."""
    alternatives = {v: m for v, m in mean_ap.items() if v != baseline}
    if not alternatives:
        return baseline, f"no alternative question set scored; {baseline} kept"
    best = max(sorted(alternatives), key=lambda v: alternatives[v])
    gain = alternatives[best] - mean_ap[baseline]
    summary = (
        f"best alternative {best} has mean AP {alternatives[best]:.4f} against "
        f"{baseline} {mean_ap[baseline]:.4f} (difference {gain:+.4f}, margin {margin})"
    )
    if gain < margin:
        return baseline, f"{summary}: {baseline} kept"
    return best, f"{summary}: {best} selected"


def build_question_set_comparison(
    y: np.ndarray,
    probs_by_version: dict[str, np.ndarray],
    n_resamples: int,
    seed: int,
    confidence: float,
    baseline: str = BASELINE_VERSION,
) -> pd.DataFrame:
    """Per version and label (plus mean): AP, interval, and paired difference to baseline."""
    versions = sorted(probs_by_version)
    if baseline not in versions:
        raise ValueError(f"no scores for baseline question set {baseline}")
    n_labels = y.shape[1]

    def ap_matrix(idx: np.ndarray) -> np.ndarray:
        return np.array(
            [
                [average_precision(y[idx, j], probs_by_version[v][idx, j]) for j in range(n_labels)]
                for v in versions
            ]
        )

    def with_mean(values: np.ndarray) -> np.ndarray:
        return np.concatenate([values, np.nanmean(values, axis=-1, keepdims=True)], axis=-1)

    point = with_mean(ap_matrix(np.arange(len(y))))
    samples = paired_bootstrap(len(y), lambda idx: ap_matrix(idx).ravel(), n_resamples, seed)
    samples = with_mean(samples.reshape(n_resamples, len(versions), n_labels))
    low, high = percentile_interval(samples, confidence)
    base = versions.index(baseline)

    rows = []
    for i, version in enumerate(versions):
        diff_samples = samples[:, i] - samples[:, base]
        diff_low, diff_high = percentile_interval(diff_samples, confidence)
        for j, label in enumerate(LABELS + ["mean"]):
            is_baseline = version == baseline
            rows.append(
                {
                    "version": version,
                    "label": label,
                    "ap": point[i, j],
                    "ap_lo": low[i, j],
                    "ap_hi": high[i, j],
                    f"diff_vs_{baseline}": float("nan") if is_baseline else point[i, j] - point[base, j],
                    f"diff_vs_{baseline}_lo": float("nan") if is_baseline else diff_low[j],
                    f"diff_vs_{baseline}_hi": float("nan") if is_baseline else diff_high[j],
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    """Write artifacts/metrics/question_set_comparison.csv and print the selection outcome."""
    config = load_config()
    predictions_dir = REPO_ROOT / "artifacts" / "predictions"
    versions = discover_versions(predictions_dir)
    if BASELINE_VERSION not in versions:
        raise SystemExit(f"no validation scores for {BASELINE_VERSION}; run policyjev.jev.scores first")

    _, validation_ids = load_split_ids(REPO_ROOT / "data" / "processed")
    y = load_train().set_index("id").loc[validation_ids, LABELS].to_numpy(dtype=int)
    probs = {
        v: load_system_probabilities("jev", validation_ids, v).to_numpy(dtype=float)
        for v in versions
    }

    table = build_question_set_comparison(
        y,
        probs,
        config["bootstrap"]["n_resamples"],
        config["seed"],
        config["bootstrap"]["confidence_level"],
    )
    metrics_dir = REPO_ROOT / "artifacts" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(metrics_dir / "question_set_comparison.csv", index=False)

    print(f"question sets scored on validation: {', '.join(versions)}")
    print("95% paired-bootstrap intervals; differences are paired against " + BASELINE_VERSION)
    print(table.round(4).to_string(index=False))
    print()
    mean_ap = table[table["label"] == "mean"].set_index("version")["ap"].to_dict()
    selected, message = select_question_set(mean_ap)
    print(f"selection rule: {message}")
    print(f"selected question set: {selected}")


if __name__ == "__main__":
    main()
