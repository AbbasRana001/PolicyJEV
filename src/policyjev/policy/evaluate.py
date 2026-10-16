"""One-time evaluation of the moderation policy on the test split, as fixed by Amendment 3.

Thresholds come only from artifacts/policy/policy_thresholds.csv. The run refuses to start
unless the protocol-frozen and policy-frozen git tags exist and the frozen files it uses match
the freeze manifest, and it refuses to overwrite existing outputs. No comment text is read.
"""

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from policyjev.config import LABELS, REPO_ROOT, load_config
from policyjev.evaluation.bootstrap import paired_bootstrap, percentile_interval
from policyjev.freeze_manifest import verify_files
from policyjev.jev.runner import PROTOCOL_TAG, tag_exists
from policyjev.policy.engine import ACTION, ALLOW, HUMAN_REVIEW, decide_frame
from policyjev.policy.thresholds import (
    PRECISION_GRID,
    PRIMARY_PRECISION,
    PRIMARY_RECALL,
    RECALL_GRID,
    SYSTEMS,
    thresholds_for,
)

POLICY_TAG = "policy-frozen"
RARE_LABELS = ["threat", "severe_toxic", "identity_hate"]
METRICS = ["m1_review_share", "m2_action_precision", "m3_miss_rate", "m4_rare_miss_rate"]
CONTRAST_REFERENCES = ["logreg_word", "distilbert"]
POLICY_THRESHOLDS = REPO_ROOT / "artifacts" / "policy" / "policy_thresholds.csv"
OUTPUT_NAMES = {
    "summary": "policy_test_summary.csv",
    "per_label": "policy_test_per_label.csv",
    "contrasts": "policy_test_contrasts.csv",
    "grid": "policy_test_grid.csv",
}


@dataclass
class DecisionData:
    """Per-comment decision codes (0 ALLOW, 1 HUMAN REVIEW, 2 ACTION) and ACTION hits."""

    codes: np.ndarray
    action_hit: np.ndarray


def decision_data(calibrated: np.ndarray, y: np.ndarray, thresholds: pd.DataFrame) -> DecisionData:
    """Run the engine over all comments; an ACTION hit means a label that reached HIGH is positive."""
    frame = pd.DataFrame(calibrated, columns=LABELS)
    result = decide_frame(frame, thresholds)
    code_of = {ALLOW: 0, HUMAN_REVIEW: 1, ACTION: 2}
    codes = result["decision"].map(code_of).to_numpy()
    position = {label: j for j, label in enumerate(LABELS)}
    hit = np.zeros(len(frame), dtype=bool)
    for i in np.flatnonzero(codes == 2):
        hit[i] = any(y[i, position[label]] == 1 for label in result["triggering_labels"].iloc[i])
    return DecisionData(codes, hit)


def policy_metrics(
    codes: np.ndarray, action_hit: np.ndarray, any_positive: np.ndarray, rare_positive: np.ndarray
) -> dict[str, float]:
    """M1 to M4 and the descriptive ACTION share and coverage (NaN when undefined)."""
    n = len(codes)
    n_action = int((codes == 2).sum())
    n_any = int(any_positive.sum())
    n_rare = int(rare_positive.sum())
    return {
        "m1_review_share": float((codes == 1).sum() / n),
        "m2_action_precision": float(action_hit[codes == 2].sum() / n_action) if n_action else float("nan"),
        "m3_miss_rate": float((codes[any_positive] == 0).sum() / n_any) if n_any else float("nan"),
        "m4_rare_miss_rate": float((codes[rare_positive] == 0).sum() / n_rare) if n_rare else float("nan"),
        "action_share": float(n_action / n),
        "action_coverage": float((codes[any_positive] == 2).sum() / n_any) if n_any else float("nan"),
    }


def positive_masks(y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Comments positive for any label, and for any of the rare labels."""
    rare = [LABELS.index(label) for label in RARE_LABELS]
    return y.sum(axis=1) > 0, y[:, rare].sum(axis=1) > 0


def summary_table(
    data: Mapping[str, DecisionData], y: np.ndarray, n_failed_rows: int
) -> pd.DataFrame:
    """Decision counts and shares plus M1 to M4 per system at the primary targets."""
    any_positive, rare_positive = positive_masks(y)
    rows = []
    for system, d in data.items():
        metrics = policy_metrics(d.codes, d.action_hit, any_positive, rare_positive)
        n = len(d.codes)
        rows.append(
            {
                "system": system,
                "n_rows": n,
                "n_failed_rows": n_failed_rows,
                "n_allow": int((d.codes == 0).sum()),
                "n_review": int((d.codes == 1).sum()),
                "n_action": int((d.codes == 2).sum()),
                "allow_share": float((d.codes == 0).mean()),
                **metrics,
                "any_positive_comments": int(any_positive.sum()),
                "rare_positive_comments": int(rare_positive.sum()),
            }
        )
    return pd.DataFrame(rows)


def per_label_table(
    calibrated: Mapping[str, np.ndarray], y: np.ndarray, thresholds: Mapping[str, pd.DataFrame]
) -> pd.DataFrame:
    """Per label: review, ACTION and flagged shares, ACTION precision and miss rate."""
    rows = []
    for system, probabilities in calibrated.items():
        table = thresholds[system].set_index("label")
        for j, label in enumerate(LABELS):
            p = probabilities[:, j]
            low, high = float(table.loc[label, "low"]), float(table.loc[label, "high"])
            positives = y[:, j] == 1
            flagged_action = p >= high
            below_low = p < low
            n_positives = int(positives.sum())
            rows.append(
                {
                    "system": system,
                    "label": label,
                    "low": low,
                    "high": high,
                    "action_enabled": bool(np.isfinite(high)),
                    "positives": n_positives,
                    "review_share_label": float(((p >= low) & (p < high)).mean()),
                    "action_share_label": float(flagged_action.mean()),
                    "flagged_share_label": float((p >= low).mean()),
                    "action_flagged": int(flagged_action.sum()),
                    "action_precision_label": (
                        float(positives[flagged_action].mean()) if flagged_action.any() else float("nan")
                    ),
                    "missed_positives": int((positives & below_low).sum()),
                    "miss_rate_label": (
                        float((positives & below_low).sum() / n_positives) if n_positives else float("nan")
                    ),
                }
            )
    return pd.DataFrame(rows)


def contrast_table(
    data: Mapping[str, DecisionData],
    y: np.ndarray,
    n_resamples: int,
    seed: int,
    confidence: float,
) -> pd.DataFrame:
    """JEV minus each baseline on M1 to M4 with paired bootstrap percentile intervals.

    Resamples in which a quantity is undefined for either system are left out of that interval.
    """
    any_positive, rare_positive = positive_masks(y)
    systems = list(data)

    def evaluate(idx: np.ndarray) -> np.ndarray:
        return np.array(
            [
                [
                    policy_metrics(
                        data[s].codes[idx], data[s].action_hit[idx], any_positive[idx], rare_positive[idx]
                    )[m]
                    for m in METRICS
                ]
                for s in systems
            ]
        ).ravel()

    shape = (len(systems), len(METRICS))
    everything = np.arange(len(y))
    point = evaluate(everything).reshape(shape)
    samples = paired_bootstrap(len(y), evaluate, n_resamples, seed).reshape((n_resamples,) + shape)
    position = {s: i for i, s in enumerate(systems)}

    rows = []
    for reference in CONTRAST_REFERENCES:
        for m, metric in enumerate(METRICS):
            draws = samples[:, position["jev"], m] - samples[:, position[reference], m]
            valid = ~np.isnan(draws)
            used = int(valid.sum())
            if used:
                low, high = (float(v[0]) for v in percentile_interval(draws[valid].reshape(-1, 1), confidence))
            else:
                low = high = float("nan")
            rows.append(
                {
                    "contrast": f"jev - {reference}",
                    "metric": metric,
                    "difference": float(point[position["jev"], m] - point[position[reference], m]),
                    "ci_low": low,
                    "ci_high": high,
                    "resamples_used": used,
                    "resamples_left_out": int(n_resamples - used),
                }
            )
    return pd.DataFrame(rows)


def grid_table(
    calibrated: Mapping[str, np.ndarray], y: np.ndarray, policy_table: pd.DataFrame
) -> pd.DataFrame:
    """Point estimates of M1 to M4, ACTION share and coverage at all nine grid points."""
    any_positive, rare_positive = positive_masks(y)
    rows = []
    for system, probabilities in calibrated.items():
        for recall_target in RECALL_GRID:
            for precision_target in PRECISION_GRID:
                thresholds = thresholds_for(policy_table, system, recall_target, precision_target)
                d = decision_data(probabilities, y, thresholds)
                metrics = policy_metrics(d.codes, d.action_hit, any_positive, rare_positive)
                rows.append(
                    {
                        "system": system,
                        "recall_target": recall_target,
                        "precision_target": precision_target,
                        "is_primary": bool(
                            np.isclose(recall_target, PRIMARY_RECALL)
                            and np.isclose(precision_target, PRIMARY_PRECISION)
                        ),
                        **metrics,
                        "n_action": int((d.codes == 2).sum()),
                    }
                )
    return pd.DataFrame(rows)


def evaluate_policy(
    calibrated: Mapping[str, np.ndarray],
    y: np.ndarray,
    policy_table: pd.DataFrame,
    n_resamples: int,
    seed: int,
    confidence: float,
    n_failed_rows: int = 0,
) -> dict[str, pd.DataFrame]:
    """All four result tables from calibrated test probabilities and the frozen thresholds."""
    thresholds = {
        s: thresholds_for(policy_table, s, PRIMARY_RECALL, PRIMARY_PRECISION) for s in calibrated
    }
    data = {s: decision_data(calibrated[s], y, thresholds[s]) for s in calibrated}
    return {
        "summary": summary_table(data, y, n_failed_rows),
        "per_label": per_label_table(calibrated, y, thresholds),
        "contrasts": contrast_table(data, y, n_resamples, seed, confidence),
        "grid": grid_table(calibrated, y, policy_table),
    }


def check_gates(output_paths: list[Path]) -> None:
    """Refuse to run without both freeze tags, matching manifest entries, or with outputs present."""
    for tag in (PROTOCOL_TAG, POLICY_TAG):
        if not tag_exists(tag):
            raise SystemExit(f"refusing to evaluate the policy: git tag '{tag}' does not exist")
    verify_files([POLICY_THRESHOLDS])
    existing = [str(p) for p in output_paths if p.exists()]
    if existing:
        raise SystemExit("refusing to overwrite existing output: " + ", ".join(existing))


def main() -> None:
    """Evaluate the policy once on test and write the policy_test_*.csv files."""
    argparse.ArgumentParser().parse_args()
    metrics_dir = REPO_ROOT / "artifacts" / "metrics"
    outputs = {key: metrics_dir / name for key, name in OUTPUT_NAMES.items()}
    check_gates(list(outputs.values()))

    from policyjev.evaluation.splits import load_split_scores_bundle

    config = load_config()
    policy_table = pd.read_csv(POLICY_THRESHOLDS)
    data = load_split_scores_bundle("test", SYSTEMS)
    n_rows = len(data.ids)

    results = evaluate_policy(
        {s: data.calibrated[s] for s in SYSTEMS},
        data.y,
        policy_table,
        config["bootstrap"]["n_resamples"],
        config["seed"],
        config["bootstrap"]["confidence_level"],
        n_failed_rows=0,
    )
    metrics_dir.mkdir(parents=True, exist_ok=True)
    for key, path in outputs.items():
        results[key].to_csv(path, index=False)

    print(
        f"test rows used (identical across systems): {n_rows}; failed rows: 0 "
        "(the prediction loaders require every test row for every system)"
    )
    print(results["summary"].round(4).to_string(index=False))
    print(results["contrasts"].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
