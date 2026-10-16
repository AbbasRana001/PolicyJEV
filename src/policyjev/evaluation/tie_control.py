"""Control for ties: baseline AP as is and after rounding probabilities to two decimals."""

import numpy as np
import pandas as pd

from policyjev.config import LABELS, REPO_ROOT
from policyjev.data.load import load_train
from policyjev.data.split import load_split_ids
from policyjev.evaluation.thresholds import load_system_probabilities
from policyjev.evaluation.validation_report import average_precision

CONTROL_SYSTEMS = ["logreg_word", "distilbert"]


def tie_control_table(
    y: np.ndarray, probs_by_system: dict[str, np.ndarray], decimals: int = 2
) -> pd.DataFrame:
    """AP per system and label (plus mean) before and after rounding the probabilities."""
    rows = []
    for system, probs in probs_by_system.items():
        rounded = np.round(probs, decimals)
        original = [average_precision(y[:, j], probs[:, j]) for j in range(y.shape[1])]
        coarse = [average_precision(y[:, j], rounded[:, j]) for j in range(y.shape[1])]
        original.append(float(np.nanmean(original)))
        coarse.append(float(np.nanmean(coarse)))
        for label, ap, ap_rounded in zip(LABELS + ["mean"], original, coarse):
            rows.append(
                {
                    "system": system,
                    "label": label,
                    "ap": ap,
                    "ap_rounded": ap_rounded,
                    "difference": ap_rounded - ap,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    """Write artifacts/metrics/tie_control.csv and print it."""
    _, validation_ids = load_split_ids(REPO_ROOT / "data" / "processed")
    y = load_train().set_index("id").loc[validation_ids, LABELS].to_numpy(dtype=int)
    probs = {
        system: load_system_probabilities(system, validation_ids).to_numpy(dtype=float)
        for system in CONTROL_SYSTEMS
    }

    table = tie_control_table(y, probs)
    metrics_dir = REPO_ROOT / "artifacts" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(metrics_dir / "tie_control.csv", index=False)

    print("Baseline AP on validation, as is and with probabilities rounded to 2 decimals")
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
