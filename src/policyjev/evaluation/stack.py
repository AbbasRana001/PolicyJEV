"""jev_stack: a supervised logistic regression on JEV's six probabilities (secondary system).

It is fit on validation labels, so it is never reported as JEV. The saved model is applied
once to the test scores of the primary question set.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from policyjev.config import LABELS, REPO_ROOT, load_config
from policyjev.data.load import load_train
from policyjev.data.split import load_split_ids
from policyjev.evaluation.thresholds import load_system_probabilities

STACK_SYSTEM = "jev_stack"
N_FOLDS = 5


def stack_path(base: Path | None = None) -> Path:
    """Where the saved stack model lives."""
    folder = base if base is not None else REPO_ROOT / "artifacts" / "calibrators"
    return folder / "jev_stack.json"


def stacked_out_of_fold(
    features: np.ndarray, y_col: np.ndarray, seed: int, n_splits: int = N_FOLDS
) -> np.ndarray:
    """Out-of-fold probabilities of a standardized logistic regression for one label."""
    result = np.empty(len(y_col))
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, test_idx in folds.split(features, y_col):
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        model.fit(features[train_idx], y_col[train_idx])
        result[test_idx] = model.predict_proba(features[test_idx])[:, 1]
    return result


def stacked_matrix(features: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    """Out-of-fold stacked probabilities for every label, shape (n, n_labels)."""
    return np.column_stack(
        [stacked_out_of_fold(features, y[:, j], seed) for j in range(y.shape[1])]
    )


def fit_stack(features: np.ndarray, y: np.ndarray) -> dict[str, dict]:
    """Fit one standardized logistic regression per label on all given rows."""
    models = {}
    for j, label in enumerate(LABELS):
        y_col = y[:, j]
        if y_col.min() == y_col.max():
            models[label] = {"constant": float(y_col[0])}
            continue
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        model.fit(features, y_col)
        scaler, logistic = model.named_steps["standardscaler"], model.named_steps["logisticregression"]
        models[label] = {
            "mean": [float(f"{v:.8g}") for v in scaler.mean_],
            "scale": [float(f"{v:.8g}") for v in scaler.scale_],
            "coef": [float(f"{v:.8g}") for v in logistic.coef_[0]],
            "intercept": float(f"{logistic.intercept_[0]:.8g}"),
        }
    return models


def save_stack(path: Path, models: dict[str, dict], question_version: str) -> None:
    """Write the stack coefficients as readable JSON, one line per label."""
    entries = ",\n".join(
        f"    {json.dumps(label)}: {json.dumps(models[label], separators=(',', ':'))}"
        for label in LABELS
    )
    text = (
        "{\n"
        '  "system": "jev_stack",\n'
        f'  "question_version": {json.dumps(question_version)},\n'
        f'  "features": {json.dumps(LABELS)},\n'
        '  "fitted_on": "all validation rows",\n'
        '  "labels": {\n' + entries + "\n  }\n}\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_stack(path: Path) -> dict:
    """Read the saved stack; raises a clear error if it does not exist."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run python -m policyjev.evaluation.stack first")
    return json.loads(path.read_text(encoding="utf-8"))


def apply_stack(table: pd.DataFrame, saved: dict) -> pd.DataFrame:
    """Apply the saved stack to a table of the six JEV probabilities (id kept if present)."""
    features = table[LABELS].to_numpy(dtype=float)
    result = table.copy()
    for label in LABELS:
        model = saved["labels"][label]
        if "constant" in model:
            result[label] = model["constant"]
            continue
        z = (features - np.array(model["mean"])) / np.array(model["scale"])
        result[label] = expit(z @ np.array(model["coef"]) + model["intercept"])
    return result


def main() -> None:
    """Fit the stack on all validation rows and write its out-of-fold validation predictions."""
    config = load_config()
    jev_version = config["jev"]["primary_question_version"]
    _, validation_ids = load_split_ids(REPO_ROOT / "data" / "processed")
    y = load_train().set_index("id").loc[validation_ids, LABELS].to_numpy(dtype=int)
    features = load_system_probabilities("jev", validation_ids, jev_version).to_numpy(float)

    save_stack(stack_path(), fit_stack(features, y), jev_version)
    oof = stacked_matrix(features, y, config["seed"])

    predictions = pd.DataFrame(oof, columns=LABELS)
    predictions.insert(0, "id", validation_ids)
    predictions_dir = REPO_ROOT / "artifacts" / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(predictions_dir / "jev_stack_validation.csv", index=False, float_format="%.8g")

    print(f"jev_stack fit on all validation rows from JEV question set {jev_version}")
    print(f"saved {stack_path()} and out-of-fold validation predictions")


if __name__ == "__main__":
    main()
