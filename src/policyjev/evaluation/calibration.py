"""Post-hoc probability calibration (Platt and isotonic), chosen per system on validation."""

import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from policyjev.config import LABELS, REPO_ROOT, load_config
from policyjev.data.load import load_train
from policyjev.data.split import load_split_ids
from policyjev.evaluation.thresholds import load_system_probabilities

SYSTEMS = ["logreg_word", "distilbert", "jev"]
METHODS = ["platt", "isotonic"]
N_FOLDS = 5
N_BINS = 5
EPSILON = 1e-6


def clipped_logit(scores: np.ndarray) -> np.ndarray:
    """Logit of scores clipped away from 0 and 1."""
    p = np.clip(np.asarray(scores, dtype=float), EPSILON, 1.0 - EPSILON)
    return np.log(p / (1.0 - p))


def _round(value: float) -> float:
    """Keep eight significant digits so saved calibrators stay readable."""
    return float(f"{float(value):.8g}")


def fit_calibrator(method: str, scores: np.ndarray, y: np.ndarray) -> dict:
    """Fit a calibrator for one label; returns a JSON-serializable description."""
    y = np.asarray(y, dtype=int)
    if y.min() == y.max():
        return {"method": "constant", "value": float(y[0])}
    if method == "platt":
        model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(clipped_logit(scores).reshape(-1, 1), y)
        return {
            "method": "platt",
            "a": _round(model.coef_[0, 0]),
            "b": _round(model.intercept_[0]),
        }
    if method == "isotonic":
        model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        model.fit(np.asarray(scores, dtype=float), y)
        return {
            "method": "isotonic",
            "x": [_round(v) for v in model.X_thresholds_],
            "y": [_round(v) for v in model.y_thresholds_],
        }
    raise ValueError(f"unknown calibration method {method!r}")


def apply_calibrator(calibrator: dict, scores: np.ndarray) -> np.ndarray:
    """Apply a fitted calibrator description to raw scores."""
    scores = np.asarray(scores, dtype=float)
    method = calibrator["method"]
    if method == "constant":
        return np.full(scores.shape, calibrator["value"])
    if method == "platt":
        return expit(calibrator["a"] * clipped_logit(scores) + calibrator["b"])
    if method == "isotonic":
        return np.interp(scores, calibrator["x"], calibrator["y"])
    raise ValueError(f"unknown calibration method {method!r}")


def out_of_fold(
    method: str, scores: np.ndarray, y: np.ndarray, seed: int, n_splits: int = N_FOLDS
) -> np.ndarray:
    """Out-of-fold calibrated probabilities for one label (stratified, seeded folds)."""
    scores = np.asarray(scores, dtype=float)
    y = np.asarray(y, dtype=int)
    result = np.empty(len(scores))
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, test_idx in folds.split(scores.reshape(-1, 1), y):
        calibrator = fit_calibrator(method, scores[train_idx], y[train_idx])
        result[test_idx] = apply_calibrator(calibrator, scores[test_idx])
    return result


def out_of_fold_matrix(
    method: str, raw: np.ndarray, y: np.ndarray, seed: int
) -> np.ndarray:
    """Out-of-fold calibrated probabilities for all labels, shape (n, n_labels)."""
    return np.column_stack(
        [out_of_fold(method, raw[:, j], y[:, j], seed) for j in range(raw.shape[1])]
    )


def brier(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Brier score per column."""
    return ((np.asarray(probs, dtype=float) - y) ** 2).mean(axis=0)


def choose_method(mean_brier: dict[str, float]) -> str:
    """Method with the lowest mean out-of-fold Brier; ties go to Platt."""
    return min(METHODS, key=lambda m: (mean_brier[m], METHODS.index(m)))


def reliability_table(
    system: str, method: str, probs: np.ndarray, y: np.ndarray, n_bins: int = N_BINS
) -> pd.DataFrame:
    """Equal-count reliability bins per label: counts, mean predicted, observed rate."""
    rows = []
    for j, label in enumerate(LABELS):
        order = np.argsort(probs[:, j], kind="stable")
        for number, bin_idx in enumerate(np.array_split(order, n_bins), start=1):
            rows.append(
                {
                    "system": system,
                    "label": label,
                    "method": method,
                    "bin": number,
                    "n": len(bin_idx),
                    "mean_predicted": float(probs[bin_idx, j].mean()),
                    "observed_rate": float(y[bin_idx, j].mean()),
                }
            )
    return pd.DataFrame(rows)


def evaluate_system(
    system: str, raw: np.ndarray, y: np.ndarray, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame, str, np.ndarray]:
    """Return (Brier table, reliability table, chosen method, out-of-fold probabilities of it)."""
    raw_brier = brier(raw, y)
    oof = {m: out_of_fold_matrix(m, raw, y, seed) for m in METHODS}
    method_brier = {m: brier(oof[m], y) for m in METHODS}
    chosen = choose_method({m: float(method_brier[m].mean()) for m in METHODS})

    table = pd.DataFrame(
        {
            "system": system,
            "label": LABELS + ["mean"],
            "brier_raw": list(raw_brier) + [float(raw_brier.mean())],
            **{
                f"brier_{m}_oof": list(method_brier[m]) + [float(method_brier[m].mean())]
                for m in METHODS
            },
            "chosen_method": chosen,
        }
    )
    reliability = reliability_table(system, chosen, oof[chosen], y)
    return table, reliability, chosen, oof[chosen]


def fit_calibrators(method: str, raw: np.ndarray, y: np.ndarray) -> dict[str, dict]:
    """Fit the chosen calibrator per label on all given rows."""
    return {label: fit_calibrator(method, raw[:, j], y[:, j]) for j, label in enumerate(LABELS)}


def calibrator_path(system: str, jev_version: str | None, base: Path | None = None) -> Path:
    """Where the saved calibrators of a system live."""
    folder = base if base is not None else REPO_ROOT / "artifacts" / "calibrators"
    name = f"jev_{jev_version}" if system == "jev" else system
    return folder / f"{name}.json"


def save_calibrators(
    path: Path,
    system: str,
    method: str,
    calibrators: dict[str, dict],
    question_version: str | None = None,
) -> None:
    """Write calibrators as readable JSON, one line per label."""
    header = [
        f'  "system": {json.dumps(system)},',
        f'  "method": {json.dumps(method)},',
        f'  "question_version": {json.dumps(question_version)},',
        '  "fitted_on": "all validation rows",',
        '  "labels": {',
    ]
    entries = [
        f"    {json.dumps(label)}: {json.dumps(calibrators[label], separators=(',', ':'))}"
        for label in LABELS
    ]
    body = ",\n".join(entries)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{\n" + "\n".join(header) + "\n" + body + "\n  }\n}\n", encoding="utf-8")


def load_calibrators(path: Path) -> dict:
    """Read saved calibrators; raises a clear error if the file is missing."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run python -m policyjev.evaluation.calibration first"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def apply_calibrators(table: pd.DataFrame, saved: dict) -> pd.DataFrame:
    """Apply saved per-label calibrators to a score table (id column kept if present)."""
    result = table.copy()
    for label in LABELS:
        result[label] = apply_calibrator(saved["labels"][label], table[label].to_numpy())
    return result


def record_calibration_methods(config_path: Path, methods: dict[str, str]) -> None:
    """Replace (or add) the calibration block in the config file, keeping other text."""
    text = config_path.read_text(encoding="utf-8")
    text = re.sub(r"(?m)^calibration:\n(?:[ \t]+.*\n|\n)*", "", text).rstrip("\n") + "\n"
    block = "calibration:\n  method:\n" + "".join(
        f"    {system}: {methods[system]}\n" for system in SYSTEMS if system in methods
    )
    config_path.write_text(text + block, encoding="utf-8")


def configured_methods() -> dict[str, str]:
    """Calibration methods recorded in the project config."""
    try:
        methods = load_config()["calibration"]["method"]
    except KeyError:
        raise SystemExit(
            "calibration.method is not set in configs/project.yaml; "
            "run python -m policyjev.evaluation.calibration first"
        ) from None
    missing = [s for s in SYSTEMS if s not in methods]
    if missing:
        raise SystemExit(f"calibration.method has no entry for: {', '.join(missing)}")
    return methods


def main() -> None:
    """Evaluate calibrators on validation, save the chosen ones and record the choice."""
    config = load_config()
    seed = config["seed"]
    jev_version = config["jev"]["primary_question_version"]

    _, validation_ids = load_split_ids(REPO_ROOT / "data" / "processed")
    y = load_train().set_index("id").loc[validation_ids, LABELS].to_numpy(dtype=int)

    tables, reliabilities, methods = [], [], {}
    for system in SYSTEMS:
        raw = load_system_probabilities(system, validation_ids, jev_version).to_numpy(float)
        table, reliability, chosen, _ = evaluate_system(system, raw, y, seed)
        tables.append(table)
        reliabilities.append(reliability)
        methods[system] = chosen
        save_calibrators(
            calibrator_path(system, jev_version),
            system,
            chosen,
            fit_calibrators(chosen, raw, y),
            jev_version if system == "jev" else None,
        )

    metrics_dir = REPO_ROOT / "artifacts" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    result = pd.concat(tables, ignore_index=True)
    result.to_csv(metrics_dir / "calibration_validation.csv", index=False)
    pd.concat(reliabilities, ignore_index=True).to_csv(
        metrics_dir / "calibration_reliability_validation.csv", index=False
    )
    record_calibration_methods(REPO_ROOT / "configs" / "project.yaml", methods)

    print(f"JEV question set version: {jev_version}")
    print("Brier on validation: raw and 5-fold out-of-fold (mean over labels)")
    print(result[result["label"] == "mean"].round(5).to_string(index=False))
    print("chosen methods recorded in configs/project.yaml:", methods)


if __name__ == "__main__":
    main()
