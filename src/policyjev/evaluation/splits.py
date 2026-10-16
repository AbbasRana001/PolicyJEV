"""Split-aware data access: validation now, test only behind the protocol-frozen tag."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from policyjev.config import LABELS, REPO_ROOT, load_config
from policyjev.data.load import load_test, load_train
from policyjev.data.split import load_split_ids
from policyjev.evaluation.calibration import (
    SYSTEMS,
    apply_calibrator,
    calibrator_path,
    configured_methods,
    load_calibrators,
    out_of_fold_matrix,
)
from policyjev.evaluation.stack import (
    STACK_SYSTEM,
    apply_stack,
    load_stack,
    stack_path,
    stacked_matrix,
)
from policyjev.evaluation.thresholds import fit_thresholds, weighted_prf
from policyjev.freeze_manifest import verify_files
from policyjev.jev.runner import PROTOCOL_TAG, tag_exists

SPLITS = ["validation", "test"]


@dataclass
class SplitScores:
    """Labels, raw scores, calibrated probabilities and thresholds for one split."""

    split: str
    ids: list[str]
    y: np.ndarray
    raw: dict[str, np.ndarray]
    calibrated: dict[str, np.ndarray]
    thresholds: dict[str, np.ndarray]
    validation_f1: dict[str, np.ndarray]


def require_split_access(split: str) -> None:
    """Refuse to touch test predictions or labels until the protocol-frozen tag exists."""
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}")
    if split == "test" and not tag_exists(PROTOCOL_TAG):
        raise SystemExit(
            f"refusing to read test predictions or labels: git tag '{PROTOCOL_TAG}' does not exist"
        )


def prediction_path(system: str, split: str, jev_version: str) -> Path:
    """Path of a system's prediction file for a split."""
    predictions_dir = REPO_ROOT / "artifacts" / "predictions"
    if split == "validation":
        if system == "jev":
            return predictions_dir / f"jev_validation_{jev_version}.csv"
        return predictions_dir / f"{system}_validation.csv"
    if system == "jev":
        return predictions_dir / f"jev_test_{jev_version}.csv"
    return predictions_dir / f"{system}_test.csv"


def load_labeled_frame(split: str) -> pd.DataFrame:
    """The labeled rows of a split (guarded), with id, comment_text and label columns."""
    require_split_access(split)
    if split == "validation":
        _, validation_ids = load_split_ids(REPO_ROOT / "data" / "processed")
        frame = load_train().set_index("id").loc[validation_ids].reset_index()
        return frame
    return load_test()


def load_split_scores(system: str, split: str, ids: list[str], jev_version: str) -> np.ndarray:
    """A system's raw scores for the ids of a split, shape (n, 6)."""
    require_split_access(split)
    path = prediction_path(system, split, jev_version)
    predictions = pd.read_csv(path, dtype={"id": str}).set_index("id")
    if set(predictions.index) != set(ids):
        raise ValueError(f"{path.name} does not cover exactly the {split} rows")
    return predictions.loc[ids, LABELS].to_numpy(dtype=float)


def validation_thresholds(
    calibrated: np.ndarray, y: np.ndarray, grid_step: float
) -> tuple[np.ndarray, np.ndarray]:
    """Per-label F1-optimal thresholds on calibrated validation scores, and their F1."""
    grid = np.round(np.arange(1, round(1 / grid_step)) * grid_step, 10)
    table = fit_thresholds(
        pd.DataFrame(calibrated, columns=LABELS),
        pd.DataFrame(y, columns=LABELS),
        np.ones(len(y)),
        LABELS,
        grid,
    )
    return table["threshold"].to_numpy(), table["f1"].to_numpy()


def f1_at_thresholds(calibrated: np.ndarray, y: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Unweighted F1 per label at fixed thresholds."""
    return np.array(
        [
            weighted_prf(y[:, j], calibrated[:, j] >= thresholds[j], np.ones(len(y)))[2]
            for j in range(y.shape[1])
        ]
    )


def thresholds_table_path() -> Path:
    """The frozen table of validation-fitted thresholds."""
    return REPO_ROOT / "artifacts" / "metrics" / "validation_thresholds.csv"


def write_thresholds_table(
    thresholds: dict[str, np.ndarray], f1: dict[str, np.ndarray], path: Path | None = None
) -> Path:
    """Write per-system, per-label thresholds (and their in-sample F1) as a CSV table."""
    rows = [
        {
            "system": system,
            "label": label,
            "threshold": float(thresholds[system][j]),
            "f1_in_sample": float(f1[system][j]),
        }
        for system in thresholds
        for j, label in enumerate(LABELS)
    ]
    path = path if path is not None else thresholds_table_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False, float_format="%.10g")
    return path


def load_thresholds_table(
    systems: list[str], path: Path | None = None
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Read frozen thresholds and in-sample F1 per label for the given systems."""
    path = path if path is not None else thresholds_table_path()
    table = pd.read_csv(path)
    thresholds, f1 = {}, {}
    for system in systems:
        rows = table[table["system"] == system].set_index("label")
        if set(rows.index) != set(LABELS):
            raise ValueError(f"{path.name} has no complete thresholds for {system}")
        thresholds[system] = rows.loc[LABELS, "threshold"].to_numpy(dtype=float)
        f1[system] = rows.loc[LABELS, "f1_in_sample"].to_numpy(dtype=float)
    return thresholds, f1


def _validation_bundle(
    systems: list[str], methods: dict[str, str] | None, config: dict
) -> SplitScores:
    """Validation scores with seeded out-of-fold calibration and out-of-fold thresholds."""
    seed = config["seed"]
    jev_version = config["jev"]["primary_question_version"]
    step = config["thresholds"]["grid_step"]
    methods = methods or configured_methods()

    validation = load_labeled_frame("validation")
    validation_ids = validation["id"].tolist()
    y_validation = validation[LABELS].to_numpy(dtype=int)

    raw_validation, oof, thresholds, validation_f1 = {}, {}, {}, {}
    for system in systems:
        if system == STACK_SYSTEM:
            continue
        raw_validation[system] = load_split_scores(system, "validation", validation_ids, jev_version)
        oof[system] = out_of_fold_matrix(methods[system], raw_validation[system], y_validation, seed)
        thresholds[system], validation_f1[system] = validation_thresholds(
            oof[system], y_validation, step
        )
    if STACK_SYSTEM in systems:
        jev_validation = raw_validation.get("jev")
        if jev_validation is None:
            jev_validation = load_split_scores("jev", "validation", validation_ids, jev_version)
        stack_oof = stacked_matrix(jev_validation, y_validation, seed)
        raw_validation[STACK_SYSTEM] = stack_oof
        oof[STACK_SYSTEM] = stack_oof
        thresholds[STACK_SYSTEM], validation_f1[STACK_SYSTEM] = validation_thresholds(
            stack_oof, y_validation, step
        )
    return SplitScores(
        "validation", validation_ids, y_validation, raw_validation, oof, thresholds, validation_f1
    )


def _test_bundle(systems: list[str], jev_version: str) -> SplitScores:
    """Test scores calibrated with the frozen calibrators and thresholds (manifest-verified)."""
    frozen_files = [calibrator_path(s, jev_version) for s in systems if s != STACK_SYSTEM]
    if STACK_SYSTEM in systems:
        frozen_files.append(stack_path())
    frozen_files.append(thresholds_table_path())
    verify_files(frozen_files)

    thresholds, validation_f1 = load_thresholds_table(systems)
    test = load_labeled_frame("test")
    test_ids = test["id"].tolist()
    raw_test, calibrated_test = {}, {}
    for system in systems:
        if system == STACK_SYSTEM:
            continue
        raw_test[system] = load_split_scores(system, "test", test_ids, jev_version)
        saved = load_calibrators(calibrator_path(system, jev_version))
        calibrated_test[system] = np.column_stack(
            [apply_calibrator(saved["labels"][label], raw_test[system][:, j]) for j, label in enumerate(LABELS)]
        )
    if STACK_SYSTEM in systems:
        jev_test = raw_test.get("jev")
        if jev_test is None:
            jev_test = load_split_scores("jev", "test", test_ids, jev_version)
        stacked = apply_stack(pd.DataFrame(jev_test, columns=LABELS), load_stack(stack_path()))
        raw_test[STACK_SYSTEM] = stacked.to_numpy(dtype=float)
        calibrated_test[STACK_SYSTEM] = raw_test[STACK_SYSTEM]
    return SplitScores(
        "test",
        test_ids,
        test[LABELS].to_numpy(dtype=int),
        raw_test,
        calibrated_test,
        thresholds,
        validation_f1,
    )


def load_split_scores_bundle(
    split: str, systems: list[str] | None = None, methods: dict[str, str] | None = None
) -> SplitScores:
    """Raw and calibrated scores for a split plus validation-fitted thresholds.

    Validation uses seeded out-of-fold calibration and out-of-fold thresholds. Test applies
    the calibrators and the thresholds table frozen from validation, after verifying them
    against the freeze manifest.
    """
    require_split_access(split)
    config = load_config()
    systems = systems or SYSTEMS
    if split == "validation":
        return _validation_bundle(systems, methods, config)
    return _test_bundle(systems, config["jev"]["primary_question_version"])


def load_texts(split: str, ids: list[str]) -> dict[str, str]:
    """Comment text for ids, read from the raw data (guarded); for printing only."""
    frame = load_labeled_frame(split).set_index("id")["comment_text"]
    return {i: frame.loc[i] for i in ids}


def validation_ids_and_labels() -> tuple[list[str], np.ndarray]:
    """Validation ids and labels, for exploratory analyses that never touch test."""
    frame = load_labeled_frame("validation")
    return frame["id"].tolist(), frame[LABELS].to_numpy(dtype=int)
