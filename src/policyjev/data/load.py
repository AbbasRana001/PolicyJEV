"""Loaders for the raw Jigsaw Toxic Comment Classification CSVs."""

from pathlib import Path

import pandas as pd

from policyjev.config import LABELS, REPO_ROOT

RAW_DIR = REPO_ROOT / "data" / "raw"


def load_train(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load train.csv with columns id, comment_text, then LABELS as int."""
    df = pd.read_csv(
        raw_dir / "train.csv", dtype={"id": str}, keep_default_na=False
    )
    df[LABELS] = df[LABELS].astype(int)
    return df[["id", "comment_text"] + LABELS]


def load_test(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Load test.csv merged with test_labels.csv, dropping unlabeled rows."""
    comments = pd.read_csv(
        raw_dir / "test.csv", dtype={"id": str}, keep_default_na=False
    )
    labels = pd.read_csv(
        raw_dir / "test_labels.csv", dtype={"id": str}, keep_default_na=False
    )
    if set(comments["id"]) != set(labels["id"]):
        raise ValueError("test.csv and test_labels.csv have different id sets")

    merged = comments.merge(labels, on="id")
    unlabeled_mask = (merged[LABELS] == -1).any(axis=1)
    n_dropped = int(unlabeled_mask.sum())
    merged = merged.loc[~unlabeled_mask].copy()

    merged[LABELS] = merged[LABELS].astype(int)
    assert merged[LABELS].isin([0, 1]).all().all()

    merged = merged[["id", "comment_text"] + LABELS]
    merged.attrs["n_dropped_unlabeled"] = n_dropped
    return merged
