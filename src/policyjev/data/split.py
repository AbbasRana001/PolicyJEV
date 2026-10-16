"""Stratified train/validation split over comment ids."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from policyjev.config import LABELS, REPO_ROOT, load_config
from policyjev.data.load import load_test, load_train

PRIORITY = ["threat", "identity_hate", "severe_toxic", "insult", "obscene", "toxic"]


def stratum_key(df: pd.DataFrame) -> pd.Series:
    """Return, per row, the first PRIORITY label set to 1, else 'none'."""

    def pick(row: pd.Series) -> str:
        for label in PRIORITY:
            if row[label] == 1:
                return label
        return "none"

    return df.apply(pick, axis=1)


def make_split(
    df: pd.DataFrame, validation_fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    """Split df's ids into stratified (train_ids, validation_ids)."""
    train_ids, validation_ids = train_test_split(
        df["id"],
        stratify=stratum_key(df),
        test_size=validation_fraction,
        random_state=seed,
    )
    return sorted(train_ids.tolist()), sorted(validation_ids.tolist())


def save_split_ids(
    train_ids: list[str], validation_ids: list[str], out_dir: Path
) -> None:
    """Write split id files to out_dir, never overwriting existing ones."""
    train_path = out_dir / "split_train_ids.csv"
    validation_path = out_dir / "split_validation_ids.csv"
    if train_path.exists() or validation_path.exists():
        raise FileExistsError("split id files already exist")

    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": train_ids}).to_csv(train_path, index=False)
    pd.DataFrame({"id": validation_ids}).to_csv(validation_path, index=False)


def load_split_ids(out_dir: Path) -> tuple[list[str], list[str]]:
    """Read back the split id files written by save_split_ids."""
    train_ids = pd.read_csv(
        out_dir / "split_train_ids.csv", dtype={"id": str}
    )["id"].tolist()
    validation_ids = pd.read_csv(
        out_dir / "split_validation_ids.csv", dtype={"id": str}
    )["id"].tolist()
    return train_ids, validation_ids


def summarize(
    df: pd.DataFrame, train_ids: list[str], validation_ids: list[str]
) -> pd.DataFrame:
    """Per-label positive counts and rates for the train and validation splits."""
    train_df = df[df["id"].isin(train_ids)]
    validation_df = df[df["id"].isin(validation_ids)]

    rows = []
    for label in LABELS:
        train_positives = int(train_df[label].sum())
        validation_positives = int(validation_df[label].sum())
        rows.append(
            {
                "label": label,
                "train_positives": train_positives,
                "train_rate": train_positives / len(train_df),
                "validation_positives": validation_positives,
                "validation_rate": validation_positives / len(validation_df),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Build and save the stratified train/validation split."""
    config = load_config()
    train_df = load_train()
    test_df = load_test()

    train_ids, validation_ids = make_split(
        train_df, config["split"]["validation_fraction"], config["seed"]
    )

    out_dir = REPO_ROOT / "data" / "processed"
    save_split_ids(train_ids, validation_ids, out_dir)

    summary = summarize(train_df, train_ids, validation_ids)

    print(f"train.csv rows: {len(train_df)}")
    print(f"labeled test rows: {len(test_df)}")
    print(f"test rows dropped as unlabeled: {test_df.attrs['n_dropped_unlabeled']}")
    print(f"train split size: {len(train_ids)}")
    print(f"validation split size: {len(validation_ids)}")
    print(summary.to_string(index=False))

    metrics_dir = REPO_ROOT / "artifacts" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(metrics_dir / "split_summary.csv", index=False)


if __name__ == "__main__":
    main()
