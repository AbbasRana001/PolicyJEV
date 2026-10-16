"""Exploratory analysis of the train partition (the test set is never loaded)."""

import numpy as np
import pandas as pd

from policyjev.config import LABELS, REPO_ROOT, load_config
from policyjev.data.load import load_train
from policyjev.data.split import load_split_ids

LENGTH_GROUPS = [
    "all",
    "no_label",
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]


def _normalize(texts: pd.Series) -> pd.Series:
    """Lower-case each text and collapse whitespace."""
    return texts.astype(str).map(lambda t: " ".join(t.lower().split()))


def label_count_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Count comments by how many labels they are positive for (0 to 6)."""
    n_labels = df[LABELS].sum(axis=1)
    counts = n_labels.value_counts().reindex(range(7), fill_value=0)
    return pd.DataFrame(
        {
            "n_labels": range(7),
            "comments": counts.to_numpy(),
            "share": counts.to_numpy() / len(df),
        }
    )


def cooccurrence(df: pd.DataFrame) -> pd.DataFrame:
    """Return the label-by-label co-occurrence count matrix."""
    values = df[LABELS].to_numpy(dtype=int)
    return pd.DataFrame(values.T @ values, index=LABELS, columns=LABELS)


def length_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Summarize word and character length percentiles per label group."""
    texts = df["comment_text"].astype(str)
    words = texts.map(lambda t: len(t.split()))
    chars = texts.map(len)

    rows = []
    for group in LENGTH_GROUPS:
        if group == "all":
            mask = pd.Series(True, index=df.index)
        elif group == "no_label":
            mask = df[LABELS].sum(axis=1) == 0
        else:
            mask = df[group] == 1

        w = words[mask].to_numpy()
        c = chars[mask].to_numpy()
        row = {"group": group, "comments": int(mask.sum())}
        for prefix, values in (("words", w), ("chars", c)):
            for p in (50, 90, 99):
                row[f"{prefix}_p{p}"] = (
                    round(float(np.percentile(values, p)), 1)
                    if len(values)
                    else float("nan")
                )
            row[f"{prefix}_max"] = int(values.max()) if len(values) else 0
        rows.append(row)

    columns = [
        "group",
        "comments",
        "words_p50",
        "words_p90",
        "words_p99",
        "words_max",
        "chars_p50",
        "chars_p90",
        "chars_p99",
        "chars_max",
    ]
    return pd.DataFrame(rows)[columns]


def quality_report(df: pd.DataFrame, validation_df: pd.DataFrame) -> pd.DataFrame:
    """Count data-quality issues in the train partition and its overlap with validation."""
    keys = _normalize(df["comment_text"])
    group_sizes = keys.map(keys.value_counts())
    duplicate_rows = int((group_sizes > 1).sum())

    label_vectors = df[LABELS].astype(int).astype(str).agg("".join, axis=1)
    per_key = pd.DataFrame({"key": keys, "vector": label_vectors})
    per_key = per_key[group_sizes > 1]
    conflicting = int((per_key.groupby("key")["vector"].nunique() > 1).sum())

    validation_keys = _normalize(validation_df["comment_text"])

    checks = {
        "empty_or_whitespace_comments": int(
            (df["comment_text"].astype(str).str.strip() == "").sum()
        ),
        "duplicate_texts_in_train": duplicate_rows,
        "duplicate_groups_with_conflicting_labels": conflicting,
        "severe_toxic_without_toxic": int(
            ((df["severe_toxic"] == 1) & (df["toxic"] == 0)).sum()
        ),
        "validation_rows_with_text_also_in_train": int(
            validation_keys.isin(set(keys)).sum()
        ),
    }
    return pd.DataFrame({"check": list(checks), "count": list(checks.values())})


def main() -> None:
    """Run the train-partition analyses, save the tables and print them."""
    load_config()
    full_train = load_train()
    train_ids, validation_ids = load_split_ids(REPO_ROOT / "data" / "processed")
    train_df = full_train[full_train["id"].isin(train_ids)].reset_index(drop=True)
    validation_df = full_train[full_train["id"].isin(validation_ids)].reset_index(
        drop=True
    )

    results = {
        "eda_label_counts.csv": (label_count_distribution(train_df), False),
        "eda_cooccurrence.csv": (cooccurrence(train_df), True),
        "eda_length_summary.csv": (length_summary(train_df), False),
        "eda_quality.csv": (quality_report(train_df, validation_df), False),
    }

    metrics_dir = REPO_ROOT / "artifacts" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    for filename, (table, keep_index) in results.items():
        table.to_csv(metrics_dir / filename, index=keep_index)
        print(filename)
        print(table.to_string(index=keep_index))
        print()


if __name__ == "__main__":
    main()
