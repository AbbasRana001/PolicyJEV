"""Token-length statistics of the train partition for choosing a max sequence length."""

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from policyjev.config import LABELS, REPO_ROOT, load_config
from policyjev.data.eda import LENGTH_GROUPS
from policyjev.data.load import load_train
from policyjev.data.split import load_split_ids

LIMITS = (64, 128, 256, 512)


def count_tokens(texts: list[str], tokenizer, batch_size: int = 1000) -> np.ndarray:
    """Return the untruncated token count (with special tokens) of each text."""
    counts: list[int] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        ids = tokenizer(batch, add_special_tokens=True, truncation=False)["input_ids"]
        counts.extend(len(item) for item in ids)
    return np.array(counts, dtype=int)


def length_report(df: pd.DataFrame, lengths: np.ndarray) -> pd.DataFrame:
    """Summarize token-length percentiles and over-limit shares per label group."""
    lengths = np.asarray(lengths)
    rows = []
    for group in LENGTH_GROUPS:
        if group == "all":
            mask = np.ones(len(df), dtype=bool)
        elif group == "no_label":
            mask = (df[LABELS].sum(axis=1) == 0).to_numpy()
        else:
            mask = (df[group] == 1).to_numpy()

        values = lengths[mask]
        row = {"group": group, "comments": int(mask.sum())}
        for p in (50, 90, 99):
            row[f"tokens_p{p}"] = (
                round(float(np.percentile(values, p)), 1) if len(values) else float("nan")
            )
        row["tokens_max"] = int(values.max()) if len(values) else 0
        for limit in LIMITS:
            row[f"share_over_{limit}"] = (
                round(float((values > limit).mean()), 4) if len(values) else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    """Tokenize the train partition and save the token-length report."""
    config = load_config()
    tokenizer = AutoTokenizer.from_pretrained(config["distilbert"]["checkpoint"])

    full_train = load_train()
    train_ids, _ = load_split_ids(REPO_ROOT / "data" / "processed")
    train_df = full_train[full_train["id"].isin(train_ids)].reset_index(drop=True)

    lengths = count_tokens(train_df["comment_text"].tolist(), tokenizer)
    report = length_report(train_df, lengths)
    print(report.to_string(index=False))

    metrics_dir = REPO_ROOT / "artifacts" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    report.to_csv(metrics_dir / "token_lengths.csv", index=False)


if __name__ == "__main__":
    main()
