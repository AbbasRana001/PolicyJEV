import numpy as np
import pandas as pd
import pytest

from policyjev.config import LABELS
from policyjev.data.split import load_split_ids, make_split, save_split_ids

LABEL_PROBABILITIES = {
    "toxic": 0.10,
    "severe_toxic": 0.01,
    "obscene": 0.05,
    "threat": 0.01,
    "insult": 0.05,
    "identity_hate": 0.01,
}


def build_synthetic_df(seed: int, n_rows: int = 4000) -> pd.DataFrame:
    """Build a synthetic labeled dataframe with independent label probabilities."""
    rng = np.random.default_rng(seed)
    data = {
        "id": [f"id{i:05d}" for i in range(1, n_rows + 1)],
        "comment_text": ["placeholder"] * n_rows,
    }
    for label in LABELS:
        data[label] = (rng.random(n_rows) < LABEL_PROBABILITIES[label]).astype(int)
    return pd.DataFrame(data)


def test_split_is_disjoint_complete_and_near_target_fraction():
    df = build_synthetic_df(seed=0)
    train_ids, validation_ids = make_split(df, validation_fraction=0.15, seed=0)

    assert set(train_ids).isdisjoint(set(validation_ids))
    assert set(train_ids) | set(validation_ids) == set(df["id"])
    assert abs(len(validation_ids) / len(df) - 0.15) <= 0.01


def test_split_is_deterministic_per_seed_and_varies_across_seeds():
    df = build_synthetic_df(seed=0)

    train_a, validation_a = make_split(df, validation_fraction=0.15, seed=0)
    train_b, validation_b = make_split(df, validation_fraction=0.15, seed=0)
    assert train_a == train_b
    assert validation_a == validation_b

    train_c, validation_c = make_split(df, validation_fraction=0.15, seed=1)
    assert train_c != train_a or validation_c != validation_a


def test_every_label_has_positives_in_both_splits():
    df = build_synthetic_df(seed=0)
    train_ids, validation_ids = make_split(df, validation_fraction=0.15, seed=0)
    train_df = df[df["id"].isin(train_ids)]
    validation_df = df[df["id"].isin(validation_ids)]

    for label in LABELS:
        assert train_df[label].sum() > 0
        assert validation_df[label].sum() > 0


def test_save_then_load_roundtrips_and_refuses_overwrite(tmp_path):
    df = build_synthetic_df(seed=0)
    train_ids, validation_ids = make_split(df, validation_fraction=0.15, seed=0)

    save_split_ids(train_ids, validation_ids, tmp_path)
    loaded_train_ids, loaded_validation_ids = load_split_ids(tmp_path)

    assert loaded_train_ids == train_ids
    assert loaded_validation_ids == validation_ids

    with pytest.raises(FileExistsError):
        save_split_ids(train_ids, validation_ids, tmp_path)
