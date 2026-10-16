import numpy as np
import pandas as pd
import pytest

from policyjev.config import LABELS
from policyjev.jev.pilot import assign_pilot_stratum, build_pilot

LABEL_PROBABILITIES = {
    "toxic": 0.10,
    "severe_toxic": 0.01,
    "obscene": 0.05,
    "threat": 0.01,
    "insult": 0.05,
    "identity_hate": 0.01,
}


def build_synthetic_df(n_rows: int = 8000) -> pd.DataFrame:
    """Synthetic labeled frame with independent label probabilities."""
    rng = np.random.default_rng(0)
    data = {
        "id": [f"id{i:05d}" for i in range(1, n_rows + 1)],
        "comment_text": ["placeholder"] * n_rows,
    }
    for label in LABELS:
        data[label] = (rng.random(n_rows) < LABEL_PROBABILITIES[label]).astype(int)
    return pd.DataFrame(data)


def test_pilot_composition_and_stratum_rules():
    df = build_synthetic_df()
    pilot = build_pilot(df, pilot_size=150, seed=7)

    counts = pilot["stratum"].value_counts().to_dict()
    assert counts == {
        "threat": 20,
        "severe_toxic": 20,
        "identity_hate": 20,
        "common_toxic": 40,
        "clean": 50,
    }
    assert len(pilot) == 150
    assert pilot["id"].is_unique

    labeled = df.set_index("id").loc[pilot["id"]]
    stratum = pilot["stratum"].to_numpy()
    assert (labeled.loc[stratum == "threat", "threat"] == 1).all()
    severe = labeled[stratum == "severe_toxic"]
    assert ((severe["severe_toxic"] == 1) & (severe["threat"] == 0)).all()
    identity = labeled[stratum == "identity_hate"]
    assert (
        (identity["identity_hate"] == 1)
        & (identity["threat"] == 0)
        & (identity["severe_toxic"] == 0)
    ).all()
    common = labeled[stratum == "common_toxic"]
    assert (common[["threat", "severe_toxic", "identity_hate"]] == 0).all().all()
    assert (common[["toxic", "obscene", "insult"]].sum(axis=1) > 0).all()
    assert (labeled[stratum == "clean"][LABELS] == 0).all().all()


def test_pilot_is_reproducible_and_seed_dependent():
    df = build_synthetic_df()
    first = build_pilot(df, pilot_size=150, seed=7)
    again = build_pilot(df, pilot_size=150, seed=7)
    other = build_pilot(df, pilot_size=150, seed=8)

    assert first["id"].tolist() == again["id"].tolist()
    assert first["id"].tolist() != other["id"].tolist()


def test_pilot_built_from_train_part_is_disjoint_from_held_out_ids():
    df = build_synthetic_df()
    train_df = df.iloc[:6000]
    held_out_ids = set(df.iloc[6000:]["id"])

    pilot = build_pilot(train_df, pilot_size=150, seed=7)

    assert set(pilot["id"]).isdisjoint(held_out_ids)
    assert set(pilot["id"]) <= set(train_df["id"])


def test_pilot_size_too_small_or_pool_too_small_raises():
    df = build_synthetic_df()
    with pytest.raises(ValueError):
        build_pilot(df, pilot_size=50, seed=7)
    with pytest.raises(ValueError):
        build_pilot(df.iloc[:500], pilot_size=150, seed=7)


def test_assign_pilot_stratum_uses_first_rare_label():
    df = pd.DataFrame(
        [
            {"id": "a", **{label: 0 for label in LABELS}, "threat": 1, "identity_hate": 1},
            {"id": "b", **{label: 0 for label in LABELS}, "severe_toxic": 1, "identity_hate": 1},
            {"id": "c", **{label: 0 for label in LABELS}, "toxic": 1},
            {"id": "d", **{label: 0 for label in LABELS}},
        ]
    )
    assert assign_pilot_stratum(df).tolist() == [
        "threat",
        "severe_toxic",
        "common_toxic",
        "clean",
    ]


def test_save_pilot_refuses_to_overwrite(tmp_path):
    from policyjev.jev.pilot import save_pilot

    pilot = pd.DataFrame({"id": ["a", "b"], "stratum": ["clean", "clean"]})
    path = tmp_path / "pilot.csv"

    save_pilot(pilot, path)
    assert pd.read_csv(path)["id"].tolist() == ["a", "b"]
    with pytest.raises(FileExistsError):
        save_pilot(pilot, path)
