"""Build the question-pilot set from train-split comments only."""

from pathlib import Path

import numpy as np
import pandas as pd

from policyjev.config import LABELS, REPO_ROOT, load_config
from policyjev.data.load import load_train
from policyjev.data.split import load_split_ids

RARE_ORDER = ["threat", "severe_toxic", "identity_hate"]
COMMON_POSITIVE_LABELS = ["toxic", "obscene", "insult"]
PER_RARE_LABEL = 20
N_COMMON_POSITIVE = 40


def assign_pilot_stratum(df: pd.DataFrame) -> pd.Series:
    """Label each row threat, severe_toxic, identity_hate, common_toxic or clean (else other)."""
    stratum = pd.Series("other", index=df.index, dtype=object)
    for label in reversed(RARE_ORDER):
        stratum[df[label] == 1] = label
    no_rare = (df[RARE_ORDER] == 0).all(axis=1)
    any_common = (df[COMMON_POSITIVE_LABELS] == 1).any(axis=1)
    stratum[no_rare & any_common] = "common_toxic"
    stratum[(df[LABELS] == 0).all(axis=1)] = "clean"
    return stratum


def build_pilot(df: pd.DataFrame, pilot_size: int, seed: int) -> pd.DataFrame:
    """Sample the pilot rows from df (which must contain train-split rows only)."""
    n_clean = pilot_size - PER_RARE_LABEL * len(RARE_ORDER) - N_COMMON_POSITIVE
    if n_clean < 0:
        raise ValueError("pilot_size is smaller than the fixed positive groups")

    stratum = assign_pilot_stratum(df)
    rng = np.random.default_rng(seed)
    wanted = {label: PER_RARE_LABEL for label in RARE_ORDER}
    wanted["common_toxic"] = N_COMMON_POSITIVE
    wanted["clean"] = n_clean

    chosen: list[tuple[str, str]] = []
    for name, count in wanted.items():
        pool = sorted(df.loc[stratum == name, "id"].tolist())
        if len(pool) < count:
            raise ValueError(f"only {len(pool)} rows available for {name}, need {count}")
        picked = rng.choice(len(pool), size=count, replace=False)
        chosen.extend((pool[i], name) for i in picked)

    order = rng.permutation(len(chosen))
    return pd.DataFrame(
        [{"id": chosen[i][0], "stratum": chosen[i][1]} for i in order]
    )


def save_pilot(pilot: pd.DataFrame, path: Path) -> None:
    """Write the pilot ids to CSV, never overwriting an existing file."""
    if path.exists():
        raise FileExistsError(f"{path} already exists")
    pilot.to_csv(path, index=False)


def main() -> None:
    """Create data/processed/pilot.csv and print its composition."""
    config = load_config()
    processed_dir = REPO_ROOT / "data" / "processed"
    train_ids, validation_ids = load_split_ids(processed_dir)
    train_df = load_train()
    train_df = train_df[train_df["id"].isin(set(train_ids))].reset_index(drop=True)

    pilot = build_pilot(train_df, config["jev"]["pilot_size"], config["seed"] + 3)

    pilot_ids = set(pilot["id"])
    assert pilot_ids <= set(train_ids)
    assert pilot_ids.isdisjoint(validation_ids)

    save_pilot(pilot, processed_dir / "pilot.csv")

    labeled = train_df.set_index("id").loc[pilot["id"].tolist()]
    print(f"pilot rows: {len(pilot)}")
    print(pilot["stratum"].value_counts().to_string())
    print("positives per label in the pilot:")
    print(labeled[LABELS].sum().to_string())
    print("pilot comes from the train split only and is disjoint from validation")


if __name__ == "__main__":
    main()
