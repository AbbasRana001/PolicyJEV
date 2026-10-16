import pytest

from policyjev.config import LABELS
from policyjev.data.load import load_test, load_train


def test_load_train_preserves_na_string_and_linebreak(tmp_path):
    """train.csv's literal 'NA' comment and an embedded line break survive loading."""
    raw_dir = tmp_path
    header = "id,comment_text," + ",".join(LABELS)
    zeros = ",".join("0" for _ in LABELS)
    content = (
        f"{header}\n"
        f"0001,NA,{zeros}\n"
        f'0002,"line one\nline two",{zeros}\n'
    )
    (raw_dir / "train.csv").write_bytes(content.encode("utf-8"))

    df = load_train(raw_dir)

    assert list(df.columns) == ["id", "comment_text"] + LABELS
    assert df.loc[df["id"] == "0001", "id"].iloc[0] == "0001"
    assert df.loc[df["id"] == "0001", "comment_text"].iloc[0] == "NA"
    assert df.loc[df["id"] == "0002", "comment_text"].iloc[0] == "line one\nline two"


def test_load_test_drops_unlabeled_rows(tmp_path):
    """load_test drops rows with -1 labels and records how many were dropped."""
    raw_dir = tmp_path
    (raw_dir / "test.csv").write_text(
        "id,comment_text\nt1,hello\nt2,world\nt3,hi\n", encoding="utf-8"
    )
    header = "id," + ",".join(LABELS)
    neg_one = ",".join("-1" for _ in LABELS)
    zeros = ",".join("0" for _ in LABELS)
    one_positive = "1," + ",".join("0" for _ in LABELS[1:])
    (raw_dir / "test_labels.csv").write_text(
        f"{header}\n"
        f"t1,{zeros}\n"
        f"t2,{neg_one}\n"
        f"t3,{one_positive}\n",
        encoding="utf-8",
    )

    df = load_test(raw_dir)

    assert sorted(df["id"].tolist()) == ["t1", "t3"]
    assert df.attrs["n_dropped_unlabeled"] == 1


def test_load_test_raises_on_mismatched_ids(tmp_path):
    """load_test raises ValueError when test.csv and test_labels.csv ids differ."""
    raw_dir = tmp_path
    (raw_dir / "test.csv").write_text(
        "id,comment_text\nt1,hello\nt2,world\n", encoding="utf-8"
    )
    header = "id," + ",".join(LABELS)
    zeros = ",".join("0" for _ in LABELS)
    (raw_dir / "test_labels.csv").write_text(
        f"{header}\nt1,{zeros}\nt9,{zeros}\n", encoding="utf-8"
    )

    with pytest.raises(ValueError):
        load_test(raw_dir)
