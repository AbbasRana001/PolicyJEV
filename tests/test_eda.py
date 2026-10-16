import pandas as pd

from policyjev.config import LABELS
from policyjev.data.eda import (
    cooccurrence,
    label_count_distribution,
    length_summary,
    quality_report,
)


def make_df(rows: list[tuple[str, str, dict[str, int]]]) -> pd.DataFrame:
    """Build a labeled dataframe from (id, text, {label: 1}) tuples."""
    data = []
    for comment_id, text, positives in rows:
        record = {"id": comment_id, "comment_text": text}
        for label in LABELS:
            record[label] = positives.get(label, 0)
        data.append(record)
    return pd.DataFrame(data)


def test_label_count_distribution_sums():
    df = make_df(
        [
            ("a", "x", {}),
            ("b", "y", {"toxic": 1}),
            ("c", "z", {"toxic": 1, "insult": 1}),
            ("d", "w", {"toxic": 1, "obscene": 1, "insult": 1}),
        ]
    )
    result = label_count_distribution(df)

    assert result["n_labels"].tolist() == list(range(7))
    assert result["comments"].sum() == len(df)
    assert abs(result["share"].sum() - 1.0) < 1e-9
    assert result.loc[result["n_labels"] == 3, "comments"].iloc[0] == 1


def test_cooccurrence_is_symmetric_with_label_totals_on_diagonal():
    df = make_df(
        [
            ("a", "x", {"toxic": 1, "insult": 1}),
            ("b", "y", {"toxic": 1}),
            ("c", "z", {"insult": 1, "threat": 1}),
        ]
    )
    matrix = cooccurrence(df)

    assert list(matrix.index) == LABELS
    assert list(matrix.columns) == LABELS
    assert (matrix.to_numpy() == matrix.to_numpy().T).all()
    for label in LABELS:
        assert matrix.loc[label, label] == df[label].sum()
    assert matrix.loc["toxic", "insult"] == 1


def test_length_summary_all_row():
    df = make_df(
        [
            ("a", "one", {}),
            ("b", "one two", {}),
            ("c", "one two three", {}),
            ("d", "one two three four", {}),
        ]
    )
    summary = length_summary(df)
    all_row = summary.loc[summary["group"] == "all"].iloc[0]

    assert all_row["comments"] == 4
    assert all_row["words_p50"] == 2.5
    assert all_row["words_max"] == 4


def test_quality_report_counts():
    train = make_df(
        [
            ("t1", "", {}),
            ("t2", "Hello  World", {"toxic": 1}),
            ("t3", "hello world", {}),
            ("t4", "bad", {"severe_toxic": 1}),
            ("t5", "something else", {}),
        ]
    )
    validation = make_df(
        [
            ("v1", "HELLO world", {}),
            ("v2", "unrelated text", {}),
        ]
    )
    report = quality_report(train, validation).set_index("check")["count"]

    assert report["empty_or_whitespace_comments"] == 1
    assert report["duplicate_texts_in_train"] == 2
    assert report["duplicate_groups_with_conflicting_labels"] == 1
    assert report["severe_toxic_without_toxic"] == 1
    assert report["validation_rows_with_text_also_in_train"] == 1
