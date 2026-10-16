import sys

import numpy as np
import pytest

from policyjev.config import LABELS
from policyjev.evaluation import disagreement as dg
from policyjev.evaluation import splits


def test_masks_partition_every_row():
    y = np.array([1, 1, 0, 0, 1, 0])
    jev = np.array([1, 0, 0, 1, 0, 0])
    distilbert = np.array([1, 1, 1, 0, 0, 0])

    masks = dg.category_masks(y, jev, distilbert)

    assert sum(masks.values()).tolist() == [1] * 6
    assert masks["both_right"].tolist() == [True, False, False, False, False, True]
    assert masks["only_jev_right"].tolist() == [False, False, True, False, False, False]
    assert masks["only_distilbert_right"].tolist() == [False, True, False, True, False, False]
    assert masks["both_wrong"].tolist() == [False, False, False, False, True, False]


def make_inputs(n: int = 300):
    rng = np.random.default_rng(0)
    y = (rng.random((n, len(LABELS))) < 0.3).astype(int)
    jev = (rng.random(y.shape) < 0.5).astype(int)
    distilbert = (rng.random(y.shape) < 0.5).astype(int)
    return y, jev, distilbert


def test_counts_cover_all_rows_per_class():
    y, jev, distilbert = make_inputs()
    table = dg.disagreement_counts(y, jev, distilbert)

    assert len(table) == len(LABELS) * 2
    assert list(table.columns) == ["label", "row_class"] + dg.CATEGORIES
    for j, label in enumerate(LABELS):
        for row_class, value in dg.ROW_CLASSES.items():
            row = table[(table["label"] == label) & (table["row_class"] == row_class)].iloc[0]
            assert sum(row[c] for c in dg.CATEGORIES) == (y[:, j] == value).sum()
    assert "comment_text" not in table.columns


def test_examples_are_seeded_bounded_and_belong_to_their_category():
    y, jev, distilbert = make_inputs()
    first = dg.sample_examples(y, jev, distilbert, 2, seed=1)
    again = dg.sample_examples(y, jev, distilbert, 2, seed=1)
    other = dg.sample_examples(y, jev, distilbert, 2, seed=2)

    assert first == again
    assert first != other
    assert len(first) <= len(LABELS) * 2 * len(dg.CATEGORIES) * 2
    for label, row_class, category, row in first:
        j = LABELS.index(label)
        masks = dg.category_masks(y[:, j], jev[:, j], distilbert[:, j])
        assert masks[category][row]
        assert y[row, j] == dg.ROW_CLASSES[row_class]


def test_format_examples_prints_text_ids_and_scores_without_writing_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ids = ["a", "b"]
    texts = {"a": "first   comment\nwith breaks", "b": "second comment"}
    jev_scores = np.full((2, 6), 0.25)
    distilbert_scores = np.full((2, 6), 0.75)

    lines = dg.format_examples([("toxic", "positive", "both_right", 0)], ids, texts, jev_scores, distilbert_scores)

    assert len(lines) == 1
    assert "id=a" in lines[0]
    assert "jev=0.250" in lines[0]
    assert "distilbert=0.750" in lines[0]
    assert "first comment with breaks" in lines[0]
    assert list(tmp_path.iterdir()) == []


def test_cli_refuses_test_split_without_the_tag(monkeypatch):
    monkeypatch.setattr(splits, "tag_exists", lambda tag: False)
    monkeypatch.setattr(
        dg, "load_split_scores_bundle", lambda *a, **k: (_ for _ in ()).throw(AssertionError("read"))
    )
    monkeypatch.setattr(sys, "argv", ["disagreement", "--split", "test", "--show", "2"])

    with pytest.raises(SystemExit) as info:
        dg.main()
    assert "protocol-frozen" in str(info.value)
