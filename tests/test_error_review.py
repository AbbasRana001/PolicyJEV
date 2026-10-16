import sys

import numpy as np
import pytest

from policyjev.config import LABELS
from policyjev.evaluation import error_review as er
from policyjev.evaluation import splits


def test_categories_follow_the_definitions():
    #            gold, logreg, distilbert, jev
    rows = [
        (1, 1, 1, 0),  # a: JEV false negative that both baselines catch
        (1, 0, 0, 1),  # b: JEV hit that both baselines miss
        (0, 0, 0, 1),  # c: JEV-only false positive
        (0, 1, 1, 1),  # d: all three above the threshold, gold 0
        (1, 0, 0, 0),  # d: all three below the threshold, gold 1
        (1, 1, 0, 0),  # none: baselines disagree
        (0, 1, 0, 1),  # none
        (1, 1, 1, 1),  # none: everyone right
    ]
    y, logreg, distilbert, jev = (np.array([r[i] for r in rows]) for i in range(4))

    masks = er.category_masks(y, logreg, distilbert, jev)

    assert np.flatnonzero(masks[er.CATEGORIES[0]]).tolist() == [0]
    assert np.flatnonzero(masks[er.CATEGORIES[1]]).tolist() == [1]
    assert np.flatnonzero(masks[er.CATEGORIES[2]]).tolist() == [2]
    assert np.flatnonzero(masks[er.CATEGORIES[3]]).tolist() == [3, 4]


def make_inputs(n: int = 600):
    rng = np.random.default_rng(0)
    y = (rng.random((n, len(LABELS))) < 0.3).astype(int)
    predictions = {
        system: (rng.random(y.shape) < 0.5).astype(int)
        for system in ("logreg_word", "distilbert", "jev")
    }
    ids = [f"id{i:04d}" for i in range(n)]
    return y, predictions, ids


def test_samples_are_seeded_bounded_and_counts_are_complete():
    y, predictions, ids = make_inputs()
    counts, sampled = er.review_samples(y, predictions, ids, seed=1, max_per_group=25)
    counts_again, sampled_again = er.review_samples(y, predictions, ids, seed=1, max_per_group=25)
    _, sampled_other = er.review_samples(y, predictions, ids, seed=2, max_per_group=25)

    assert len(counts) == len(LABELS) * len(er.CATEGORIES)
    assert (counts["sampled"] == counts["count"].clip(upper=25)).all()
    assert counts.equals(counts_again)
    assert sampled.equals(sampled_again)
    assert not sampled.equals(sampled_other)
    assert sampled.groupby(["label", "category"]).size().max() <= 25
    assert set(sampled.columns) == {"label", "category", "id"}
    assert sampled["id"].isin(ids).all()
    assert len(sampled) == counts["sampled"].sum()


def test_sampled_ids_belong_to_their_category():
    y, predictions, ids = make_inputs()
    _, sampled = er.review_samples(y, predictions, ids, seed=1)
    position = {comment_id: i for i, comment_id in enumerate(ids)}

    for row in sampled.itertuples():
        j = LABELS.index(row.label)
        masks = er.category_masks(
            y[:, j], predictions["logreg_word"][:, j], predictions["distilbert"][:, j], predictions["jev"][:, j]
        )
        assert masks[row.category][position[row.id]]


def test_format_review_prints_text_with_a_content_warning_and_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    y, predictions, ids = make_inputs(200)
    _, sampled = er.review_samples(y, predictions, ids, seed=1, max_per_group=5)
    texts = {comment_id: f"sample   text of {comment_id}" for comment_id in ids}
    scores = {s: np.full(y.shape, 0.5) for s in ("logreg_word", "distilbert", "jev")}
    position = {comment_id: i for i, comment_id in enumerate(ids)}

    lines = er.format_review(sampled, texts, scores, position, show=2)

    assert lines[0] == er.WARNING
    assert any("sample text of id" in line for line in lines)
    assert any("jev=0.500" in line for line in lines)
    assert list(tmp_path.iterdir()) == []
    first_group = sampled.iloc[0]
    shown = [line for line in lines if line.startswith("id=")]
    assert len(shown) <= 2 * len(LABELS) * len(er.CATEGORIES)
    assert first_group["id"] in "".join(lines)


def test_output_tables_contain_no_comment_text():
    y, predictions, ids = make_inputs(200)
    counts, sampled = er.review_samples(y, predictions, ids, seed=1)
    assert "comment_text" not in counts.columns
    assert "comment_text" not in sampled.columns


def test_cli_refuses_test_split_without_the_tag(monkeypatch):
    monkeypatch.setattr(splits, "tag_exists", lambda tag: False)
    monkeypatch.setattr(
        er, "load_split_scores_bundle", lambda *a, **k: (_ for _ in ()).throw(AssertionError("read"))
    )
    monkeypatch.setattr(sys, "argv", ["error_review", "--split", "test", "--show", "3"])

    with pytest.raises(SystemExit) as info:
        er.main()
    assert "protocol-frozen" in str(info.value)
