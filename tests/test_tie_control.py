import numpy as np
import pytest

from policyjev.config import LABELS
from policyjev.evaluation.tie_control import tie_control_table
from policyjev.evaluation.validation_report import average_precision


def make_inputs(n: int = 300):
    """Labels plus one fine-grained and one already two-decimal score set."""
    rng = np.random.default_rng(0)
    y = (rng.random((n, len(LABELS))) < 0.3).astype(int)
    fine = np.clip(y * 0.4 + rng.random(y.shape) * 0.6, 0, 1)
    coarse = np.round(fine, 2)
    return y, fine, coarse


def test_table_layout_and_rounding_effect():
    y, fine, _ = make_inputs()
    table = tie_control_table(y, {"fine": fine})

    assert table["system"].unique().tolist() == ["fine"]
    assert table["label"].tolist() == LABELS + ["mean"]
    expected_rounded = average_precision(y[:, 0], np.round(fine[:, 0], 2))
    assert table.loc[0, "ap_rounded"] == pytest.approx(expected_rounded)
    assert table.loc[0, "ap"] == pytest.approx(average_precision(y[:, 0], fine[:, 0]))
    assert np.allclose(table["difference"], table["ap_rounded"] - table["ap"])
    assert table.loc[6, "ap"] == pytest.approx(table.loc[:5, "ap"].mean())


def test_already_rounded_scores_are_unchanged():
    y, _, coarse = make_inputs()
    table = tie_control_table(y, {"coarse": coarse})

    assert np.allclose(table["difference"], 0.0)


def test_several_systems_are_stacked():
    y, fine, coarse = make_inputs()
    table = tie_control_table(y, {"a": fine, "b": coarse})

    assert table["system"].unique().tolist() == ["a", "b"]
    assert len(table) == 2 * (len(LABELS) + 1)
