import numpy as np
import pytest

from policyjev.config import LABELS
from policyjev.evaluation.validation_report import SYSTEMS, build_comparison


def make_inputs(n: int = 400):
    """Synthetic labels, three score sets (jev equals logreg_word), and in-sample F1 values."""
    rng = np.random.default_rng(0)
    y = (rng.random((n, len(LABELS))) < 0.3).astype(int)
    signal = np.clip(y * 0.4 + rng.random(y.shape) * 0.6, 0, 1)
    other = np.clip(y * 0.2 + rng.random(y.shape) * 0.8, 0, 1)
    probs = {"logreg_word": signal, "distilbert": other, "jev": signal.copy()}
    f1 = {system: np.linspace(0.3, 0.6, len(LABELS)) for system in SYSTEMS}
    return y, probs, f1


def test_comparison_layout_and_basic_values():
    y, probs, f1 = make_inputs()
    table = build_comparison(y, probs, f1, n_resamples=40, seed=1, confidence=0.95)

    assert table["label"].tolist() == LABELS + ["mean"]
    assert np.allclose(table["prevalence"][:-1], y.mean(axis=0))
    assert table["positives"][:-1].tolist() == y.sum(axis=0).tolist()
    assert (table["ap_logreg_word_lo"] <= table["ap_logreg_word_hi"]).all()
    assert table.loc[6, "ap_jev"] == pytest.approx(table.loc[:5, "ap_jev"].mean())
    assert np.allclose(
        table["brier_jev"][:-1], ((probs["jev"] - y) ** 2).mean(axis=0)
    )
    assert np.allclose(table["f1_in_sample_distilbert"][:-1], f1["distilbert"])
    assert table.loc[6, "f1_in_sample_jev"] == pytest.approx(f1["jev"].mean())


def test_identical_systems_have_exactly_zero_paired_difference():
    y, probs, f1 = make_inputs()
    table = build_comparison(y, probs, f1, n_resamples=40, seed=1, confidence=0.95)

    name = "diff_ap_jev_minus_logreg_word"
    assert np.allclose(table[name], 0.0)
    assert np.allclose(table[f"{name}_lo"], 0.0)
    assert np.allclose(table[f"{name}_hi"], 0.0)

    other = "diff_ap_jev_minus_distilbert"
    assert (table[f"{other}_lo"] <= table[f"{other}_hi"]).all()
    assert (table[other] > 0).all()


def test_comparison_is_reproducible_for_a_seed():
    y, probs, f1 = make_inputs()
    first = build_comparison(y, probs, f1, n_resamples=30, seed=5, confidence=0.95)
    again = build_comparison(y, probs, f1, n_resamples=30, seed=5, confidence=0.95)
    other = build_comparison(y, probs, f1, n_resamples=30, seed=6, confidence=0.95)

    assert first.equals(again)
    assert not first["ap_jev_lo"].equals(other["ap_jev_lo"])


def test_label_without_positives_gives_nan_ap_and_does_not_break_the_mean():
    y, probs, f1 = make_inputs()
    y = y.copy()
    y[:, 3] = 0
    table = build_comparison(y, probs, f1, n_resamples=20, seed=1, confidence=0.95)

    assert np.isnan(table.loc[3, "ap_jev"])
    assert not np.isnan(table.loc[6, "ap_jev"])
