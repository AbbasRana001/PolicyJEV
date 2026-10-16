import numpy as np
import pandas as pd
import pytest

from policyjev.config import LABELS
from policyjev.evaluation import reliability_plots as rp


def make_inputs(n: int = 400):
    rng = np.random.default_rng(0)
    y = (rng.random((n, len(LABELS))) < 0.3).astype(int)
    raw = np.clip(y * 0.3 + rng.random(y.shape) * 0.7, 0, 1)
    calibrated = np.clip(raw * 0.8, 0, 1)
    return y, raw, calibrated


def test_bins_are_equal_count_and_cover_all_rows():
    probs = np.linspace(0, 1, 103)
    y = (probs > 0.5).astype(int)
    bins = rp.reliability_bins(probs, y)

    assert len(bins) == rp.N_BINS
    assert bins["n"].sum() == 103
    assert bins["n"].max() - bins["n"].min() <= 1
    assert bins["positives"].sum() == y.sum()
    assert bins["mean_predicted"].is_monotonic_increasing
    assert bins["observed_rate"].iloc[0] == 0.0
    assert bins["observed_rate"].iloc[-1] == 1.0


def test_panels_cover_every_label_and_the_pooled_set():
    y, raw, calibrated = make_inputs()
    panels = rp.panel_inputs(raw, calibrated, y)

    assert list(panels) == LABELS + ["pooled"]
    pooled_raw, pooled_cal, pooled_y = panels["pooled"]
    assert len(pooled_raw) == len(pooled_cal) == len(pooled_y) == y.size
    assert pooled_y.sum() == y.sum()
    assert np.array_equal(panels["threat"][2], y[:, LABELS.index("threat")])


def test_figures_are_written_as_png_and_svg(tmp_path):
    pytest.importorskip("matplotlib")
    y, raw, calibrated = make_inputs()

    fig = rp.reliability_figure("jev", raw, calibrated, y)
    paths = rp.save_figure(fig, "reliability_test_jev", tmp_path)
    assert sorted(p.suffix for p in paths) == [".png", ".svg"]
    assert all(p.stat().st_size > 0 for p in paths)

    systems = ["logreg_word", "distilbert", "jev"]
    comparison = pd.DataFrame({"label": LABELS + ["mean"]})
    for i, system in enumerate(systems):
        comparison[f"ap_{system}"] = np.linspace(0.3, 0.8, 7) + i * 0.01
        comparison[f"ap_{system}_lo"] = comparison[f"ap_{system}"] - 0.03
        comparison[f"ap_{system}_hi"] = comparison[f"ap_{system}"] + 0.03
    ap_paths = rp.save_figure(rp.ap_figure(comparison, systems), "ap", tmp_path)
    assert all(p.exists() for p in ap_paths)

    rows = [
        {"budget_pct": b, "metric": m, "comparison": s, "value": 0.1 * (i + 1) * b / 20, "lo": 0.0, "hi": 1.0}
        for s in systems
        for i, b in enumerate([1, 2, 5, 10, 20])
        for m in ("recall_any", "recall_rare", "precision")
    ]
    budget_paths = rp.save_figure(rp.review_budget_figure(pd.DataFrame(rows), systems), "budget", tmp_path)
    assert all(p.exists() for p in budget_paths)


def test_fixed_bins_use_the_specified_edges_and_half_open_intervals():
    assert rp.BIN_EDGES == [0.0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
    probs = np.array([0.0, 0.005, 0.01, 0.049, 0.05, 0.399, 0.4, 0.8, 0.999, 1.0])
    bins = rp.fixed_bins(probs, np.zeros(10, dtype=int), min_comments=1)

    assert bins["n"].tolist() == [2, 2, 1, 0, 1, 1, 0, 3]
    assert bins["n"].sum() == len(probs)
    assert bins["lower"].tolist() == rp.BIN_EDGES[:-1]
    assert bins["upper"].tolist() == rp.BIN_EDGES[1:]
    assert bins.loc[bins["n"] == 0, "mean_predicted"].isna().all()


def test_fixed_bin_counts_and_flags():
    rng = np.random.default_rng(0)
    probs = np.concatenate([rng.uniform(0, 0.01, 5000), rng.uniform(0.4, 0.6, 30), rng.uniform(0.8, 1.0, 10)])
    y = np.concatenate([np.zeros(5000), np.ones(30), np.ones(10)]).astype(int)
    bins = rp.fixed_bins(probs, y)

    first = bins.iloc[0]
    assert first["n"] == 5000 and first["positives"] == 0
    assert first["plotted"] and first["low_support"]
    middle = bins.iloc[5]
    assert middle["n"] == 30 and middle["plotted"] and not middle["low_support"]
    last = bins.iloc[7]
    assert last["n"] == 10 and not last["plotted"] and last["low_support"]
    assert bins["n"].sum() == len(probs)


def test_equal_count_bins_are_kept_for_tables():
    probs = np.linspace(0, 1, 100)
    bins = rp.reliability_bins(probs, (probs > 0.5).astype(int))
    assert len(bins) == rp.N_BINS and bins["n"].tolist() == [20] * 5


def test_reliability_figure_axes_labels_limits_reference_line_and_strips():
    pytest.importorskip("matplotlib")
    rng = np.random.default_rng(1)
    n = 3000
    y = (rng.random((n, len(LABELS))) < 0.08).astype(int)
    raw = np.clip(y * 0.4 + rng.random(y.shape) * 0.5, 0, 1)
    calibrated = np.clip(raw * 0.7, 0, 1)

    fig = rp.reliability_figure("jev", raw, calibrated, y)
    axes = fig.axes
    mains, strips = axes[0:14:2], axes[1:14:2]

    assert len(mains) == 7 and len(strips) == 7
    for ax in mains:
        assert ax.get_xlabel() == "mean predicted probability"
        assert ax.get_ylabel() == "observed rate"
        assert ax.get_xlim() == ax.get_ylim()
        diagonal = [line for line in ax.get_lines() if line.get_linestyle() == ":"]
        assert len(diagonal) == 1
        assert list(diagonal[0].get_xdata()) == list(diagonal[0].get_ydata())
        assert diagonal[0].get_xdata()[-1] == pytest.approx(ax.get_xlim()[1])
    for strip in strips:
        assert strip.get_yscale() == "log"
        assert "comments" in strip.get_ylabel()

    assert fig.legends and any("low support" in text.get_text() for text in fig.legends[0].get_texts())

    toxic = mains[0]
    expected = int(rp.fixed_bins(calibrated[:, 0], y[:, 0])["plotted"].sum())
    solid = [line for line in toxic.get_lines() if line.get_linestyle() == "-" and line.get_marker() == "s"]
    assert len(solid) == 1 and len(solid[0].get_xdata()) == expected
    dashed = [line for line in toxic.get_lines() if line.get_linestyle() == "--"]
    assert len(dashed) == 1


def test_low_support_bins_get_open_markers():
    pytest.importorskip("matplotlib")
    rng = np.random.default_rng(2)
    y = (rng.random((2000, len(LABELS))) < 0.01).astype(int)
    scores = np.clip(y * 0.5 + rng.random(y.shape) * 0.3, 0, 1)

    fig = rp.reliability_figure("distilbert", scores, scores, y)

    threat_axis = fig.axes[2 * LABELS.index("threat")]
    assert len(threat_axis.collections) >= 1
