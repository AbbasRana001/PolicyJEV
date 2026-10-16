"""Figures: reliability diagrams, per-label AP with intervals, review-budget recall curves.

Reliability diagrams use fixed probability bins (edges 0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1),
because equal-count bins put almost all positives of these imbalanced labels in one bin. They are
drawn per system, per label and pooled over labels, for raw and calibrated scores. Only bins with
at least 20 comments are plotted, and bins with fewer than 20 positives are marked as low support.
The equal-count reliability bins (reliability_bins) remain available for tables. Figures are saved
as PNG and SVG under reports/figures/.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from policyjev.config import LABELS, REPO_ROOT
from policyjev.evaluation.splits import SPLITS, load_split_scores_bundle, require_split_access
from policyjev.evaluation.validation_report import SYSTEMS

N_BINS = 5
BIN_EDGES = [0.0, 0.01, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]
MIN_COMMENTS_PLOTTED = 20
MIN_POSITIVES_SUPPORTED = 20
FIGURES_DIR = REPO_ROOT / "reports" / "figures"
COLORS = {
    "logreg_word": "#0072B2",
    "distilbert": "#D55E00",
    "jev": "#009E73",
    "jev_stack": "#CC79A7",
}
DISPLAY = {
    "logreg_word": "Logistic regression",
    "distilbert": "DistilBERT",
    "jev": "JEV",
    "jev_stack": "JEV stack (supervised)",
}


def reliability_bins(probs: np.ndarray, y: np.ndarray, n_bins: int = N_BINS) -> pd.DataFrame:
    """Equal-count bins over sorted probabilities: counts, positives, mean predicted, observed."""
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(y, dtype=int)
    order = np.argsort(probs, kind="stable")
    rows = []
    for number, idx in enumerate(np.array_split(order, n_bins), start=1):
        rows.append(
            {
                "bin": number,
                "n": len(idx),
                "positives": int(y[idx].sum()),
                "mean_predicted": float(probs[idx].mean()),
                "observed_rate": float(y[idx].mean()),
            }
        )
    return pd.DataFrame(rows)


def fixed_bins(
    probs: np.ndarray,
    y: np.ndarray,
    edges: list[float] = BIN_EDGES,
    min_comments: int = MIN_COMMENTS_PLOTTED,
    min_positives: int = MIN_POSITIVES_SUPPORTED,
) -> pd.DataFrame:
    """Fixed probability bins [lower, upper) with the last bin closed at 1.

    plotted is True for bins with at least min_comments comments; low_support is True for
    bins with fewer than min_positives positives.
    """
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(y, dtype=int)
    index = np.digitize(probs, edges[1:-1])
    rows = []
    for number in range(len(edges) - 1):
        mask = index == number
        n = int(mask.sum())
        positives = int(y[mask].sum())
        rows.append(
            {
                "bin": number + 1,
                "lower": edges[number],
                "upper": edges[number + 1],
                "n": n,
                "positives": positives,
                "mean_predicted": float(probs[mask].mean()) if n else float("nan"),
                "observed_rate": float(y[mask].mean()) if n else float("nan"),
                "plotted": n >= min_comments,
                "low_support": positives < min_positives,
            }
        )
    return pd.DataFrame(rows)


def bin_labels(edges: list[float] = BIN_EDGES) -> list[str]:
    """Short tick labels for the fixed bins, such as <.01, .01-.05 and >.8."""

    def short(value: float) -> str:
        return f"{value:g}".lstrip("0")

    labels = [f"{short(lo)}-{short(hi)}" for lo, hi in zip(edges[:-1], edges[1:])]
    labels[0] = f"<{short(edges[1])}"
    labels[-1] = f">{short(edges[-2])}"
    return labels


def panel_inputs(
    raw: np.ndarray, calibrated: np.ndarray, y: np.ndarray
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Per-label panels plus a pooled panel: name -> (raw scores, calibrated scores, labels)."""
    panels = {
        label: (raw[:, j], calibrated[:, j], y[:, j]) for j, label in enumerate(LABELS)
    }
    panels["pooled"] = (raw.ravel(), calibrated.ravel(), y.ravel())
    return panels


def _pyplot():
    """Import pyplot with a non-interactive backend."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def save_figure(fig, name: str, out_dir: Path | None = None) -> list[Path]:
    """Save a figure as PNG and SVG; returns the two paths."""
    out_dir = out_dir if out_dir is not None else FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix, options in ((".png", {"dpi": 200}), (".svg", {})):
        path = out_dir / f"{name}{suffix}"
        fig.savefig(path, bbox_inches="tight", **options)
        paths.append(path)
    return paths


def reliability_figure(
    system: str, raw: np.ndarray, calibrated: np.ndarray, y: np.ndarray, edges: list[float] = BIN_EDGES
):
    """One system: a reliability panel per label and pooled, with a log-scale count strip.

    Bins with fewer than 20 comments are not plotted; plotted bins with fewer than 20
    positives are drawn with open markers (low support).
    """
    plt = _pyplot()
    from matplotlib.lines import Line2D

    panels = panel_inputs(raw, calibrated, y)
    color = COLORS.get(system, "#444444")
    labels = bin_labels(edges)

    fig = plt.figure(figsize=(14, 8.5))
    outer = fig.add_gridspec(
        2, 4, left=0.055, right=0.99, top=0.92, bottom=0.17, hspace=0.3, wspace=0.3
    )
    for number, (name, (raw_p, cal_p, y_col)) in enumerate(panels.items()):
        inner = outer[number // 4, number % 4].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.46)
        ax = fig.add_subplot(inner[0])
        strip = fig.add_subplot(inner[1])

        bins_raw = fixed_bins(raw_p, y_col, edges)
        bins_cal = fixed_bins(cal_p, y_col, edges)
        extent = 1e-6
        for bins, style, marker in ((bins_raw, "--", "o"), (bins_cal, "-", "s")):
            shown = bins[bins["plotted"]]
            if shown.empty:
                continue
            extent = max(extent, shown["mean_predicted"].max(), shown["observed_rate"].max())
            ax.plot(
                shown["mean_predicted"], shown["observed_rate"], linestyle=style, marker=marker,
                color=color, alpha=0.6 if style == "--" else 1.0, markerfacecolor=color,
                clip_on=False,
            )
            weak = shown[shown["low_support"]]
            ax.scatter(
                weak["mean_predicted"], weak["observed_rate"], s=70, marker=marker,
                facecolors="white", edgecolors=color, zorder=3, clip_on=False,
            )
        limit = min(1.0, extent * 1.1)
        ax.plot([0, limit], [0, limit], color="#777777", linewidth=1, linestyle=":")
        ax.set_xlim(0, limit)
        ax.set_ylim(0, limit)
        ax.set_xlabel("mean predicted probability", fontsize=8)
        ax.set_ylabel("observed rate", fontsize=8)
        ax.set_title(f"{name} (positives: {int(y_col.sum())})", fontsize=10)
        ax.tick_params(labelsize=8)

        positions = np.arange(1, len(labels) + 1)
        strip.bar(positions - 0.2, bins_raw["n"], width=0.4, color=color, alpha=0.55)
        strip.bar(positions + 0.2, bins_cal["n"], width=0.4, color=color)
        strip.set_yscale("log")
        strip.set_xticks(positions)
        strip.set_xticklabels(labels, rotation=45, ha="right", rotation_mode="anchor", fontsize=6)
        strip.set_ylabel("comments (log)", fontsize=7)
        strip.tick_params(axis="y", labelsize=6)

    handles = [
        Line2D([0], [0], color=color, linestyle="--", marker="o", alpha=0.6, label="raw"),
        Line2D([0], [0], color=color, linestyle="-", marker="s", label="calibrated"),
        Line2D([0], [0], color=color, linestyle="none", marker="o", markerfacecolor="white",
               label="fewer than 20 positives (low support)"),
        Line2D([0], [0], color="#777777", linestyle=":", label="y = x"),
    ]
    fig.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.04), ncol=4,
        fontsize=9, frameon=False,
    )
    fig.text(
        0.5, 0.012,
        "Bins by predicted probability (fixed bins; bins with fewer than 20 comments are not plotted). "
        "Lower strips: comments per bin on a log scale (light = raw, dark = calibrated).",
        ha="center", va="bottom", fontsize=8,
    )
    fig.suptitle(f"Reliability: {DISPLAY.get(system, system)}", fontsize=12, y=0.985)
    return fig


def ap_figure(comparison: pd.DataFrame, systems: list[str]):
    """Average precision per label (and mean) with 95% bootstrap intervals, grouped by label."""
    plt = _pyplot()
    labels = comparison["label"].tolist()
    x = np.arange(len(labels))
    width = 0.8 / len(systems)

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, system in enumerate(systems):
        ap = comparison[f"ap_{system}"].to_numpy()
        low = comparison[f"ap_{system}_lo"].to_numpy()
        high = comparison[f"ap_{system}_hi"].to_numpy()
        ax.errorbar(
            x + (i - (len(systems) - 1) / 2) * width, ap,
            yerr=np.vstack([ap - low, high - ap]), fmt="o", capsize=3,
            color=COLORS.get(system, "#444444"), label=DISPLAY.get(system, system),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20)
    ax.set_ylabel("average precision")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", linewidth=0.5, alpha=0.5)
    ax.legend()
    ax.set_title("Average precision per label (95% paired-bootstrap intervals)")
    return fig


def review_budget_figure(table: pd.DataFrame, systems: list[str]):
    """Review-budget recall curves with interval bands: any positive, and rare labels."""
    plt = _pyplot()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True)
    for ax, metric, title in (
        (axes[0], "recall_any", "Comments with any positive label"),
        (axes[1], "recall_rare", "threat, severe_toxic or identity_hate"),
    ):
        for system in systems:
            rows = table[(table["comparison"] == system) & (table["metric"] == metric)]
            rows = rows.sort_values("budget_pct")
            color = COLORS.get(system, "#444444")
            ax.plot(rows["budget_pct"], rows["value"], marker="o", color=color, label=DISPLAY.get(system, system))
            ax.fill_between(rows["budget_pct"], rows["lo"], rows["hi"], color=color, alpha=0.15)
        ax.set_xlabel("review budget (% of all comments)")
        ax.set_title(title, fontsize=10)
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 5, 10, 20])
        ax.set_xticklabels(["1", "2", "5", "10", "20"])
        ax.set_ylim(0, 1)
        ax.grid(linewidth=0.5, alpha=0.5)
    axes[0].set_ylabel("recall")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Recall at fixed review budgets (95% paired-bootstrap bands)")
    return fig


def main() -> None:
    """Write the reliability, average-precision and review-budget figures."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=SPLITS, default="validation")
    args = parser.parse_args()
    require_split_access(args.split)

    data = load_split_scores_bundle(args.split, SYSTEMS)
    written: list[Path] = []
    for system in SYSTEMS:
        fig = reliability_figure(system, data.raw[system], data.calibrated[system], data.y)
        written += save_figure(fig, f"reliability_{args.split}_{system}")

    metrics_dir = REPO_ROOT / "artifacts" / "metrics"
    comparison_path = metrics_dir / f"{args.split}_comparison.csv"
    if comparison_path.exists():
        fig = ap_figure(pd.read_csv(comparison_path), SYSTEMS)
        written += save_figure(fig, f"ap_by_label_{args.split}")
    else:
        print(f"skipped the AP figure: {comparison_path.name} not found")

    budget_path = metrics_dir / f"review_budget_{args.split}.csv"
    if budget_path.exists():
        fig = review_budget_figure(pd.read_csv(budget_path), SYSTEMS)
        written += save_figure(fig, f"review_budget_{args.split}")
    else:
        print(f"skipped the review-budget figure: {budget_path.name} not found")

    for path in written:
        print(path)


if __name__ == "__main__":
    main()
