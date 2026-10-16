"""Review-budget analysis: how many harmful comments a fixed review capacity catches."""

import argparse

import numpy as np
import pandas as pd

from policyjev.config import LABELS, REPO_ROOT, load_config
from policyjev.evaluation.bootstrap import paired_bootstrap, percentile_interval
from policyjev.evaluation.splits import SPLITS, load_split_scores_bundle, require_split_access
from policyjev.evaluation.stack import STACK_SYSTEM

SYSTEMS = ["logreg_word", "distilbert", "jev"]
BUDGETS_PCT = [1, 2, 5, 10, 20]
METRICS = ["recall_any", "precision", "recall_rare"]
RARE_LABELS = ["threat", "severe_toxic", "identity_hate"]
DIFFERENCES = [("jev", "logreg_word"), ("jev", "distilbert")]
PRESPECIFIED_BUDGETS = (5, 10)
PRESPECIFIED_METRICS = ("recall_any", "recall_rare")
HYPOTHESIS_BUDGETS = (1, 2)
NOTE = (
    "Pre-specified contrasts: jev against each primary baseline on recall_any and recall_rare at the "
    "5 and 10 percent budgets.\n"
    "Hypothesis generated on validation: jev has higher recall_rare than distilbert at the 1 and 2 "
    "percent budgets.\n"
    "All other cells are descriptive. Many intervals are shown and none is adjusted for multiplicity."
)


def cell_status(comparison: str, metric: str, budget_pct: int) -> str:
    """Classify a table cell as pre-specified, validation-generated hypothesis, or descriptive."""
    if comparison in ("jev - logreg_word", "jev - distilbert"):
        if metric in PRESPECIFIED_METRICS and budget_pct in PRESPECIFIED_BUDGETS:
            return "pre-specified"
        if (
            comparison == "jev - distilbert"
            and metric == "recall_rare"
            and budget_pct in HYPOTHESIS_BUDGETS
        ):
            return "hypothesis generated on validation"
    return "descriptive"


def review_scores(calibrated: np.ndarray) -> np.ndarray:
    """Maximum calibrated probability across labels, per comment."""
    return np.asarray(calibrated, dtype=float).max(axis=1)


def budget_metrics(
    scores: np.ndarray,
    tiebreak: np.ndarray,
    any_positive: np.ndarray,
    rare_positive: np.ndarray,
    budgets_pct: list[int] = BUDGETS_PCT,
) -> np.ndarray:
    """Metrics for the top-scored comments at each budget: shape (n_budgets, 3)."""
    n = len(scores)
    order = np.lexsort((tiebreak, -scores))
    caught_any = np.cumsum(any_positive[order])
    caught_rare = np.cumsum(rare_positive[order])
    total_any = any_positive.sum()
    total_rare = rare_positive.sum()

    result = np.empty((len(budgets_pct), len(METRICS)))
    for i, pct in enumerate(budgets_pct):
        k = max(1, int(round(n * pct / 100)))
        result[i, 0] = caught_any[k - 1] / total_any if total_any else np.nan
        result[i, 1] = caught_any[k - 1] / k
        result[i, 2] = caught_rare[k - 1] / total_rare if total_rare else np.nan
    return result


def build_review_budget_table(
    y: np.ndarray,
    calibrated: dict[str, np.ndarray],
    n_resamples: int,
    seed: int,
    confidence: float,
    budgets_pct: list[int] = BUDGETS_PCT,
    with_stack: bool = False,
) -> pd.DataFrame:
    """Tidy table of per-system metrics and paired differences with percentile intervals."""
    systems = SYSTEMS + [STACK_SYSTEM] if with_stack else SYSTEMS
    differences = DIFFERENCES + [(STACK_SYSTEM, s) for s in SYSTEMS] if with_stack else DIFFERENCES
    scores = {system: review_scores(calibrated[system]) for system in systems}
    any_positive = (y.sum(axis=1) > 0).astype(int)
    rare_idx = [LABELS.index(label) for label in RARE_LABELS]
    rare_positive = (y[:, rare_idx].sum(axis=1) > 0).astype(int)
    tiebreak = np.random.default_rng(seed).random(len(y))

    def evaluate(idx: np.ndarray) -> np.ndarray:
        return np.array(
            [
                budget_metrics(
                    scores[system][idx], tiebreak[idx], any_positive[idx],
                    rare_positive[idx], budgets_pct,
                )
                for system in systems
            ]
        ).ravel()

    shape = (len(systems), len(budgets_pct), len(METRICS))
    point = evaluate(np.arange(len(y))).reshape(shape)
    samples = paired_bootstrap(len(y), evaluate, n_resamples, seed).reshape(
        (n_resamples,) + shape
    )
    position = {system: i for i, system in enumerate(systems)}

    rows = []

    def add(comparison: str, values: np.ndarray, draws: np.ndarray) -> None:
        low, high = percentile_interval(draws, confidence)
        for b, pct in enumerate(budgets_pct):
            for m, metric in enumerate(METRICS):
                rows.append(
                    {
                        "budget_pct": pct,
                        "metric": metric,
                        "comparison": comparison,
                        "value": values[b, m],
                        "lo": low[b, m],
                        "hi": high[b, m],
                        "status": cell_status(comparison, metric, pct),
                    }
                )

    for system in systems:
        add(system, point[position[system]], samples[:, position[system]])
    for a, b in differences:
        add(
            f"{a} - {b}",
            point[position[a]] - point[position[b]],
            samples[:, position[a]] - samples[:, position[b]],
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Write artifacts/metrics/review_budget_<split>.csv and print it."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=SPLITS, default="validation")
    parser.add_argument("--with-stack", action="store_true")
    args = parser.parse_args()
    require_split_access(args.split)

    config = load_config()
    data = load_split_scores_bundle(
        args.split, SYSTEMS + [STACK_SYSTEM] if args.with_stack else SYSTEMS
    )
    table = build_review_budget_table(
        data.y,
        data.calibrated,
        config["bootstrap"]["n_resamples"],
        config["seed"],
        config["bootstrap"]["confidence_level"],
        with_stack=args.with_stack,
    )

    suffix = "_with_stack" if args.with_stack else ""
    metrics_dir = REPO_ROOT / "artifacts" / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(metrics_dir / f"review_budget_{args.split}{suffix}.csv", index=False)

    any_positive = int((data.y.sum(axis=1) > 0).sum())
    print(
        f"split: {args.split}, comments: {len(data.y)}, with any positive label: {any_positive}; "
        f"JEV question set {config['jev']['primary_question_version']}"
    )
    print("score = maximum calibrated probability across the six labels; 95% paired-bootstrap intervals")
    print(NOTE)
    if args.with_stack:
        print(
            "jev_stack is a secondary supervised system fit on validation labels; "
            "it is not JEV and is never reported as JEV."
        )
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
