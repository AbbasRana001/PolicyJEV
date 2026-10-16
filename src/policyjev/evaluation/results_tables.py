"""Markdown results tables built from the committed aggregate metric files.

Nothing is recomputed from predictions: every table is a reformatting of columns that already
exist in artifacts/metrics/*.csv. A table whose source file or columns are missing is skipped and
listed. The output, docs/results_tables.md, is deterministic (no dates, fixed order, 3 decimals).
"""

from pathlib import Path

import pandas as pd

from policyjev.config import LABELS, REPO_ROOT

SYSTEMS = ["logreg_word", "distilbert", "jev", "jev_stack"]
DISPLAY = {
    "logreg_word": "Logistic regression",
    "distilbert": "DistilBERT",
    "jev": "JEV",
    "jev_stack": "JEV stack (supervised)",
}
SOURCES = {
    "comparison": "test_comparison_with_stack.csv",
    "threshold": "test_threshold_metrics_with_stack.csv",
    "budget": "review_budget_test_with_stack.csv",
    "second_pass": "second_pass_report.csv",
    "latency": "latency_cost.csv",
    "policy_thresholds": "policy/policy_thresholds.csv",
    "policy_summary": "policy_test_summary.csv",
    "policy_contrasts": "policy_test_contrasts.csv",
    "policy_grid": "policy_test_grid.csv",
    "policy_per_label": "policy_test_per_label.csv",
}
POLICY_SYSTEMS = ["logreg_word", "distilbert", "jev"]
POLICY_CONTRASTS = {
    "jev - logreg_word": "JEV minus logistic regression",
    "jev - distilbert": "JEV minus DistilBERT",
}
POLICY_METRICS = {
    "m1_review_share": "M1 review share",
    "m2_action_precision": "M2 ACTION precision",
    "m3_miss_rate": "M3 miss rate",
    "m4_rare_miss_rate": "M4 rare miss rate",
}
CONTRASTS = [
    ("jev", "logreg_word", "JEV minus logistic regression"),
    ("jev", "distilbert", "JEV minus DistilBERT"),
    ("jev_stack", "jev", "JEV stack minus JEV"),
]
ROW_LABELS = LABELS + ["micro", "macro"]
BUDGET_METRICS = ["recall_any", "recall_rare"]
OUTPUT_PATH = REPO_ROOT / "docs" / "results_tables.md"


class TableSkipped(Exception):
    """A table cannot be built from the columns that exist."""


def fmt(value: object) -> str:
    """Format a number to three decimals; missing values are empty."""
    if value is None or pd.isna(value):
        return ""
    number = float(value)
    return "0.000" if round(number, 3) == 0 else f"{number:.3f}"


def fmt_count(value: object) -> str:
    """Format a count as an integer; missing values are empty."""
    if value is None or pd.isna(value):
        return ""
    return str(int(round(float(value))))


def interval(value: object, low: object, high: object) -> str:
    """Format 'value [low, high]' to three decimals; empty if the value is missing."""
    if value is None or pd.isna(value):
        return ""
    return f"{fmt(value)} [{fmt(low)}, {fmt(high)}]"


def md_table(headers: list[str], rows: list[list[str]]) -> str:
    """A Markdown table."""
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines)


def require(frame: pd.DataFrame | None, columns: list[str], source: str) -> pd.DataFrame:
    """Return the frame or raise TableSkipped if it or any column is missing."""
    if frame is None:
        raise TableSkipped(f"{source} not found")
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise TableSkipped(f"{source} has no column(s): {', '.join(missing)}")
    return frame


def ap_columns(systems: list[str]) -> list[str]:
    """The AP value and interval column names for the given systems."""
    return [f"ap_{s}{suffix}" for s in systems for suffix in ("", "_lo", "_hi")]


def table_mean_ap(frame: pd.DataFrame | None) -> str:
    """Table a: mean AP per system with its 95% interval."""
    frame = require(frame, ["label"] + ap_columns(SYSTEMS), SOURCES["comparison"])
    mean = frame[frame["label"] == "mean"]
    if mean.empty:
        raise TableSkipped(f"{SOURCES['comparison']} has no mean row")
    row = mean.iloc[0]
    rows = [
        [DISPLAY[s], interval(row[f"ap_{s}"], row[f"ap_{s}_lo"], row[f"ap_{s}_hi"])]
        for s in SYSTEMS
    ]
    return md_table(["System", "Mean AP [95% interval]"], rows)


def table_label_ap(frame: pd.DataFrame | None) -> str:
    """Table b: per-label AP per system with intervals."""
    frame = require(frame, ["label", "positives"] + ap_columns(SYSTEMS), SOURCES["comparison"])
    rows = []
    for label in LABELS:
        match = frame[frame["label"] == label]
        if match.empty:
            raise TableSkipped(f"{SOURCES['comparison']} has no row for {label}")
        row = match.iloc[0]
        rows.append(
            [label, fmt_count(row["positives"])]
            + [interval(row[f"ap_{s}"], row[f"ap_{s}_lo"], row[f"ap_{s}_hi"]) for s in SYSTEMS]
        )
    return md_table(["Label", "Positives"] + [DISPLAY[s] for s in SYSTEMS], rows)


def table_differences(frame: pd.DataFrame | None) -> str:
    """Table c: paired AP differences, overall (mean) and per label, with intervals."""
    columns = ["label", "positives"]
    for a, b, _ in CONTRASTS:
        columns += [f"diff_ap_{a}_minus_{b}{suffix}" for suffix in ("", "_lo", "_hi")]
    frame = require(frame, columns, SOURCES["comparison"])
    rows = []
    for label in LABELS + ["mean"]:
        match = frame[frame["label"] == label]
        if match.empty:
            raise TableSkipped(f"{SOURCES['comparison']} has no row for {label}")
        row = match.iloc[0]
        rows.append(
            [label if label != "mean" else "mean (overall)", fmt_count(row["positives"])]
            + [
                interval(
                    row[f"diff_ap_{a}_minus_{b}"],
                    row[f"diff_ap_{a}_minus_{b}_lo"],
                    row[f"diff_ap_{a}_minus_{b}_hi"],
                )
                for a, b, _ in CONTRASTS
            ]
        )
    return md_table(["Label", "Positives"] + [name for _, _, name in CONTRASTS], rows)


def tables_f1(frame: pd.DataFrame | None) -> list[tuple[str, str, str]]:
    """Table d: F1 per system for each threshold basis; (basis, sample status, table)."""
    frame = require(
        frame,
        ["threshold_basis", "system", "label", "positives", "f1", "sample_status"],
        SOURCES["threshold"],
    )
    results = []
    for basis in sorted(frame["threshold_basis"].unique()):
        subset = frame[frame["threshold_basis"] == basis]
        rows = []
        for label in ROW_LABELS:
            per_system = {
                s: subset[(subset["system"] == s) & (subset["label"] == label)] for s in SYSTEMS
            }
            if any(match.empty for match in per_system.values()):
                raise TableSkipped(f"{SOURCES['threshold']} has no complete rows for {label}")
            first = per_system[SYSTEMS[0]].iloc[0]
            status = "; ".join(sorted(per_system[SYSTEMS[0]]["sample_status"].astype(str).unique()))
            rows.append(
                [label, fmt_count(first["positives"])]
                + [fmt(per_system[s].iloc[0]["f1"]) for s in SYSTEMS]
                + [status]
            )
        table = md_table(
            ["Label", "Positives"] + [DISPLAY[s] for s in SYSTEMS] + ["Sample status"], rows
        )
        statuses = "; ".join(sorted(subset["sample_status"].astype(str).unique()))
        results.append((str(basis), statuses, table))
    return results


def table_brier(frame: pd.DataFrame | None) -> str:
    """Table e: Brier score, raw and calibrated, per label and mean."""
    columns = ["label"] + [f"brier{kind}_{s}" for s in SYSTEMS for kind in ("", "_raw")]
    frame = require(frame, columns, SOURCES["comparison"])
    rows = []
    for label in LABELS + ["mean"]:
        match = frame[frame["label"] == label]
        if match.empty:
            raise TableSkipped(f"{SOURCES['comparison']} has no row for {label}")
        row = match.iloc[0]
        rows.append(
            [label]
            + [fmt(row[f"brier{kind}_{s}"]) for s in SYSTEMS for kind in ("_raw", "")]
        )
    headers = ["Label"] + [
        f"{DISPLAY[s]} {kind}" for s in SYSTEMS for kind in ("raw", "calibrated")
    ]
    return md_table(headers, rows)


def tables_review_budget(frame: pd.DataFrame | None) -> tuple[str, str]:
    """Table f: recall per system at each budget, and the pre-specified paired contrasts."""
    frame = require(
        frame, ["budget_pct", "metric", "comparison", "value", "lo", "hi", "status"], SOURCES["budget"]
    )
    budgets = sorted(frame["budget_pct"].unique())
    rows = []
    for metric in BUDGET_METRICS:
        for budget in budgets:
            cells = []
            for system in SYSTEMS:
                match = frame[
                    (frame["comparison"] == system)
                    & (frame["metric"] == metric)
                    & (frame["budget_pct"] == budget)
                ]
                if match.empty:
                    raise TableSkipped(f"{SOURCES['budget']} has no row for {system}, {metric}, {budget}")
                cell = match.iloc[0]
                cells.append(interval(cell["value"], cell["lo"], cell["hi"]))
            rows.append([metric, fmt_count(budget)] + cells)
    recall_table = md_table(["Metric", "Budget %"] + [DISPLAY[s] for s in SYSTEMS], rows)

    order = {name: i for i, (_, _, name) in enumerate(CONTRASTS)}
    contrast_names = {f"{a} - {b}": name for a, b, name in CONTRASTS}
    selected = frame[(frame["status"] == "pre-specified") & frame["comparison"].isin(contrast_names)]
    if selected.empty:
        raise TableSkipped(f"{SOURCES['budget']} has no pre-specified contrast rows")
    selected = selected.assign(
        _name=selected["comparison"].map(contrast_names),
        _order=selected["comparison"].map(contrast_names).map(order),
    ).sort_values(["_order", "metric", "budget_pct"], kind="stable")
    contrast_rows = [
        [r["_name"], r["metric"], fmt_count(r["budget_pct"]), interval(r["value"], r["lo"], r["hi"])]
        for _, r in selected.iterrows()
    ]
    contrast_table = md_table(["Contrast", "Metric", "Budget %", "Difference [95% interval]"], contrast_rows)
    return recall_table, contrast_table


def table_second_pass(frame: pd.DataFrame | None) -> str:
    """Table g: first versus second JEV pass."""
    columns = [
        "label", "positives", "mean_abs_probability_change", "share_rows_changed",
        "ap_first", "ap_second", "ap_change", "f1_first", "f1_second", "f1_change",
    ]
    frame = require(frame, columns, SOURCES["second_pass"])
    rows = [
        [str(r["label"]), fmt_count(r["positives"])] + [fmt(r[c]) for c in columns[2:]]
        for _, r in frame.iterrows()
    ]
    headers = [
        "Label", "Positives", "Mean abs. probability change", "Share of rows changed",
        "AP first pass", "AP second pass", "AP change", "F1 first pass", "F1 second pass", "F1 change",
    ]
    return md_table(headers, rows)


def table_latency(frame: pd.DataFrame | None) -> str:
    """Table h: latency, throughput and API request count per system."""
    columns = [
        "system", "device", "device_name", "latency_p50_ms", "latency_p95_ms",
        "throughput_per_s_batch32", "n_requests", "n_comments",
    ]
    frame = require(frame, columns, SOURCES["latency"])
    rows = [
        [
            DISPLAY.get(str(r["system"]), str(r["system"])),
            "" if pd.isna(r["device"]) else str(r["device"]),
            "" if pd.isna(r["device_name"]) else str(r["device_name"]),
            fmt(r["latency_p50_ms"]),
            fmt(r["latency_p95_ms"]),
            fmt(r["throughput_per_s_batch32"]),
            fmt_count(r["n_requests"]),
            fmt_count(r["n_comments"]),
        ]
        for _, r in frame.iterrows()
    ]
    headers = [
        "System", "Device", "Device name", "Latency p50 (ms)", "Latency p95 (ms)",
        "Throughput (comments/s, batch 32)", "API requests", "Comments timed",
    ]
    return md_table(headers, rows)


def table_policy_thresholds(frame: pd.DataFrame | None) -> str:
    """Table i: policy thresholds at the primary targets (fit on validation data)."""
    source = SOURCES["policy_thresholds"]
    columns = [
        "system", "label", "is_primary", "low", "high", "action_enabled",
        "validation_recall_at_low", "validation_precision_at_high", "validation_flagged_at_high",
    ]
    frame = require(frame, columns, source)
    primary = frame[frame["is_primary"].astype(str).str.lower() == "true"]
    rows = []
    for system in POLICY_SYSTEMS:
        for label in LABELS:
            match = primary[(primary["system"] == system) & (primary["label"] == label)]
            if match.empty:
                raise TableSkipped(f"{source} has no primary row for {system}, {label}")
            r = match.iloc[0]
            rows.append(
                [
                    DISPLAY[system], label, fmt(r["low"]), fmt(r["high"]),
                    "yes" if str(r["action_enabled"]).lower() == "true" else "no",
                    fmt(r["validation_recall_at_low"]), fmt(r["validation_precision_at_high"]),
                    fmt_count(r["validation_flagged_at_high"]),
                ]
            )
    headers = [
        "System", "Label", "LOW", "HIGH", "ACTION enabled", "Validation recall at LOW",
        "Validation precision at HIGH", "Validation comments at or above HIGH",
    ]
    return md_table(headers, rows)


def table_policy_summary(frame: pd.DataFrame | None) -> str:
    """Table j: policy test summary at the primary targets."""
    columns = [
        "system", "n_rows", "n_failed_rows", "n_allow", "n_review", "n_action", "allow_share",
        "m1_review_share", "m2_action_precision", "m3_miss_rate", "m4_rare_miss_rate",
        "action_share", "action_coverage",
    ]
    frame = require(frame, columns, SOURCES["policy_summary"])
    rows = []
    for system in POLICY_SYSTEMS:
        match = frame[frame["system"] == system]
        if match.empty:
            raise TableSkipped(f"{SOURCES['policy_summary']} has no row for {system}")
        r = match.iloc[0]
        rows.append(
            [DISPLAY[system]]
            + [fmt_count(r[c]) for c in ("n_rows", "n_failed_rows", "n_allow", "n_review", "n_action")]
            + [fmt(r[c]) for c in columns[6:]]
        )
    headers = [
        "System", "Comments", "Failed rows", "ALLOW", "HUMAN REVIEW", "ACTION", "ALLOW share",
        "M1 review share", "M2 ACTION precision", "M3 miss rate", "M4 rare miss rate",
        "ACTION share", "ACTION coverage",
    ]
    return md_table(headers, rows)


def table_policy_contrasts(frame: pd.DataFrame | None) -> str:
    """Table k: paired policy contrasts with intervals and bootstrap resample counts."""
    columns = [
        "contrast", "metric", "difference", "ci_low", "ci_high", "resamples_used", "resamples_left_out",
    ]
    frame = require(frame, columns, SOURCES["policy_contrasts"])
    rows = []
    for contrast, name in POLICY_CONTRASTS.items():
        for metric, metric_name in POLICY_METRICS.items():
            match = frame[(frame["contrast"] == contrast) & (frame["metric"] == metric)]
            if match.empty:
                raise TableSkipped(f"{SOURCES['policy_contrasts']} has no row for {contrast}, {metric}")
            r = match.iloc[0]
            rows.append(
                [
                    name, metric_name, interval(r["difference"], r["ci_low"], r["ci_high"]),
                    fmt_count(r["resamples_used"]), fmt_count(r["resamples_left_out"]),
                ]
            )
    return md_table(
        ["Contrast", "Metric", "Difference [95% interval]", "Resamples used", "Resamples left out"], rows
    )


def table_policy_grid(frame: pd.DataFrame | None) -> str:
    """Table l: sensitivity grid (point estimates), with the primary point marked."""
    columns = [
        "system", "recall_target", "precision_target", "is_primary", "m1_review_share",
        "m2_action_precision", "m3_miss_rate", "m4_rare_miss_rate", "action_share", "action_coverage",
    ]
    frame = require(frame, columns, SOURCES["policy_grid"])
    rows = []
    for system in POLICY_SYSTEMS:
        subset = frame[frame["system"] == system]
        if subset.empty:
            raise TableSkipped(f"{SOURCES['policy_grid']} has no rows for {system}")
        for _, r in subset.iterrows():
            rows.append(
                [
                    DISPLAY[system], fmt(r["recall_target"]), fmt(r["precision_target"]),
                    "primary" if str(r["is_primary"]).lower() == "true" else "",
                ]
                + [fmt(r[c]) for c in columns[4:]]
            )
    headers = [
        "System", "Recall target", "Precision target", "Primary", "M1 review share",
        "M2 ACTION precision", "M3 miss rate", "M4 rare miss rate", "ACTION share", "ACTION coverage",
    ]
    return md_table(headers, rows)


def dash(text: str) -> str:
    """A dash for an undefined (empty) value."""
    return text if text else "-"


def table_policy_per_label(per_label: pd.DataFrame | None, thresholds: pd.DataFrame | None) -> str:
    """Table m: per-label policy on test at the primary targets, joined with validation values."""
    source = f"{SOURCES['policy_per_label']} / {SOURCES['policy_thresholds']}"
    per_label = require(
        per_label,
        [
            "system", "label", "positives", "low", "high", "action_enabled", "review_share_label",
            "action_share_label", "action_precision_label", "miss_rate_label",
        ],
        SOURCES["policy_per_label"],
    )
    thresholds = require(
        thresholds,
        ["system", "label", "is_primary", "validation_recall_at_low", "validation_precision_at_high"],
        SOURCES["policy_thresholds"],
    )
    primary = thresholds[thresholds["is_primary"].astype(str).str.lower() == "true"]
    rows = []
    for system in POLICY_SYSTEMS:
        for label in LABELS:
            test = per_label[(per_label["system"] == system) & (per_label["label"] == label)]
            validation = primary[(primary["system"] == system) & (primary["label"] == label)]
            if test.empty or validation.empty:
                raise TableSkipped(f"{source} has no matching rows for {system}, {label}")
            a, v = test.iloc[0], validation.iloc[0]
            test_recall = 1.0 - float(a["miss_rate_label"]) if not pd.isna(a["miss_rate_label"]) else float("nan")
            rows.append(
                [
                    DISPLAY[system], label, fmt_count(a["positives"]), fmt(a["low"]), fmt(a["high"]),
                    "yes" if str(a["action_enabled"]).lower() == "true" else "no",
                    dash(fmt(v["validation_recall_at_low"])), dash(fmt(test_recall)),
                    dash(fmt(v["validation_precision_at_high"])), dash(fmt(a["action_precision_label"])),
                    dash(fmt(a["review_share_label"])), dash(fmt(a["action_share_label"])),
                ]
            )
    headers = [
        "System", "Label", "Positives", "LOW", "HIGH", "ACTION enabled", "Validation recall at LOW",
        "Test recall at LOW", "Validation precision at HIGH", "Test ACTION precision",
        "Review share", "ACTION share",
    ]
    return md_table(headers, rows)


def source_path(name: str) -> str:
    """Display path of a source file: policy files live under artifacts/policy/."""
    return f"artifacts/{name}" if "/" in name else f"artifacts/metrics/{name}"


def build_results(frames: dict[str, pd.DataFrame | None]) -> tuple[str, list[str]]:
    """The full Markdown document and the list of tables that could not be built."""
    skipped: list[str] = []
    sections: list[str] = [
        "# Results tables",
        "Built by `python -m policyjev.evaluation.results_tables` from the committed aggregate files "
        "in `artifacts/metrics/`. Nothing is recomputed from predictions. Values are formatted to "
        "three decimals; intervals are 95% paired-bootstrap percentile intervals.",
    ]

    def add(title: str, source: str, build) -> None:
        sections.append(f"## {title}")
        names = source if isinstance(source, tuple) else (source,)
        sections.append("Source: " + " and ".join(f"`{source_path(n)}`" for n in names) + ".")
        try:
            sections.extend(build())
        except TableSkipped as reason:
            skipped.append(f"{title}: {reason}")
            sections.append(f"Not built: {reason}.")

    comparison = frames.get("comparison")
    add("a. Mean average precision per system", SOURCES["comparison"], lambda: [table_mean_ap(comparison)])
    add("b. Average precision per label", SOURCES["comparison"], lambda: [table_label_ap(comparison)])
    add(
        "c. Paired AP differences (overall and per label)",
        SOURCES["comparison"],
        lambda: [table_differences(comparison)],
    )

    def f1_sections() -> list[str]:
        out = []
        for basis, statuses, table in tables_f1(frames.get("threshold")):
            out += [f"Threshold basis `{basis}` (sample status: {statuses}).", table]
        return out

    add("d. F1 at the frozen validation-fit thresholds and at fixed 0.5", SOURCES["threshold"], f1_sections)
    add("e. Brier score, raw and calibrated", SOURCES["comparison"], lambda: [table_brier(comparison)])

    def budget_sections() -> list[str]:
        recall_table, contrast_table = tables_review_budget(frames.get("budget"))
        return [
            "Recall at each review budget (value [95% interval]).",
            recall_table,
            "Pre-specified paired contrasts (difference [95% interval]).",
            contrast_table,
        ]

    add("f. Review budget", SOURCES["budget"], budget_sections)
    add(
        "g. Second-pass stability (first pass against second pass)",
        SOURCES["second_pass"],
        lambda: [table_second_pass(frames.get("second_pass"))],
    )
    add("h. Latency", SOURCES["latency"], lambda: [table_latency(frames.get("latency"))])
    add(
        "i. Policy thresholds at the primary targets (fit on validation data)",
        SOURCES["policy_thresholds"],
        lambda: [table_policy_thresholds(frames.get("policy_thresholds"))],
    )
    add(
        "j. Policy on the test set at the primary targets",
        SOURCES["policy_summary"],
        lambda: [table_policy_summary(frames.get("policy_summary"))],
    )
    add(
        "k. Policy contrasts (paired bootstrap)",
        SOURCES["policy_contrasts"],
        lambda: [table_policy_contrasts(frames.get("policy_contrasts"))],
    )
    add(
        "l. Policy sensitivity grid (point estimates)",
        SOURCES["policy_grid"],
        lambda: [table_policy_grid(frames.get("policy_grid"))],
    )

    add(
        "m. Policy per label on the test set at the primary targets",
        (SOURCES["policy_per_label"], SOURCES["policy_thresholds"]),
        lambda: [table_policy_per_label(frames.get("policy_per_label"), frames.get("policy_thresholds"))],
    )

    if skipped:
        sections.append("## Tables not built")
        sections.extend(f"- {item}" for item in skipped)
    return "\n\n".join(sections) + "\n", skipped


def load_frames(metrics_dir: Path) -> dict[str, pd.DataFrame | None]:
    """Read the source files that exist; missing files map to None."""
    frames: dict[str, pd.DataFrame | None] = {}
    for key, name in SOURCES.items():
        path = metrics_dir.parent / name if "/" in name else metrics_dir / name
        frames[key] = pd.read_csv(path) if path.is_file() else None
    return frames


def write_results(metrics_dir: Path, output: Path) -> list[str]:
    """Write the results document (LF line endings); returns the skipped tables."""
    text, skipped = build_results(load_frames(metrics_dir))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as f:
        f.write(text)
    return skipped


def main() -> None:
    """Write docs/results_tables.md from artifacts/metrics."""
    skipped = write_results(REPO_ROOT / "artifacts" / "metrics", OUTPUT_PATH)
    print(f"wrote {OUTPUT_PATH}")
    for item in skipped:
        print(f"skipped: {item}")


if __name__ == "__main__":
    main()
