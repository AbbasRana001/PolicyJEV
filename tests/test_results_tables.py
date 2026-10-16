import numpy as np
import pandas as pd
import pytest

from policyjev.config import LABELS
from policyjev.evaluation import results_tables as rt

SYSTEMS = rt.SYSTEMS


def comparison_frame() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    for label in LABELS + ["mean"]:
        row = {"label": label, "positives": float("nan") if label == "mean" else 100.0, "prevalence": 0.01}
        for s in SYSTEMS:
            ap = float(rng.uniform(0.3, 0.8))
            row.update({f"ap_{s}": ap, f"ap_{s}_lo": ap - 0.02, f"ap_{s}_hi": ap + 0.02})
            row[f"brier_{s}"] = float(rng.uniform(0.01, 0.05))
            row[f"brier_raw_{s}"] = float(rng.uniform(0.01, 0.09))
            row[f"f1_{s}"] = 0.5
        for a, b, _ in rt.CONTRASTS:
            diff = float(rng.uniform(-0.1, 0.1))
            row.update(
                {
                    f"diff_ap_{a}_minus_{b}": diff,
                    f"diff_ap_{a}_minus_{b}_lo": diff - 0.03,
                    f"diff_ap_{a}_minus_{b}_hi": diff + 0.03,
                }
            )
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.loc[frame["label"] == "mean", "positives"] = float("nan")
    frame.loc[0, "ap_jev"] = 0.123456
    frame.loc[0, "ap_jev_lo"] = 0.1
    frame.loc[0, "ap_jev_hi"] = 0.15
    return frame


def threshold_frame() -> pd.DataFrame:
    rows = []
    for basis, status in (
        ("validation_oof_fit", "out-of-sample (thresholds were fit on validation)"),
        ("fixed_0.5", "not fitted"),
    ):
        for s in SYSTEMS:
            for label in rt.ROW_LABELS:
                rows.append(
                    {
                        "threshold_basis": basis, "system": s, "label": label,
                        "is_rare_label": label in ("threat", "severe_toxic", "identity_hate"),
                        "positives": float("nan") if label == "macro" else 50.0,
                        "threshold": 0.5, "precision": 0.5, "recall": 0.5,
                        "f1": 0.25 if basis == "fixed_0.5" else 0.75, "sample_status": status,
                    }
                )
    return pd.DataFrame(rows)


def budget_frame() -> pd.DataFrame:
    rows = []
    comparisons = SYSTEMS + [f"{a} - {b}" for a, b, _ in rt.CONTRASTS]
    for comparison in comparisons:
        for metric in ("recall_any", "precision", "recall_rare"):
            for budget in (1, 2, 5, 10, 20):
                rows.append(
                    {
                        "budget_pct": budget, "metric": metric, "comparison": comparison,
                        "value": 0.5, "lo": 0.4, "hi": 0.6,
                        "status": "pre-specified" if comparison.startswith("jev - ")
                        and metric != "precision" and budget in (5, 10) else "descriptive",
                    }
                )
    return pd.DataFrame(rows)


def second_pass_frame() -> pd.DataFrame:
    rows = [
        {
            "label": label, "positives": 7.0 if label != "mean" else float("nan"),
            "mean_abs_probability_change": 0.0123, "share_rows_changed": 0.25,
            "ap_first": 0.5, "ap_second": 0.51, "ap_change": 0.01,
            "f1_first": 0.4, "f1_second": 0.39, "f1_change": -0.0004,
        }
        for label in LABELS + ["mean"]
    ]
    return pd.DataFrame(rows)


def latency_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"system": "logreg_word", "latency_p50_ms": 1.23456, "latency_p95_ms": 2.0,
             "throughput_per_s_batch32": 1000.0, "device": "cpu", "device_name": "A CPU",
             "n_requests": float("nan"), "n_comments": 500.0},
            {"system": "jev", "latency_p50_ms": 800.0, "latency_p95_ms": 1500.0,
             "throughput_per_s_batch32": float("nan"), "device": "api", "device_name": float("nan"),
             "n_requests": 1000.0, "n_comments": float("nan")},
        ]
    )


def policy_thresholds_frame() -> pd.DataFrame:
    rows = []
    for system in ("logreg_word", "distilbert", "jev"):
        for label in LABELS:
            for primary, recall, precision in ((True, 0.9, 0.9), (False, 0.8, 0.95)):
                disabled = label == "threat"
                rows.append(
                    {
                        "system": system, "label": label, "recall_target": recall,
                        "precision_target": precision, "is_primary": primary,
                        "low": 0.12345 if primary else 0.5, "high": float("inf") if disabled else 0.75,
                        "action_enabled": not disabled, "validation_positives": 100,
                        "validation_recall_at_low": 0.9031 if primary else 0.5,
                        "validation_precision_at_high": float("nan") if disabled else 0.9044,
                        "validation_flagged_at_high": 0 if disabled else 1067,
                        "validation_review_share_label": 0.1,
                    }
                )
    return pd.DataFrame(rows)


def policy_summary_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "system": s, "n_rows": 1000, "n_failed_rows": 0, "n_allow": 800, "n_review": 150,
                "n_action": 50, "allow_share": 0.8, "m1_review_share": 0.15,
                "m2_action_precision": 0.95 if s != "jev" else float("nan"), "m3_miss_rate": 0.2,
                "m4_rare_miss_rate": 0.1, "action_share": 0.05, "action_coverage": 0.1,
                "any_positive_comments": 100, "rare_positive_comments": 20,
            }
            for s in ("logreg_word", "distilbert", "jev")
        ]
    )


def policy_contrasts_frame() -> pd.DataFrame:
    rows = []
    for contrast in ("jev - logreg_word", "jev - distilbert"):
        for metric in ("m1_review_share", "m2_action_precision", "m3_miss_rate", "m4_rare_miss_rate"):
            undefined = metric == "m2_action_precision" and contrast.endswith("distilbert")
            rows.append(
                {
                    "contrast": contrast, "metric": metric,
                    "difference": float("nan") if undefined else 0.0123,
                    "ci_low": float("nan") if undefined else -0.01,
                    "ci_high": float("nan") if undefined else 0.04,
                    "resamples_used": 0 if undefined else 997,
                    "resamples_left_out": 1000 if undefined else 3,
                }
            )
    return pd.DataFrame(rows)


def policy_grid_frame() -> pd.DataFrame:
    rows = []
    for system in ("logreg_word", "distilbert", "jev"):
        for recall in (0.8, 0.9, 0.95):
            for precision in (0.8, 0.9, 0.95):
                rows.append(
                    {
                        "system": system, "recall_target": recall, "precision_target": precision,
                        "is_primary": recall == 0.9 and precision == 0.9, "m1_review_share": 0.1,
                        "m2_action_precision": 0.9, "m3_miss_rate": 0.2, "m4_rare_miss_rate": 0.1,
                        "action_share": 0.05, "action_coverage": 0.1, "n_action": 50,
                    }
                )
    return pd.DataFrame(rows)


def policy_per_label_frame() -> pd.DataFrame:
    rows = []
    for system in ("logreg_word", "distilbert", "jev"):
        for label in LABELS:
            disabled = label == "threat"
            rows.append(
                {
                    "system": system, "label": label, "low": 0.12345,
                    "high": float("inf") if disabled else 0.75, "action_enabled": not disabled,
                    "positives": 211, "review_share_label": 0.0456, "action_share_label": 0.0 if disabled else 0.0234,
                    "flagged_share_label": 0.07, "action_flagged": 0 if disabled else 10,
                    "action_precision_label": float("nan") if disabled else 0.6,
                    "missed_positives": 21, "miss_rate_label": 0.1,
                }
            )
    return pd.DataFrame(rows)


def all_frames() -> dict:
    return {
        "comparison": comparison_frame(),
        "threshold": threshold_frame(),
        "budget": budget_frame(),
        "second_pass": second_pass_frame(),
        "latency": latency_frame(),
        "policy_thresholds": policy_thresholds_frame(),
        "policy_summary": policy_summary_frame(),
        "policy_contrasts": policy_contrasts_frame(),
        "policy_grid": policy_grid_frame(),
        "policy_per_label": policy_per_label_frame(),
    }


def test_formatting_helpers():
    assert rt.fmt(0.123456) == "0.123"
    assert rt.fmt(1) == "1.000"
    assert rt.fmt(float("nan")) == ""
    assert rt.fmt(-0.0004) == "0.000"
    assert rt.fmt_count(12.0) == "12"
    assert rt.fmt_count(float("nan")) == ""
    assert rt.interval(0.5, 0.4, 0.6) == "0.500 [0.400, 0.600]"
    assert rt.interval(float("nan"), 0.4, 0.6) == ""
    table = rt.md_table(["A", "B"], [["1", "2"]])
    assert table.splitlines() == ["| A | B |", "| --- | --- |", "| 1 | 2 |"]


def test_full_document_has_all_twelve_tables_with_sources_and_no_skips():
    text, skipped = rt.build_results(all_frames())

    assert skipped == []
    assert text.startswith("# Results tables")
    assert text.endswith("\n") and "\r" not in text
    for title in ("a. ", "b. ", "c. ", "d. ", "e. ", "f. ", "g. ", "h. ", "i. ", "j. ", "k. ", "l. ", "m. "):
        assert f"## {title}" in text
    assert "Tables not built" not in text
    for key, source in rt.SOURCES.items():
        if key != "policy_per_label":
            assert f"Source: `{rt.source_path(source)}`." in text
    assert "Source: `artifacts/policy/policy_thresholds.csv`." in text
    assert (
        "Source: `artifacts/metrics/policy_test_per_label.csv` and `artifacts/policy/policy_thresholds.csv`."
        in text
    )
    assert text.count("Source: `artifacts/") == 13


def test_each_table_follows_its_source_columns():
    frames = all_frames()
    mean_ap = rt.table_mean_ap(frames["comparison"]).splitlines()
    assert mean_ap[0] == "| System | Mean AP [95% interval] |"
    assert len(mean_ap) == 2 + len(SYSTEMS)
    assert mean_ap[2].startswith("| Logistic regression | ")
    assert "JEV stack (supervised)" in mean_ap[-1]

    per_label = rt.table_label_ap(frames["comparison"]).splitlines()
    assert len(per_label) == 2 + len(LABELS)
    assert "0.123 [0.100, 0.150]" in per_label[2]
    assert per_label[2].startswith("| toxic | 100 | ")

    diffs = rt.table_differences(frames["comparison"]).splitlines()
    assert len(diffs) == 2 + len(LABELS) + 1
    assert diffs[-1].startswith("| mean (overall) |  | ")
    assert "JEV minus logistic regression" in diffs[0] and "JEV stack minus JEV" in diffs[0]

    brier = rt.table_brier(frames["comparison"]).splitlines()
    assert "Logistic regression raw" in brier[0] and "JEV stack (supervised) calibrated" in brier[0]
    assert len(brier) == 2 + len(LABELS) + 1


def test_f1_tables_cover_both_threshold_bases_with_positives_and_status():
    tables = rt.tables_f1(threshold_frame())

    assert [basis for basis, _, _ in tables] == ["fixed_0.5", "validation_oof_fit"]
    by_basis = {basis: (status, table) for basis, status, table in tables}
    status, table = by_basis["validation_oof_fit"]
    assert status.startswith("out-of-sample")
    lines = table.splitlines()
    assert len(lines) == 2 + len(rt.ROW_LABELS)
    assert lines[2].startswith("| toxic | 50 | 0.750 | 0.750 | 0.750 | 0.750 | out-of-sample")
    assert lines[-1].startswith("| macro |  | 0.750")
    assert by_basis["fixed_0.5"][1].splitlines()[2].endswith("| not fitted |")
    assert "| 0.250 |" in by_basis["fixed_0.5"][1]


def test_review_budget_tables():
    recall, contrasts = rt.tables_review_budget(budget_frame())

    lines = recall.splitlines()
    assert len(lines) == 2 + 2 * 5
    assert lines[2].startswith("| recall_any | 1 | 0.500 [0.400, 0.600]")
    assert all("precision" not in line for line in lines[2:])

    contrast_lines = contrasts.splitlines()
    assert len(contrast_lines) == 2 + 2 * 2 * 2
    assert contrast_lines[2].startswith("| JEV minus logistic regression | recall_any | 5 |")
    assert all(line.split(" | ")[0].startswith("| JEV minus") for line in contrast_lines[2:])
    assert "JEV stack" not in contrasts


def test_second_pass_and_latency_tables():
    second = rt.table_second_pass(second_pass_frame()).splitlines()
    assert len(second) == 2 + len(LABELS) + 1
    assert second[2].startswith("| toxic | 7 | 0.012 | 0.250 | 0.500 | 0.510 | 0.010 | 0.400 | 0.390 | 0.000 |")
    assert second[-1].startswith("| mean |  | ")

    latency = rt.table_latency(latency_frame()).splitlines()
    assert latency[2] == "| Logistic regression | cpu | A CPU | 1.235 | 2.000 | 1000.000 |  | 500 |"
    assert latency[3] == "| JEV | api |  | 800.000 | 1500.000 |  | 1000 |  |"


def test_missing_columns_or_files_skip_only_that_table_and_are_listed():
    frames = all_frames()
    frames["comparison"] = frames["comparison"].drop(columns=["brier_raw_jev"])
    frames["latency"] = None

    text, skipped = rt.build_results(frames)

    assert len(skipped) == 2
    assert any(item.startswith("e. Brier") and "brier_raw_jev" in item for item in skipped)
    assert any(item.startswith("h. Latency") and "not found" in item for item in skipped)
    assert "Not built: " in text
    assert "## Tables not built" in text
    assert "## a. " not in text.split("## Tables not built")[1]
    assert rt.table_mean_ap(frames["comparison"])


def test_output_is_deterministic_and_main_writes_the_file(tmp_path):
    metrics = tmp_path / "metrics"
    metrics.mkdir()
    for key, frame in all_frames().items():
        name = rt.SOURCES[key]
        path = metrics.parent / name if "/" in name else metrics / name
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
    output = tmp_path / "docs" / "results_tables.md"

    first_skipped = rt.write_results(metrics, output)
    first = output.read_bytes()
    second_skipped = rt.write_results(metrics, output)

    assert first_skipped == second_skipped == []
    assert output.read_bytes() == first
    assert b"\r\n" not in first


def test_missing_source_files_do_not_fail_the_run(tmp_path):
    skipped = rt.write_results(tmp_path, tmp_path / "out.md")
    assert len(skipped) == 13
    assert "Tables not built" in (tmp_path / "out.md").read_text(encoding="utf-8")


def test_policy_tables_follow_their_source_columns():
    thresholds = rt.table_policy_thresholds(policy_thresholds_frame()).splitlines()
    assert len(thresholds) == 2 + 3 * len(LABELS)
    assert thresholds[2] == "| Logistic regression | toxic | 0.123 | 0.750 | yes | 0.903 | 0.904 | 1067 |"
    threat = [line for line in thresholds if "| threat |" in line][0]
    assert threat == "| Logistic regression | threat | 0.123 | inf | no | 0.903 |  | 0 |"
    assert all("0.500" not in line for line in thresholds[2:])

    summary = rt.table_policy_summary(policy_summary_frame()).splitlines()
    assert len(summary) == 2 + 3
    assert summary[2].startswith("| Logistic regression | 1000 | 0 | 800 | 150 | 50 | 0.800 | 0.150 | 0.950 |")
    assert summary[4].count("|  |") >= 1

    contrasts = rt.table_policy_contrasts(policy_contrasts_frame()).splitlines()
    assert len(contrasts) == 2 + 2 * 4
    assert contrasts[2] == "| JEV minus logistic regression | M1 review share | 0.012 [-0.010, 0.040] | 997 | 3 |"
    undefined = [line for line in contrasts if "JEV minus DistilBERT | M2" in line][0]
    assert undefined.endswith("|  | 0 | 1000 |")

    grid = rt.table_policy_grid(policy_grid_frame()).splitlines()
    assert len(grid) == 2 + 3 * 9
    assert sum("| primary |" in line for line in grid) == 3
    primary = [line for line in grid if "| primary |" in line][0]
    assert primary.startswith("| Logistic regression | 0.900 | 0.900 | primary |")


def test_policy_tables_are_skipped_when_their_files_or_columns_are_missing():
    frames = all_frames()
    frames["policy_summary"] = None
    frames["policy_contrasts"] = frames["policy_contrasts"].drop(columns=["resamples_left_out"])

    text, skipped = rt.build_results(frames)

    assert any(item.startswith("j. ") and "not found" in item for item in skipped)
    assert any(item.startswith("k. ") and "resamples_left_out" in item for item in skipped)
    assert "## i. " in text and "## l. " in text


def test_policy_per_label_table_joins_validation_and_test_values():
    lines = rt.table_policy_per_label(policy_per_label_frame(), policy_thresholds_frame()).splitlines()

    assert len(lines) == 2 + 3 * len(LABELS)
    assert lines[2] == (
        "| Logistic regression | toxic | 211 | 0.123 | 0.750 | yes | 0.903 | 0.900 | 0.904 | 0.600 | 0.046 | 0.023 |"
    )
    threat = [line for line in lines if line.startswith("| DistilBERT | threat")][0]
    assert threat == "| DistilBERT | threat | 211 | 0.123 | inf | no | 0.903 | 0.900 | - | - | 0.046 | 0.000 |"


def test_policy_per_label_table_is_skipped_when_a_source_is_missing():
    frames = all_frames()
    frames["policy_per_label"] = None
    text, skipped = rt.build_results(frames)
    assert any(item.startswith("m. ") for item in skipped)
    assert "## m. " in text
