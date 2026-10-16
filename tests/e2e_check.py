"""Checks of what the console showed in the end-to-end run (console/e2e), against the committed
files and the real policy engine. Not part of the normal suite: run it with the run's recording present.

    cd console && npm run e2e          (writes console/e2e/out/observations.json, then runs this file)
    python -m pytest tests/e2e_check.py

Everything is compared with the CSVs in artifacts/ or with src/policyjev/policy/engine.py, never with
numbers copied into this file. Test names start with test_sNN_ so that the report can file each result
under the scenario it belongs to.
"""

import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from policyjev.config import LABELS, REPO_ROOT
from policyjev.policy.engine import decide, label_severity
from policyjev.policy.thresholds import PRIMARY_PRECISION, PRIMARY_RECALL, thresholds_for

OBSERVATIONS = REPO_ROOT / "console" / "e2e" / "out" / "observations.json"
METRICS = REPO_ROOT / "artifacts" / "metrics"
SYSTEMS = ["logreg_word", "distilbert", "jev"]
NAMES = {"logreg_word": "Logistic regression", "distilbert": "DistilBERT", "jev": "JEV"}
IN_SENTENCE = {"logreg_word": "logistic regression", "distilbert": "DistilBERT", "jev": "JEV"}
DECISION_WORD = {"ALLOW": "Allow", "HUMAN REVIEW": "Human review", "ACTION": "Action"}
MINUS = "−"
# Chrome writes inline percentages to about four decimals, so a width read back from the page is exact only to that
PERCENT_TOLERANCE = 5e-4

pytestmark = pytest.mark.skipif(not OBSERVATIONS.is_file(), reason="no recording from an end-to-end run")


@pytest.fixture(scope="module")
def obs() -> dict:
    return json.loads(OBSERVATIONS.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def table() -> pd.DataFrame:
    return pd.read_csv(REPO_ROOT / "artifacts" / "policy" / "policy_thresholds.csv")


@pytest.fixture(scope="module")
def primary(table) -> pd.DataFrame:
    return table[table["is_primary"].astype(str).str.lower() == "true"]


@pytest.fixture(scope="module")
def summary() -> pd.DataFrame:
    return pd.read_csv(METRICS / "policy_test_summary.csv")


@pytest.fixture(scope="module")
def per_label() -> pd.DataFrame:
    return pd.read_csv(METRICS / "policy_test_per_label.csv")


@pytest.fixture(scope="module")
def contrasts() -> pd.DataFrame:
    return pd.read_csv(METRICS / "policy_test_contrasts.csv")


@pytest.fixture(scope="module")
def comparison() -> pd.DataFrame:
    return pd.read_csv(METRICS / "test_comparison_with_stack.csv")


@pytest.fixture(scope="module")
def budget() -> pd.DataFrame:
    return pd.read_csv(METRICS / "review_budget_test_with_stack.csv")


def fx(value: float, places: int) -> str:
    """Round half up on the exact value of the double, as JavaScript's toFixed does (the UI's one rule).

    Python's format rounds an exact tie to even (0.03125 gives 0.0312); toFixed gives 0.0313.
    """
    from decimal import ROUND_HALF_UP, Decimal

    quantum = Decimal(1).scaleb(-places)
    return str(Decimal(float(value)).quantize(quantum, rounding=ROUND_HALF_UP))


def engine_table(table, system):
    return thresholds_for(table, system, PRIMARY_RECALL, PRIMARY_PRECISION)


def words(label: str) -> str:
    return label.replace("_", " ")


def join_list(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def signed(value: float) -> str:
    if value > 0:
        return f"+{fx(value, 3)}"
    if value < 0:
        return f"{MINUS}{fx(abs(value), 3)}"
    return f"{fx(0, 3)}"


def mentions(text: str, label: str) -> bool:
    if label == "toxic":
        return re.search(r"(?<!severe )\btoxic\b", text) is not None
    return re.search(rf"\b{re.escape(words(label))}\b", text) is not None


def js_round(x: float) -> int:
    return math.floor(x + 0.5)


def rounded(value: float, places: int) -> str:
    return f"{value:.{places}f}"


# ---------------------------------------------------------------- S04 verdicts

def case_ids() -> list[str]:
    if not OBSERVATIONS.is_file():
        return []
    return [c["id"] for c in json.loads(OBSERVATIONS.read_text(encoding="utf-8")).get("verdictCases", [])]


@pytest.mark.parametrize("case_id", case_ids())
def test_s04_verdict_equals_the_engine(obs, table, record_property, case_id):
    case = next(c for c in obs["verdictCases"] if c["id"] == case_id)
    shown = []
    for result in case["api"]["results"]:
        system = result["system"]
        engine = decide(result["probabilities"], engine_table(table, system))
        tile = case["ui"]["tiles"][system]
        assert tile["decision"] == engine.decision, f"{system}: screen {tile['decision']}, engine {engine.decision}"
        assert tile["word"] == DECISION_WORD[engine.decision], f"{system}: tile word {tile['word']!r}"
        assert result["decision"] == engine.decision, f"{system}: API {result['decision']}, engine {engine.decision}"
        assert result["triggering_labels"] == list(engine.triggering_labels)
        why = tile["why"]
        if engine.decision == "ALLOW":
            assert why == "No label reached its review threshold."
        else:
            verb = "Action" if engine.decision == "ACTION" else "Review"
            assert why.startswith(f"{verb} triggered by "), why
            for label in LABELS:
                assert mentions(why, label) == (label in engine.triggering_labels), f"{system}: {label!r} in {why!r}"
        shown.append(f"{system}={engine.decision}")
    record_property("observed", "screen = API = engine for " + ", ".join(shown))


@pytest.mark.parametrize("case_id", case_ids())
def test_s04_probabilities_are_in_range(obs, record_property, case_id):
    case = next(c for c in obs["verdictCases"] if c["id"] == case_id)
    values = [p for r in case["api"]["results"] for p in r["probabilities"].values()]
    assert len(values) == 18
    assert all(isinstance(p, float) or isinstance(p, int) for p in values)
    assert all(math.isfinite(p) and 0.0 <= p <= 1.0 for p in values), (min(values), max(values))
    record_property("observed", f"18 probabilities in [{fx(min(values), 4)}, {fx(max(values), 4)}]")


@pytest.mark.parametrize("case_id", case_ids())
def test_s04_numbers_and_meters_match_the_response(obs, record_property, case_id):
    case = next(c for c in obs["verdictCases"] if c["id"] == case_id)
    checked = 0
    for result in case["api"]["results"]:
        system = result["system"]
        for label, p in result["probabilities"].items():
            cell = case["ui"]["cells"][f"{system}|{label}"]
            assert cell["val"] == f"{fx(p, 4)}", f"{system} {label}: shows {cell['val']}, response {fx(p, 4)}"
            lit = js_round(math.sqrt(p) * 40)
            assert cell["lit"] == lit, f"{system} {label}: {cell['lit']} cells lit, expected {lit} for p={fx(p, 4)}"
            assert cell["peak"] == (1 if lit > 0 else 0)
            checked += 1
        ms = case["ui"]["tiles"][system]["ms"]
        assert ms == f"{fx(result['time_ms'], 1)} ms", f"{system}: tile shows {ms!r}, response {result['time_ms']}"
    record_property("observed", f"{checked} numbers equal the response to 4 decimals, meters lit to round(sqrt(p) x 40), times shown as returned")


def test_s04_meter_colours_follow_the_thresholds(obs, table, record_property):
    case = obs["verdictCases"][0]
    cells = 0
    for system in SYSTEMS:
        rules = engine_table(table, system).set_index("label")
        enabled = primary_enabled(table, system)
        for label in LABELS:
            low, high = rules.loc[label, "low"], rules.loc[label, "high"]
            high = high if enabled[label] else math.inf
            zones = case["ui"]["cells"][f"{system}|{label}"]["zones"]
            assert len(zones) == 40
            expected = [("za", "zr", "zx")[label_severity(((i + 0.5) / 40) ** 2, low, high)] for i in range(40)]
            assert zones == expected, f"{system} {label}"
            cells += 40
    record_property("observed", f"{cells} meter cells coloured as the engine would decide for the probability at the centre of each cell")


def primary_enabled(table, system) -> dict[str, bool]:
    rows = table[(table["system"] == system) & (table["is_primary"].astype(str).str.lower() == "true")]
    return {r["label"]: str(r["action_enabled"]).lower() == "true" for _, r in rows.iterrows()}


def test_s04_the_jev_stub_answers_the_same_every_time(obs, record_property):
    answers = {json.dumps([r for r in c["api"]["results"] if r["system"] == "jev"][0]["probabilities"], sort_keys=True) for c in obs["verdictCases"]}
    assert len(answers) == 1, "the stub is meant to return fixed probabilities"
    record_property("observed", "the stub JEV returned identical probabilities for all inputs")


# ---------------------------------------------------------------- S06 Policy numbers

def test_s06_api_results_equal_the_csvs(obs, primary, summary, per_label, contrasts, record_property):
    api = obs["api"]["results"]
    for _, row in summary.iterrows():
        shown = next(r for r in api["policy_summary"] if r["system"] == row["system"])
        for column in ["n_rows", "n_allow", "n_review", "n_action"]:
            assert shown[column] == row[column]
        for column in ["m1_review_share", "m2_action_precision", "m3_miss_rate", "m4_rare_miss_rate", "action_coverage"]:
            assert shown[column] == pytest.approx(row[column], abs=1e-12)
    for _, row in contrasts.iterrows():
        shown = next(r for r in api["policy_contrasts"] if r["contrast"] == row["contrast"] and r["metric"] == row["metric"])
        for column in ["difference", "ci_low", "ci_high"]:
            assert shown[column] == pytest.approx(row[column], abs=1e-12)
    for _, row in per_label.iterrows():
        shown = next(r for r in api["policy_per_label"] if r["system"] == row["system"] and r["label"] == row["label"])
        threshold = primary[(primary["system"] == row["system"]) & (primary["label"] == row["label"])].iloc[0]
        assert shown["test_recall_at_low"] == pytest.approx(1 - row["miss_rate_label"], abs=1e-12)
        assert shown["validation_recall_at_low"] == pytest.approx(threshold["validation_recall_at_low"], abs=1e-12)
        if not math.isnan(row["action_precision_label"]):
            assert shown["test_action_precision"] == pytest.approx(row["action_precision_label"], abs=1e-12)
            assert shown["validation_precision_at_high"] == pytest.approx(threshold["validation_precision_at_high"], abs=1e-12)
        else:
            assert shown["test_action_precision"] is None
    api_thresholds = obs["api"]["thresholds"]
    assert len(api_thresholds) == 18
    for row in api_thresholds:
        csv = primary[(primary["system"] == row["system"]) & (primary["label"] == row["label"])].iloc[0]
        assert row["low"] == pytest.approx(csv["low"], abs=1e-12)
        assert (row["high"] is None) == (not math.isfinite(csv["high"]))
    record_property("observed", "the API's summary, contrasts, per-label and threshold rows equal the CSVs to 1e-12")


def explorer_params() -> list[tuple[str, str]]:
    return [(s, label) for s in SYSTEMS for label in LABELS]


@pytest.mark.parametrize("system,label", explorer_params())
def test_s06_explorer_thresholds_equal_the_csv(obs, primary, record_property, system, label):
    explorer = obs["policy"]["explorer"][system]
    row = next(r for r in explorer["rows"] if r["label"] == label)
    csv = primary[(primary["system"] == system) & (primary["label"] == label)].iloc[0]
    enabled = str(csv["action_enabled"]).lower() == "true"
    assert explorer["system"] == NAMES[system]
    assert f"LOW {fx(csv['low'], 4)}" in row["scale"], row["scale"]
    if enabled:
        assert f"HIGH {fx(csv['high'], 4)}" in row["scale"], row["scale"]
        assert row["note"] == "Can trigger action"
    else:
        assert "no HIGH threshold" in row["scale"] and "HIGH 0" not in row["scale"]
        assert row["note"] == "Cannot trigger action"
    # zone bar widths: the square-root scale
    widths = {w["zone"]: w["width"] for w in row["widths"]}
    low_at = math.sqrt(csv["low"]) * 100
    assert widths["allow"] == pytest.approx(low_at, abs=PERCENT_TOLERANCE)
    if enabled:
        high_at = math.sqrt(csv["high"]) * 100
        assert widths["review"] == pytest.approx(high_at - low_at, abs=PERCENT_TOLERANCE)
        assert widths["action"] == pytest.approx(100 - high_at, abs=PERCENT_TOLERANCE)
    else:
        assert widths["review"] == pytest.approx(100 - low_at, abs=PERCENT_TOLERANCE) and "action" not in widths
    # at the page's starting probability the row decides as the engine does
    p = (explorer["slider"] / 1000) ** 2
    expected = DECISION_WORD[("ALLOW", "HUMAN REVIEW", "ACTION")[label_severity(p, csv["low"], csv["high"] if enabled else math.inf)]]
    assert row["chip"] == expected
    record_property("observed", f"{NAMES[system]} {label}: {row['scale']}; {row['note']}; at p={fx(p, 4)} {row['chip']}")


def test_s06_flow_counts_and_percentages(obs, summary, record_property):
    charts = obs["policy"]["charts"]
    totals = set()
    for _, row in summary.iterrows():
        flow = next(f for f in charts["flows"] if f["system"] == row["system"])
        n = {"ALLOW": row["n_allow"], "HUMAN REVIEW": row["n_review"], "ACTION": row["n_action"]}
        total = sum(n.values())
        totals.add(total)
        assert total == row["n_rows"] - row["n_failed_rows"]
        legend = flow["legend"]
        assert f"Allow {row['n_allow']:,}" in legend
        assert f"Human review {row['n_review']:,} ({fx(row['n_review'] / total * 100, 1)}%)" in legend
        assert f"Action {row['n_action']:,} ({fx(row['n_action'] / total * 100, 1)}%)" in legend
        widths = 0.0
        for bar in flow["bars"]:
            share = n[bar["flow"]] / total * 100
            assert bar["width"] == pytest.approx(share, abs=PERCENT_TOLERANCE)
            assert bar["text"] in ("", fx(share, 1) + "%"), (flow["system"], bar)  # printed only where it fits, and then exactly
            widths += bar["width"]
        assert widths == pytest.approx(100, abs=PERCENT_TOLERANCE)
    assert totals == {int(summary["n_rows"].iloc[0])}, "every system has the same number of comments"
    sections = charts["sections"]
    assert f"All {int(summary['n_rows'].iloc[0]):,} labelled test comments" in sections[1]["sub"]
    record_property("observed", f"counts sum to {int(summary['n_rows'].iloc[0]):,} for each of the 3 systems; bar widths sum to 100; percentages printed to 1 decimal")


def test_s06_five_measures(obs, summary, record_property):
    charts = obs["policy"]["charts"]
    columns = {
        "m1_review_share": "min", "m3_miss_rate": "min", "m4_rare_miss_rate": "min",
        "m2_action_precision": "max", "action_coverage": "max",
    }
    assert [m["key"] for m in charts["measures"]] == list(columns)
    for measure in charts["measures"]:
        values = {r["system"]: r[measure["key"]] for _, r in summary.iterrows()}
        shown = {v["system"]: v["text"] for v in measure["values"]}
        for system in SYSTEMS:
            assert shown[system] == f"{fx(values[system], 3)}", (measure["key"], system, shown[system])
        best = (min if columns[measure["key"]] == "min" else max)(values, key=values.get)
        flagged = [v["system"] for v in measure["values"] if v["best"]]
        assert flagged == [best], (measure["key"], flagged, best)
        dots = {d["system"]: d["f"] for d in measure["dots"]}
        assert [d["system"] for d in measure["dots"] if d["best"]] == [best]
        # the dots are at the values on one axis: position / value is the same for the three
        ratios = [dots[s] / values[s] for s in SYSTEMS]
        assert max(ratios) == pytest.approx(min(ratios), rel=1e-6), (measure["key"], ratios)
        # the end values printed under the track: 0 on the left, the value the dots are scaled to on the right
        axis_max = values[SYSTEMS[0]] / dots[SYSTEMS[0]]
        assert measure["scale"] == ["0", fx(axis_max, 2)], (measure["key"], measure["scale"], axis_max)
        assert axis_max >= max(values.values()), "every value is on its own scale"
    record_property("observed", "15 values equal policy_test_summary.csv to 3 decimals; the best of each measure is flagged; dots on one axis")


def test_s06_comparison_rows_values_intervals_and_sign(obs, contrasts, summary, record_property):
    charts = obs["policy"]["charts"]
    plan = [("m1_review_share", "lower"), ("m3_miss_rate", "lower"), ("m4_rare_miss_rate", "lower"), ("m2_action_precision", "higher")]
    baselines = [("jev - logreg_word", "logreg_word"), ("jev - distilbert", "distilbert")]
    assert [row["measure"] for row in charts["comparison"]] == [m for m, _ in plan]
    ks = []
    flipped_all = []
    for row, (metric, better) in zip(charts["comparison"], plan):
        for index, (contrast, baseline) in enumerate(baselines):
            csv = contrasts[(contrasts["contrast"] == contrast) & (contrasts["metric"] == metric)].iloc[0]
            if better == "lower":
                value, low, high = -csv["difference"], -csv["ci_high"], -csv["ci_low"]
            else:
                value, low, high = csv["difference"], csv["ci_low"], csv["ci_high"]
            flipped_all.append((low, high))
            line = row["lines"][index]
            assert line["b"] == signed(value), (metric, baseline, line["b"], signed(value))
            assert f"[{signed(low)}, {signed(high)}]" in line["full"], (metric, baseline, line["full"])
            # JEV worse (red side, left of the middle line) exactly when its own value is worse than the baseline's
            jev = summary[summary["system"] == "jev"][metric if metric != "m2_action_precision" else "m2_action_precision"].iloc[0]
            other = summary[summary["system"] == baseline][metric].iloc[0]
            worse = jev > other if better == "lower" else jev < other
            dot = row["dots"][index]
            assert (dot < 50) == worse, f"{metric} vs {baseline}: dot at {dot}%, JEV worse={worse}"
            assert (value < 0) == worse
            ks.append((dot - 50) / value)
    assert max(ks) == pytest.approx(min(ks), rel=2e-3), "dots are on one scale around the middle line"
    touching = [(lo, hi) for lo, hi in flipped_all if lo <= 0 <= hi]
    assert not touching
    record_property("observed", f"8 lines match the CSV after flipping lower-is-better measures; every dot is left of the middle exactly when JEV is worse (red = JEV worse); {len(flipped_all)} intervals, none touches zero")


def test_s06_dumbbells_values_changes_and_targets(obs, primary, per_label, record_property):
    both = {"precision": obs["policy"]["charts"]["dumbbells"]["precision"], "recall": obs["policy"]["chartsRecall"]["dumbbells"]["recall"]}
    ticks = obs["policy"]["chartsRecall"]["dumbbellTicks"]
    axis_min = float(ticks[0])
    checked = 0
    for measure, rows in both.items():
        expected = {}
        for _, row in per_label.iterrows():
            threshold = primary[(primary["system"] == row["system"]) & (primary["label"] == row["label"])].iloc[0]
            if measure == "precision":
                if not (str(row["action_enabled"]).lower() == "true" and not math.isnan(row["action_precision_label"])):
                    continue
                validation, test = threshold["validation_precision_at_high"], row["action_precision_label"]
                target = threshold["precision_target"]
            else:
                validation, test = threshold["validation_recall_at_low"], 1 - row["miss_rate_label"]
                target = threshold["recall_target"]
            expected[f"{row['system']}|{row['label']}"] = (validation, test, target)
        assert sorted(r["key"] for r in rows) == sorted(expected), measure
        for row in rows:
            validation, test, target = expected[row["key"]]
            assert row["validation"] == f"Validation {fx(validation, 3)}", (row["key"], row["validation"])
            assert row["test"] == f"Test {fx(test, 3)}", (row["key"], row["test"])
            change = test - validation
            assert row["change"] == signed(change) if change != 0 else True
            if change >= 0:
                assert row["change"] == f"+{fx(change, 3)}", (row["key"], row["change"], change)
            else:
                assert row["change"] == f"{MINUS}{fx(abs(change), 3)}", (row["key"], row["change"], change)
            assert row["up"] == (change >= 0)
            position = lambda v: (v - axis_min) / (1 - axis_min) * 100
            assert row["positions"][0] == pytest.approx(position(validation), abs=PERCENT_TOLERANCE)
            assert row["positions"][1] == pytest.approx(position(test), abs=PERCENT_TOLERANCE)
            assert row["target"] == pytest.approx(position(target), abs=PERCENT_TOLERANCE)
            checked += 1
    record_property("observed", f"{checked} rows: validation, test and change equal the CSVs to 3 decimals; positions on the axis from {axis_min}; the dashed line at the target")


def test_s06_summary_sentences_are_true(obs, summary, per_label, primary, contrasts, record_property):
    charts = obs["policy"]["charts"]
    sections = charts["sections"]
    by = {r["system"]: r for _, r in summary.iterrows()}

    def best(column, which):
        pick = min if which == "min" else max
        return pick(by, key=lambda s: by[s][column])

    # heading and lead of the test-set section
    headline, detail = sections[1]["h2"], sections[1]["sub"]
    least, fewest = best("m1_review_share", "min"), best("m3_miss_rate", "min")
    expected_headline = (
        f"On the test set, {NAMES[least]} had the lowest review share and the lowest share of harmful comments allowed"
        if least == fewest
        else f"On the test set, {NAMES[least]} had the lowest review share and {NAMES[fewest]} the lowest share of harmful comments allowed"
    )
    assert headline == expected_headline
    most_review, most_missed, most_precise = best("m1_review_share", "max"), best("m3_miss_rate", "max"), best("m2_action_precision", "max")
    if most_review == most_missed == most_precise:
        assert f"{NAMES[most_review]} needed the most review and missed the most. Its automatic actions had the highest precision" in detail
    else:
        assert f"{NAMES[most_review]} needed the most review" in detail

    # only JEV was contrasted with each baseline (protocol, Amendment 3): the sentence says so, and the contrasts agree
    assert "Only JEV was compared with each baseline; the difference between logistic regression and DistilBERT was not tested." in detail
    assert set(contrasts["contrast"]) == {"jev - logreg_word", "jev - distilbert"}
    assert set(contrasts["metric"]) == {"m1_review_share", "m2_action_precision", "m3_miss_rate", "m4_rare_miss_rate"}
    # each system automates its own labels, so the caveat and the footnote are due
    enabled_sets = {
        system: [r["label"] for _, r in per_label[per_label["system"] == system].iterrows() if str(r["action_enabled"]).lower() == "true"]
        for system in SYSTEMS
    }
    assert len({tuple(v) for v in enabled_sets.values()}) > 1
    assert "but each system automates a different set of labels" in detail
    footnote = next(n for n in charts["notes"] if n["note"] == "automation")["text"]
    for system, labels in enabled_sets.items():
        assert f"{IN_SENTENCE[system]} on {join_list([words(l) for l in labels])}" in footnote, (system, footnote)
    assert charts["flags"] == ["m2_action_precision", "action_coverage"]
    assert charts["comparisonHeader"].endswith("JEV minus baseline, signed so that negative means JEV is worse.")

    # the three readings of the plots
    reads = [r["text"] for r in charts["reads"]]
    enabled_rows = per_label[(per_label["action_enabled"].astype(str).str.lower() == "true") & per_label["action_precision_label"].notna()]
    val = lambda r: primary[(primary["system"] == r["system"]) & (primary["label"] == r["label"])].iloc[0]["validation_precision_at_high"]
    fell, held = [], []
    for label in LABELS:
        rows = enabled_rows[enabled_rows["label"] == label]
        if rows.empty:
            continue
        verdicts = {row["action_precision_label"] < val(row) for _, row in rows.iterrows()}
        if verdicts == {True}:
            fell.append(label)
        elif verdicts == {False}:
            held.append(label)
    rng = lambda xs: (f"{fx(min(xs), 2)}" if f"{fx(min(xs), 2)}" == f"{fx(max(xs), 2)}" else f"{fx(min(xs), 2)} to {fx(max(xs), 2)}")
    fell_rows = enabled_rows[enabled_rows["label"].isin(fell)]
    systems_with_all = all(len(enabled_rows[enabled_rows["label"] == label]) == 3 for label in fell)
    assert reads[0] == (
        f"Precision at the action threshold fell on {join_list([words(l) for l in fell])} "
        f"{'for all three systems' if systems_with_all else 'where it is on'}: "
        f"{rng([val(r) for _, r in fell_rows.iterrows()])} on validation, {rng(list(fell_rows['action_precision_label']))} on test."
    ), reads[0]
    held_rows = enabled_rows[enabled_rows["label"].isin(held)]
    first = join_list([words(l) for l in held])
    assert reads[1] == f"{first[0].upper()}{first[1:]} held where automatic action is on: {rng(list(held_rows['action_precision_label']))} on test.", reads[1]

    # recall: the sentence built from the CSV, and the same counts read out of docs/results_policy.md
    parts = []
    counts_from_csv = {}
    for index, system in enumerate(SYSTEMS):
        rows = per_label[per_label["system"] == system]
        recall = {r["label"]: 1 - r["miss_rate_label"] for _, r in rows.iterrows()}
        target = primary[primary["system"] == system]["recall_target"].iloc[0]
        below = sum(1 for v in recall.values() if v < target)
        lowest = min(recall, key=recall.get)
        counts_from_csv[system] = (below, lowest, recall[lowest])
        parts.append(f"{below} of {len(recall)}{' labels' if index == 0 else ''} for {IN_SENTENCE[system]} (lowest {fx(recall[lowest], 3)}, {words(lowest)})")
    target = primary["recall_target"].iloc[0]
    assert reads[2] == f"Recall at the review threshold was below the {fx(target, 2)} target on {parts[0]}, {parts[1]} and {parts[2]}.", reads[2]
    doc = (REPO_ROOT / "docs" / "results_policy.md").read_text(encoding="utf-8")
    block = doc.split("Recall at LOW (about 0.90 on validation by construction):")[1].split("ACTION precision on test")[0]
    table = {}
    for line in block.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 4 and cells[0] in LABELS:
            table[cells[0]] = [float(c) for c in cells[1:]]
    assert len(table) == 6
    for column, system in enumerate(SYSTEMS):
        values = {label: row[column] for label, row in table.items()}
        below = sum(1 for v in values.values() if v < target)
        lowest = min(values, key=values.get)
        assert (below, lowest, round(values[lowest], 3)) == (counts_from_csv[system][0], counts_from_csv[system][1], round(counts_from_csv[system][2], 3)), system

    # the comparison sentence about the middle line
    flipped = []
    for _, row in contrasts.iterrows():
        lower_better = row["metric"] != "m2_action_precision"
        lo, hi = (-row["ci_high"], -row["ci_low"]) if lower_better else (row["ci_low"], row["ci_high"])
        flipped.append((lo, hi))
    note = "No interval touches the middle line." if all(not (lo <= 0 <= hi) for lo, hi in flipped) else f"{sum(1 for lo, hi in flipped if lo <= 0 <= hi)} of {len(flipped)} intervals touch the middle line."
    assert note in sections[2]["sub"]

    # the paragraph on trust
    trust = charts["trust"]
    recall_t, precision_t = primary["recall_target"].iloc[0], primary["precision_target"].iloc[0]
    assert trust.startswith(f"The {fx(recall_t, 2)} recall and precision targets" if recall_t == precision_t else f"The {fx(recall_t, 2)} recall and {fx(precision_t, 2)} precision targets")
    record_property("observed", f"headline names {NAMES[least]}; readings: {reads[2][:120]}...")


# ---------------------------------------------------------------- S07 explorer

def explorer_cases() -> list[dict]:
    if not OBSERVATIONS.is_file():
        return []
    return json.loads(OBSERVATIONS.read_text(encoding="utf-8")).get("explorerCases", [])


@pytest.mark.parametrize("index", range(len(explorer_cases())))
def test_s07_explorer_equals_the_engine(obs, table, primary, record_property, index):
    case = obs["explorerCases"][index]
    system, p = case["system"], case["p"]
    assert p == (case["v"] / 1000) ** 2
    rules = engine_table(table, system)
    enabled = primary_enabled(table, system)
    ui = case["ui"]
    assert ui["slider"] == case["v"]
    assert ui["probabilityText"] == f"{fx(p, 4)}"
    # the probe is on the right side of the threshold it targets
    target = rules.set_index("label").loc[case["label"], case["edge"]]
    if case["side"] == "below":
        assert p < target, (p, target)
    else:
        assert p >= target, (p, target)
    shown = []
    for label in LABELS:
        row = next(r for r in ui["rows"] if r["label"] == label)
        low, high = rules.set_index("label").loc[label, ["low", "high"]]
        severity = label_severity(p, low, high if enabled[label] else math.inf)
        expected = ("ALLOW", "HUMAN REVIEW", "ACTION")[severity]
        assert row["decision"] == expected, f"{label}: row {row['decision']}, engine {expected}"
        assert row["chip"] == DECISION_WORD[expected], f"{label}: chip {row['chip']!r}"
        if not enabled[label]:
            assert row["chip"] != "Action"
        shown.append(f"{label}={DECISION_WORD[expected]}")
    outcome = decide({label: p for label in LABELS}, rules.assign(high=[h if enabled[l] else math.inf for l, h in zip(rules["label"], rules["high"])]))
    assert ui["outcome"]["decision"] == outcome.decision, f"capsule {ui['outcome']['decision']}, engine {outcome.decision}"
    # the capsule: the decision word, then the same sentence the Score tiles use
    if outcome.decision == "ALLOW":
        why = "No label reached its review threshold."
    else:
        verb = "Action" if outcome.decision == "ACTION" else "Review"
        why = f"{verb} triggered by {join_list([words(l) for l in outcome.triggering_labels])}."
    assert ui["outcome"]["text"] == DECISION_WORD[outcome.decision] + why, (ui["outcome"]["text"], why)
    assert "most severe" not in ui["outcome"]["text"]
    assert ui["outcome"]["live"] == "polite"
    record_property("observed", f"{case['method']}: position {case['v']}, p={fx(p, 6)} ({case['side']} {case['label']} {case['edge'].upper()}={fx(target, 6)}): " + ", ".join(shown) + f"; capsule {DECISION_WORD[outcome.decision]}")


# ---------------------------------------------------------------- S08 Results

def test_s08_ap_bars_intervals_and_labels(obs, comparison, record_property):
    r = obs["results"]
    by = {row["label"]: row for _, row in comparison.iterrows()}
    ratios = []
    for bar in r["bars"]:
        row = by[bar["label"]]
        ap, lo, hi = row[f"ap_{bar['system']}"], row[f"ap_{bar['system']}_lo"], row[f"ap_{bar['system']}_hi"]
        assert bar["title"] == f"{NAMES[bar['system']]} {fx(ap, 3)} [{fx(lo, 3)}, {fx(hi, 3)}]", bar["title"]
        tag = f"{fx(ap, 2)}"
        assert bar["tag"] == fx(ap, 3)  # one rule for value labels: leading zero, three decimals
        assert bar["whiskerBottom"] == pytest.approx(lo / ap * 100, abs=PERCENT_TOLERANCE)
        assert bar["whiskerHeight"] == pytest.approx((hi - lo) / ap * 100, abs=PERCENT_TOLERANCE)
        ratios.append(bar["height"] / ap)
    assert max(ratios) == pytest.approx(min(ratios), rel=1e-5), "bar heights are proportional to AP on one axis"
    assert [g["label"] for g in r["groups"]] == LABELS
    for g in r["groups"]:
        assert g["name"] == words(g["label"])
        assert g["positives"] == f"{int(by[g['label']]['positives']):,} positives"
    record_property("observed", f"{len(r['bars'])} bars: AP and 95% interval equal the CSV to 3 decimals; heights proportional to AP; {len(r['groups'])} positive counts match")


def test_s08_budget_points_equal_the_csv(obs, budget, record_property):
    r = obs["results"]
    rows = budget[budget["metric"] == "recall_rare"]
    levels = sorted(rows["budget_pct"].unique())
    assert sorted({p["level"] for p in r["points"]}) == levels
    top, bottom = 16, 40
    inner_height = r["svg"]["height"] - top - bottom
    for point in r["points"]:
        csv = rows[(rows["comparison"] == point["system"]) & (rows["budget_pct"] == point["level"])].iloc[0]["value"]
        assert point["value"] == pytest.approx(csv, abs=1e-12)
        assert point["title"] == f"{NAMES[point['system']]} at {int(point['level'])}%: {fx(csv, 3)}"
        assert point["cy"] == pytest.approx(top + inner_height * (1 - csv), abs=0.01)
    for system in SYSTEMS:
        xs = [p["cx"] for p in sorted(r["points"], key=lambda p: p["level"]) if p["system"] == system]
        assert all(b > a for a, b in zip(xs, xs[1:])), "budgets run left to right"
    record_property("observed", f"{len(r['points'])} points equal review_budget_test_with_stack.csv (recall_rare) and sit at the right height")


def relation(lo: float, hi: float) -> str:
    return "ahead" if lo > 0 else "behind" if hi < 0 else "level"


def test_s08_sentences_are_true_statements(obs, comparison, budget, record_property):
    r = obs["results"]
    by = {row["label"]: row for _, row in comparison.iterrows()}
    ap_sub, budget_h2, budget_sub = r["sections"][0]["sub"], r["sections"][1]["h2"], r["sections"][1]["sub"]

    # which labels JEV trails, from the paired intervals of the CSV
    relations = {}
    for label in LABELS:
        row = by[label]
        relations[label] = (
            relation(row["diff_ap_jev_minus_logreg_word_lo"], row["diff_ap_jev_minus_logreg_word_hi"]),
            relation(row["diff_ap_jev_minus_distilbert_lo"], row["diff_ap_jev_minus_distilbert_hi"]),
        )
    behind = [l for l in LABELS if relations[l] == ("behind", "behind")]
    ahead = [l for l in LABELS if relations[l] == ("ahead", "ahead")]
    level = [l for l in LABELS if relations[l] == ("level", "level")]
    mixed = [l for l in LABELS if relations[l][0] != relations[l][1]]
    phrase = {"ahead": "ahead of", "behind": "behind", "level": "level with"}
    if behind:
        assert f"JEV trails both baselines on {join_list([words(l) for l in behind])}." in ap_sub, (behind, ap_sub)
    if ahead:
        assert f"JEV leads both baselines on {join_list([words(l) for l in ahead])}." in ap_sub
    if level:
        assert f"On {join_list([words(l) for l in level])} the test set cannot separate it from either baseline." in ap_sub
    for label in mixed:
        assert f"On {words(label)} it is {phrase[relations[label][0]]} logistic regression and {phrase[relations[label][1]]} DistilBERT." in ap_sub
    claimed = sum(1 for l in LABELS if mentions(ap_sub, l))
    assert claimed >= 6 - 0, "every label is accounted for in the sentence"

    # the review budget: relation at each budget from the CSV, and the order of the three systems
    rows = budget[budget["metric"] == "recall_rare"]
    levels = sorted(rows["budget_pct"].unique())
    steps = []
    orderings = {}
    for level_pct in levels:
        pick = lambda name: rows[(rows["comparison"] == name) & (rows["budget_pct"] == level_pct)].iloc[0]
        lr, db = pick("jev - logreg_word"), pick("jev - distilbert")
        steps.append((int(level_pct), relation(lr["lo"], lr["hi"]), relation(db["lo"], db["hi"])))
        values = {s: pick(s)["value"] for s in SYSTEMS}
        orderings[int(level_pct)] = sorted(SYSTEMS, key=lambda s: -values[s])
    groups = []
    for pct, a, b in steps:
        if groups and groups[-1][1:] == [a, b]:
            groups[-1][0].append(pct)
        else:
            groups.append([[pct], a, b])
    sentences = []
    for index, (pcts, a, b) in enumerate(groups):
        at = join_list([f"{p}%" for p in pcts])
        subject = "JEV" if index == 0 else "It"
        if a == b:
            verb = {"ahead": "leads both baselines", "behind": "trails both baselines", "level": "is level with both baselines"}[a]
            sentences.append(f"{subject} {verb} at {at}.")
        else:
            lead = "JEV" if index == 0 else "it"
            sentences.append(f"At {at} {lead} is {phrase[a]} logistic regression and {phrase[b]} DistilBERT.")
    assert " ".join(sentences) in budget_sub, (" ".join(sentences), budget_sub)
    lowest_budget = levels[0]
    ahead_at_smallest = steps[0][1] == "ahead" and steps[0][2] == "ahead"
    ahead_both = [a == "ahead" and b == "ahead" for _, a, b in steps]
    prefix = ahead_both.index(False) if False in ahead_both else len(ahead_both)
    if prefix == 0 or any(ahead_both[prefix:]):
        expected_h2 = "Recall of rare-label comments as the review budget grows"
    elif prefix == len(ahead_both):
        expected_h2 = "JEV has higher rare-label recall at every review budget"
    else:
        expected_h2 = "JEV has higher rare-label recall only at the smallest review budgets"
    assert budget_h2 == expected_h2, (budget_h2, expected_h2, ahead_both)
    if budget_h2.endswith("only at the smallest review budgets"):
        assert ahead_both[:prefix] == [True] * prefix and not any(ahead_both[prefix:])  # the heading holds for every budget it covers
    assert "are spaced evenly along the axis, not to scale" in r["spacingNote"]
    assert r["yTitle"] == "Share of rare-label comments caught"
    assert r["sections"][0]["h2"] == "Average precision by label"

    # the order of the lines on the chart matches the order of the values at every budget
    for level_pct, expected in orderings.items():
        points = [p for p in r["points"] if p["level"] == level_pct]
        by_height = [p["system"] for p in sorted(points, key=lambda p: p["cy"])]  # smaller cy = higher value
        assert by_height == expected, (level_pct, by_height, expected)
    order_text = "; ".join(f"{pct}%: " + " > ".join(NAMES[s] for s in orderings[pct]) for pct in sorted(orderings))
    # the direct labels name the highest and the lowest system at the smallest budget
    first = {s: rows[(rows["comparison"] == s) & (rows["budget_pct"] == lowest_budget)].iloc[0]["value"] for s in SYSTEMS}
    top, low = max(first, key=first.get), min(first, key=first.get)
    labels = {l["system"]: l["text"] for l in r["directLabels"]}
    short = NAMES  # the direct labels use the same names as the legend
    assert set(labels) == set(SYSTEMS), "all three lines are labelled"
    for system in SYSTEMS:
        assert labels[system] == f"{short[system]} {fx(first[system], 3)}", (system, labels[system])
    record_property("observed", f"JEV trails both on {', '.join(behind)}; budget claims: {' '.join(sentences)} Order of recall: {order_text}")


# ---------------------------------------------------------------- S16 contrast

def contrast_scenes() -> list[tuple[str, int]]:
    if not OBSERVATIONS.is_file():
        return []
    return [(s["scene"], s["width"]) for s in json.loads(OBSERVATIONS.read_text(encoding="utf-8")).get("contrast", [])]


def srgb_to_linear(channel: np.ndarray) -> np.ndarray:
    c = channel / 255.0
    return np.where(c <= 0.03928, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def luminance(rgb: np.ndarray) -> np.ndarray:
    return 0.2126 * srgb_to_linear(rgb[..., 0]) + 0.7152 * srgb_to_linear(rgb[..., 1]) + 0.0722 * srgb_to_linear(rgb[..., 2])


@pytest.mark.parametrize("scene,width", contrast_scenes())
def test_s16_text_contrast_on_the_rendered_background(obs, record_property, scene, width):
    entry = next(s for s in obs["contrast"] if s["scene"] == scene and s["width"] == width)
    image = np.asarray(Image.open(entry["png"]).convert("RGB"), dtype=float)
    failures = []
    checked = 0
    lowest = (math.inf, "")
    for item in entry["items"]:
        x, y, w, h = item["rect"]
        x0, y0, x1, y1 = int(round(x)) + 1, int(round(y)) + 1, int(round(x + w)) - 1, int(round(y + h)) - 1
        x0, y0 = max(x0, 0), max(y0, 0)
        x1, y1 = min(x1, image.shape[1]), min(y1, image.shape[0])
        if x1 <= x0 or y1 <= y0:
            continue
        patch = image[y0:y1, x0:x1]
        r, g, b, a = item["color"]
        alpha = a * item["opacity"]
        text = np.array([r, g, b], dtype=float)
        blended = text * alpha + patch * (1 - alpha)
        lt, lb = luminance(blended), luminance(patch)
        hi, lo = np.maximum(lt, lb), np.minimum(lt, lb)
        ratio = float(((hi + 0.05) / (lo + 0.05)).min())
        large = item["size"] >= 24 or (item["size"] >= 18.66 and item["weight"] >= 700)
        need = 3.0 if large else 4.5
        checked += 1
        if ratio < lowest[0]:
            lowest = (ratio, f"{item['where']} {item['text']!r}")
        if ratio < need:
            failures.append(f"{fx(ratio, 2)}:1 (needs {need}) {item['where']} {item['text']!r} {item['size']}px")
    record_property("observed", f"{checked} pieces of text; lowest contrast {fx(lowest[0], 2)}:1 ({lowest[1]})")
    assert not failures, f"{len(failures)} below the limit: " + " | ".join(failures[:6])
