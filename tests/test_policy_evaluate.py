import math
import sys

import numpy as np
import pandas as pd
import pytest

from policyjev.config import LABELS
from policyjev.evaluation.splits import SplitScores
from policyjev.policy import evaluate as ev
from policyjev.policy import thresholds as pt


def thresholds_frame(low=0.2, high=0.8, threat_high=math.inf):
    highs = [high] * len(LABELS)
    highs[LABELS.index("threat")] = threat_high
    return pd.DataFrame({"label": LABELS, "low": [low] * len(LABELS), "high": highs})


def hand_arrays():
    codes = np.array([2, 2, 2, 1, 1, 1, 0, 0, 0, 0])
    hit = np.array([True, True, False] + [False] * 7)
    any_positive = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0, 0], dtype=bool)
    rare_positive = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 0], dtype=bool)
    return codes, hit, any_positive, rare_positive


def test_m1_to_m4_on_a_hand_computed_example():
    metrics = ev.policy_metrics(*hand_arrays())

    assert metrics["m1_review_share"] == pytest.approx(3 / 10)
    assert metrics["m2_action_precision"] == pytest.approx(2 / 3)
    assert metrics["m3_miss_rate"] == pytest.approx(2 / 5)
    assert metrics["m4_rare_miss_rate"] == pytest.approx(1 / 3)
    assert metrics["action_share"] == pytest.approx(3 / 10)
    assert metrics["action_coverage"] == pytest.approx(2 / 5)


def test_m2_is_undefined_when_no_comment_receives_action():
    codes, _, any_positive, rare_positive = hand_arrays()
    codes = np.where(codes == 2, 1, codes)

    metrics = ev.policy_metrics(codes, np.zeros(10, dtype=bool), any_positive, rare_positive)

    assert math.isnan(metrics["m2_action_precision"])
    assert metrics["action_share"] == 0.0
    assert metrics["action_coverage"] == 0.0
    assert metrics["m1_review_share"] == pytest.approx(6 / 10)


def test_decision_data_uses_the_engine_and_the_labels_that_reached_high():
    calibrated = np.zeros((4, len(LABELS)))
    y = np.zeros((4, len(LABELS)), dtype=int)
    toxic, severe, obscene, threat = (LABELS.index(n) for n in ("toxic", "severe_toxic", "obscene", "threat"))
    calibrated[0, toxic] = 0.9
    y[0, toxic] = 1
    calibrated[1, toxic] = 0.9
    calibrated[1, severe] = 0.5
    y[1, severe] = 1
    calibrated[2, obscene] = 0.5
    calibrated[3, threat] = 1.0
    y[3, threat] = 1

    d = ev.decision_data(calibrated, y, thresholds_frame())

    assert d.codes.tolist() == [2, 2, 1, 1]
    assert d.action_hit.tolist() == [True, False, False, False]


def test_per_label_table_hand_example():
    p = np.zeros((4, len(LABELS)))
    p[:, 0] = [0.9, 0.9, 0.5, 0.1]
    y = np.zeros((4, len(LABELS)), dtype=int)
    y[:, 0] = [1, 0, 1, 1]
    table = ev.per_label_table({"jev": p}, y, {"jev": thresholds_frame()})
    row = table[(table["system"] == "jev") & (table["label"] == "toxic")].iloc[0]

    assert row["positives"] == 3
    assert row["action_flagged"] == 2
    assert row["action_precision_label"] == pytest.approx(0.5)
    assert row["review_share_label"] == pytest.approx(0.25)
    assert row["action_share_label"] == pytest.approx(0.5)
    assert row["flagged_share_label"] == pytest.approx(0.75)
    assert row["missed_positives"] == 1
    assert row["miss_rate_label"] == pytest.approx(1 / 3)
    assert row["action_enabled"]

    threat = table[table["label"] == "threat"].iloc[0]
    assert not threat["action_enabled"]
    assert math.isnan(threat["action_precision_label"])
    assert threat["action_flagged"] == 0


def decision_pair(n, jev_actions, baseline_actions, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random((n, len(LABELS))) < 0.2).astype(int)
    any_positive = y.sum(axis=1) > 0

    def make(n_action):
        codes = np.where(rng.random(n) < 0.2, 1, 0)
        action_rows = rng.choice(n, size=n_action, replace=False)
        codes[action_rows] = 2
        hit = np.zeros(n, dtype=bool)
        hit[action_rows] = any_positive[action_rows]
        return ev.DecisionData(codes, hit)

    return y, {"logreg_word": make(baseline_actions), "distilbert": make(baseline_actions), "jev": make(jev_actions)}


def test_bootstrap_left_out_counting_for_undefined_quantities():
    y, data = decision_pair(200, jev_actions=1, baseline_actions=40)

    table = ev.contrast_table(data, y, n_resamples=300, seed=7, confidence=0.95)
    by = table.set_index(["contrast", "metric"])

    assert list(table.columns) == [
        "contrast", "metric", "difference", "ci_low", "ci_high", "resamples_used", "resamples_left_out",
    ]
    assert len(table) == 2 * 4
    assert (table["resamples_used"] + table["resamples_left_out"] == 300).all()
    m1 = by.loc[("jev - logreg_word", "m1_review_share")]
    assert m1["resamples_left_out"] == 0
    m2 = by.loc[("jev - logreg_word", "m2_action_precision")]
    assert 0 < m2["resamples_left_out"] < 300
    assert not math.isnan(m2["ci_low"]) and m2["ci_low"] <= m2["ci_high"]


def test_a_system_without_action_leaves_every_m2_resample_out():
    y, data = decision_pair(150, jev_actions=30, baseline_actions=0)

    table = ev.contrast_table(data, y, n_resamples=100, seed=7, confidence=0.95)
    m2 = table[table["metric"] == "m2_action_precision"]

    assert (m2["resamples_used"] == 0).all()
    assert (m2["resamples_left_out"] == 100).all()
    assert m2["difference"].isna().all() and m2["ci_low"].isna().all() and m2["ci_high"].isna().all()
    m3 = table[table["metric"] == "m3_miss_rate"]
    assert (m3["resamples_left_out"] == 0).all()


def test_contrasts_are_reproducible_for_a_seed():
    y, data = decision_pair(200, 20, 20)
    first = ev.contrast_table(data, y, 50, 3, 0.95)
    again = ev.contrast_table(data, y, 50, 3, 0.95)
    other = ev.contrast_table(data, y, 50, 4, 0.95)

    assert first.equals(again)
    assert not first["ci_low"].equals(other["ci_low"])


def synthetic_world(n=800):
    rng = np.random.default_rng(2)
    y = (rng.random((n, len(LABELS))) < 0.2).astype(int)
    calibrated = {
        s: np.round(np.clip(y * 0.5 + rng.random(y.shape) * 0.5, 0, 1), 3) for s in pt.SYSTEMS
    }
    return y, calibrated, pt.build_policy_table(calibrated, y)


def test_evaluate_policy_returns_the_four_tables_with_the_primary_grid_point_marked():
    y, calibrated, policy_table = synthetic_world()

    results = ev.evaluate_policy(calibrated, y, policy_table, 40, 1, 0.95)

    assert set(results) == {"summary", "per_label", "contrasts", "grid"}
    summary = results["summary"].set_index("system")
    assert set(summary.index) == set(pt.SYSTEMS)
    assert (summary["n_rows"] == len(y)).all() and (summary["n_failed_rows"] == 0).all()
    assert (summary["n_allow"] + summary["n_review"] + summary["n_action"] == len(y)).all()
    assert len(results["per_label"]) == 3 * len(LABELS)
    grid = results["grid"]
    assert len(grid) == 3 * 9
    assert grid.groupby("system")["is_primary"].sum().tolist() == [1, 1, 1]
    primary = grid[grid["is_primary"]].set_index("system")
    for system in pt.SYSTEMS:
        assert primary.loc[system, "m1_review_share"] == pytest.approx(summary.loc[system, "m1_review_share"])
        assert primary.loc[system, "action_share"] == pytest.approx(summary.loc[system, "action_share"])


def test_stricter_precision_target_never_increases_action_share():
    y, calibrated, policy_table = synthetic_world()
    grid = ev.grid_table(calibrated, y, policy_table)

    for system in pt.SYSTEMS:
        rows = grid[(grid["system"] == system) & (grid["recall_target"] == 0.90)].sort_values("precision_target")
        assert rows["n_action"].is_monotonic_decreasing or rows["n_action"].diff().dropna().le(0).all()


def test_gates_refuse_without_the_tags(monkeypatch):
    monkeypatch.setattr(ev, "verify_files", lambda paths: (_ for _ in ()).throw(AssertionError("verified")))

    monkeypatch.setattr(ev, "tag_exists", lambda tag: False)
    with pytest.raises(SystemExit) as info:
        ev.check_gates([])
    assert "protocol-frozen" in str(info.value)

    monkeypatch.setattr(ev, "tag_exists", lambda tag: tag == "protocol-frozen")
    with pytest.raises(SystemExit) as info:
        ev.check_gates([])
    assert "policy-frozen" in str(info.value)

    monkeypatch.setattr(ev, "tag_exists", lambda tag: tag == "policy-frozen")
    with pytest.raises(SystemExit) as info:
        ev.check_gates([])
    assert "protocol-frozen" in str(info.value)


def test_gates_refuse_a_manifest_mismatch_and_existing_outputs(monkeypatch, tmp_path):
    from policyjev.freeze_manifest import ManifestError

    monkeypatch.setattr(ev, "tag_exists", lambda tag: True)

    def mismatch(paths):
        raise ManifestError("hash does not match")

    monkeypatch.setattr(ev, "verify_files", mismatch)
    with pytest.raises(ManifestError):
        ev.check_gates([])

    monkeypatch.setattr(ev, "verify_files", lambda paths: None)
    existing = tmp_path / "policy_test_summary.csv"
    existing.write_text("x", encoding="utf-8")
    with pytest.raises(SystemExit) as info:
        ev.check_gates([tmp_path / "other.csv", existing])
    assert "refusing to overwrite" in str(info.value)
    ev.check_gates([tmp_path / "other.csv"])


def test_main_refuses_before_reading_any_test_data(monkeypatch, tmp_path):
    from policyjev.evaluation import splits

    monkeypatch.setattr(sys, "argv", ["evaluate"])
    monkeypatch.setattr(ev, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(ev, "tag_exists", lambda tag: tag == "protocol-frozen")
    monkeypatch.setattr(
        splits, "load_split_scores_bundle", lambda *a, **k: (_ for _ in ()).throw(AssertionError("read test"))
    )
    with pytest.raises(SystemExit) as info:
        ev.main()
    assert "policy-frozen" in str(info.value)


def test_main_writes_the_four_files_once_and_refuses_to_overwrite(monkeypatch, tmp_path, capsys):
    from policyjev.evaluation import splits

    y, calibrated, policy_table = synthetic_world(600)
    policy_path = tmp_path / "policy_thresholds.csv"
    policy_table.to_csv(policy_path, index=False)
    bundle = SplitScores("test", [f"t{i}" for i in range(len(y))], y, calibrated, calibrated, {}, {})
    calls = []

    monkeypatch.setattr(sys, "argv", ["evaluate"])
    monkeypatch.setattr(ev, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(ev, "POLICY_THRESHOLDS", policy_path)
    monkeypatch.setattr(ev, "tag_exists", lambda tag: True)
    monkeypatch.setattr(ev, "verify_files", lambda paths: calls.append(list(paths)))
    monkeypatch.setattr(splits, "load_split_scores_bundle", lambda split, systems: bundle)

    ev.main()

    metrics = tmp_path / "artifacts" / "metrics"
    for name in ev.OUTPUT_NAMES.values():
        assert (metrics / name).exists()
    assert calls == [[policy_path]]
    contrasts = pd.read_csv(metrics / "policy_test_contrasts.csv")
    assert {"contrast", "metric", "difference", "ci_low", "ci_high", "resamples_used", "resamples_left_out"} == set(contrasts.columns)
    out = capsys.readouterr().out
    assert "test rows used" in out and "600" in out and "failed rows: 0" in out
    before = (metrics / "policy_test_summary.csv").read_bytes()

    with pytest.raises(SystemExit) as info:
        ev.main()
    assert "refusing to overwrite" in str(info.value)
    assert (metrics / "policy_test_summary.csv").read_bytes() == before
