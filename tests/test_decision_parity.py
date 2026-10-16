"""The console's decision rule must agree with the policy engine.

The console decides in TypeScript (console/src/decision.ts) so that the Score and Policy pages
can move a probability without calling the API. This test keeps that rule tied to
src/policyjev/policy/engine.py: it builds probe probabilities around every threshold in the
real /api/thresholds response, asks the engine for the decision, and compares the result with
console/src/fixtures/decision_parity.json. The front-end test replays the same file through the
TypeScript function, so the TypeScript rule, the file and the engine all have to agree.

Regenerate the file after the thresholds change: python tests/test_decision_parity.py
"""

import json
import math
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from policyjev.api.app import create_app
from policyjev.api.scoring import POLICY_THRESHOLDS, ScoringService
from policyjev.config import LABELS, REPO_ROOT
from policyjev.policy.engine import decide, label_severity
from policyjev.policy.thresholds import PRIMARY_PRECISION, PRIMARY_RECALL, thresholds_for

FIXTURE = REPO_ROOT / "console" / "src" / "fixtures" / "decision_parity.json"
DECISIONS = {0: "ALLOW", 1: "HUMAN REVIEW", 2: "ACTION"}


def real_thresholds_response() -> list[dict]:
    """The body of GET /api/thresholds for the committed policy thresholds."""
    app = create_app(ScoringService({}, {}, None), warm_up=False)
    return TestClient(app).get("/api/thresholds").json()


def probes(low: float, high: float | None) -> list[float]:
    """Probabilities just below and at each threshold, plus the ends of the scale."""
    values = [0.0, math.nextafter(low, 0.0), low, math.nextafter(low, 1.0)]
    if high is not None:
        values += [math.nextafter(high, 0.0), high, math.nextafter(high, 1.0)]
    values.append(1.0)
    return sorted(set(values))


def build_cases(thresholds: list[dict], table: pd.DataFrame) -> dict:
    """Expected decisions from the engine for per-label probes and for whole comments."""
    label_cases = []
    comment_cases = []
    for system in sorted({row["system"] for row in thresholds}):
        rows = {row["label"]: row for row in thresholds if row["system"] == system}
        for label in LABELS:
            row = rows[label]
            high = row["high"] if row["action_enabled"] else None
            engine_high = math.inf if high is None else high
            for p in probes(row["low"], high):
                severity = label_severity(p, row["low"], engine_high)
                label_cases.append(
                    {"system": system, "label": label, "p": p, "decision": DECISIONS[severity]}
                )

        engine_table = thresholds_for(table, system, PRIMARY_RECALL, PRIMARY_PRECISION)
        highs = {r["label"]: (r["high"] if r["action_enabled"] else None) for r in rows.values()}

        def comment(probabilities: dict[str, float]) -> dict:
            outcome = decide(probabilities, engine_table)
            return {
                "system": system,
                "probabilities": probabilities,
                "decision": outcome.decision,
                "triggering_labels": list(outcome.triggering_labels),
            }

        below = {label: math.nextafter(rows[label]["low"], 0.0) for label in LABELS}
        at_low = {label: rows[label]["low"] for label in LABELS}
        comment_cases.append(comment(below))
        comment_cases.append(comment(at_low))
        comment_cases.append(comment({label: 1.0 for label in LABELS}))
        for label in LABELS:
            only_low = dict(below)
            only_low[label] = rows[label]["low"]
            comment_cases.append(comment(only_low))
            if highs[label] is not None:
                just_under = dict(at_low)
                just_under[label] = math.nextafter(highs[label], 0.0)
                comment_cases.append(comment(just_under))
                at_high = dict(below)
                at_high[label] = highs[label]
                comment_cases.append(comment(at_high))
    return {"thresholds": thresholds, "label_cases": label_cases, "comment_cases": comment_cases}


def build_fixture() -> dict:
    table = pd.read_csv(POLICY_THRESHOLDS)
    return build_cases(real_thresholds_response(), table)


def test_real_thresholds_response_covers_every_system_and_label():
    thresholds = real_thresholds_response()

    assert {(row["system"], row["label"]) for row in thresholds} == {
        (system, label) for system in ("logreg_word", "distilbert", "jev") for label in LABELS
    }


def test_engine_decides_by_the_documented_rule_at_every_threshold():
    fixture = build_fixture()
    by_key = {(row["system"], row["label"]): row for row in fixture["thresholds"]}
    for case in fixture["label_cases"]:
        row = by_key[(case["system"], case["label"])]
        high = row["high"] if row["action_enabled"] else None
        if high is not None and case["p"] >= high:
            expected = "ACTION"
        elif case["p"] >= row["low"]:
            expected = "HUMAN REVIEW"
        else:
            expected = "ALLOW"
        assert case["decision"] == expected, case


def test_probes_include_just_below_and_exactly_at_each_threshold():
    fixture = build_fixture()
    by_key = {}
    for case in fixture["label_cases"]:
        by_key.setdefault((case["system"], case["label"]), []).append(case)
    for row in fixture["thresholds"]:
        cases = by_key[(row["system"], row["label"])]
        assert any(c["p"] == row["low"] and c["decision"] == "HUMAN REVIEW" for c in cases)
        assert any(c["p"] < row["low"] and c["decision"] == "ALLOW" for c in cases)
        if row["action_enabled"]:
            assert any(c["p"] == row["high"] and c["decision"] == "ACTION" for c in cases)
            assert any(c["p"] < row["high"] and c["decision"] == "HUMAN REVIEW" for c in cases)
        else:
            assert not any(c["decision"] == "ACTION" for c in cases)


def test_committed_fixture_matches_the_engine_and_the_real_thresholds():
    if not FIXTURE.is_file():
        pytest.fail("console/src/fixtures/decision_parity.json is missing; run tests/test_decision_parity.py")
    committed = json.loads(FIXTURE.read_text(encoding="utf-8"))

    assert committed == json.loads(json.dumps(build_fixture())), (
        "the thresholds or the engine changed; regenerate with: python tests/test_decision_parity.py"
    )


def write_fixture(path: Path = FIXTURE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(build_fixture(), f, indent=1)
        f.write("\n")


if __name__ == "__main__":
    write_fixture()
    print(f"wrote {FIXTURE}")
