import json
import logging
import math
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from policyjev.api import app as api_app
from policyjev.api import results as api_results
from policyjev.api import scoring
from policyjev.api.app import ALLOWED_ORIGINS, create_app
from policyjev.api.scoring import Scorer, ScoringService
from policyjev.config import LABELS, REPO_ROOT
from policyjev.freeze_manifest import ManifestError

MARKER = "zqxjv-private-marker-text"
IDENTITY = {"method": "isotonic", "x": [0.0, 1.0], "y": [0.0, 1.0]}


class FakeScorer(Scorer):
    """Returns fixed raw probabilities; can be made unavailable or to fail."""

    def __init__(self, name, raw=None, reason=None, fail=False, code=None):
        super().__init__(name)
        self._raw = raw if raw is not None else [0.0] * len(LABELS)
        self.reason = reason
        self.reason_code = code
        self.fail = fail
        self.seen = []

    def raw(self, text):
        if self.fail:
            raise RuntimeError(f"cannot score: {text}")
        self.seen.append(len(text))
        return list(self._raw)


def policy_table(systems=("logreg_word", "distilbert", "jev")):
    rows = []
    for system in systems:
        for label in LABELS:
            disabled = label == "threat"
            for primary, recall, precision in ((True, 0.9, 0.9), (False, 0.8, 0.8)):
                rows.append(
                    {
                        "system": system, "label": label, "recall_target": recall,
                        "precision_target": precision, "is_primary": primary,
                        "low": 0.2 if primary else 0.1,
                        "high": math.inf if disabled else (0.8 if primary else 0.5),
                        "action_enabled": not disabled,
                    }
                )
    return pd.DataFrame(rows)


RAW = [0.9, 0.0, 0.3, 0.0, 0.0, 0.05]


def make_service(**overrides):
    scorers = {
        "logreg_word": FakeScorer("logreg_word", RAW),
        "distilbert": FakeScorer("distilbert", [0.1] * 6),
        "jev": FakeScorer("jev", reason="API key not configured", code="no_api_key"),
    }
    scorers.update(overrides)
    calibrators = {s: {"labels": {label: IDENTITY for label in LABELS}} for s in scorers}
    return ScoringService(scorers, calibrators, policy_table())


def make_client(service=None, table=None, provider=None):
    return TestClient(create_app(service or make_service(), table if table is not None else policy_table(), provider or (lambda: {})))


def test_health_lists_each_system_with_availability():
    body = make_client().get("/api/health").json()

    assert body["status"] == "ok"
    assert body["systems"]["logreg_word"] == {"status": "available", "reason": None, "reason_code": None}
    assert body["systems"]["jev"] == {
        "status": "unavailable", "reason": "API key not configured", "reason_code": "no_api_key",
    }
    assert set(body["systems"]) == {"logreg_word", "distilbert", "jev"}


def test_score_response_schema_decisions_and_triggering_labels():
    body = make_client().post("/api/score", json={"text": "hello there"}).json()

    assert set(body) == {"results", "issues"}
    assert [r["system"] for r in body["results"]] == ["logreg_word", "distilbert"]
    first = body["results"][0]
    assert set(first) == {"system", "probabilities", "decision", "triggering_labels", "time_ms"}
    assert list(first["probabilities"]) == LABELS
    assert first["probabilities"]["toxic"] == pytest.approx(0.9)
    assert first["decision"] == "ACTION"
    assert first["triggering_labels"] == ["toxic"]
    assert first["time_ms"] >= 0
    second = body["results"][1]
    assert second["decision"] == "ALLOW" and second["triggering_labels"] == []
    assert body["issues"] == [
        {"system": "jev", "status": "unavailable", "reason": "API key not configured", "reason_code": "no_api_key"}
    ]


def test_review_decision_and_disabled_action_label():
    service = make_service(logreg_word=FakeScorer("logreg_word", [0.3, 0, 0, 1.0, 0, 0]))
    body = make_client(service).post("/api/score", json={"text": "x", "systems": ["logreg_word"]}).json()

    result = body["results"][0]
    assert result["decision"] == "HUMAN REVIEW"
    assert result["triggering_labels"] == ["toxic", "threat"]


def test_raw_probabilities_only_when_requested():
    client = make_client()
    plain = client.post("/api/score", json={"text": "x", "systems": ["logreg_word"]}).json()
    raw = client.post("/api/score", json={"text": "x", "systems": ["logreg_word"], "include_raw": True}).json()

    assert "raw_probabilities" not in plain["results"][0]
    assert raw["results"][0]["raw_probabilities"] == dict(zip(LABELS, RAW))


def test_requested_unavailable_system_is_reported_not_an_error():
    response = make_client().post("/api/score", json={"text": "x", "systems": ["jev"]})

    assert response.status_code == 200
    assert response.json() == {
        "results": [],
        "issues": [
            {"system": "jev", "status": "unavailable", "reason": "API key not configured", "reason_code": "no_api_key"}
        ],
    }


def test_local_only_request_never_reaches_an_available_jev_scorer():
    jev = FakeScorer("jev", [0.9] * 6)
    service = make_service(jev=jev)
    client = make_client(service)

    local = client.post("/api/score", json={"text": MARKER, "systems": ["logreg_word", "distilbert"]}).json()
    assert [r["system"] for r in local["results"]] == ["logreg_word", "distilbert"]
    assert local["issues"] == []
    assert jev.seen == []

    everything = client.post("/api/score", json={"text": MARKER}).json()
    assert [r["system"] for r in everything["results"]] == ["logreg_word", "distilbert", "jev"]
    assert len(jev.seen) == 1


def test_systems_list_is_validated_against_the_known_ids_without_echoing_the_text():
    client = make_client()
    for systems in (["jev", "gpt"], [], "jev", [1], None):
        response = client.post("/api/score", json={"text": MARKER, "systems": systems})
        if systems is None:
            assert response.status_code == 200
            continue
        assert response.status_code == 422
        assert MARKER not in response.text and "input" not in response.text


def test_missing_calibrator_or_thresholds_make_a_system_unavailable():
    scorers = {"logreg_word": FakeScorer("logreg_word", RAW)}
    no_calibrator = ScoringService(scorers, {}, policy_table())
    assert no_calibrator.status("logreg_word") == ("unavailable", "calibrator not found")
    no_thresholds = ScoringService(scorers, {"logreg_word": {"labels": {}}}, None)
    assert no_thresholds.status("logreg_word") == ("unavailable", "policy thresholds not found")


@pytest.mark.parametrize(
    "payload",
    [
        {"text": ""},
        {"text": "a" * 2001},
        {},
        {"text": "x", "systems": ["unknown"]},
        {"text": "x", "systems": []},
        {"text": "x", "extra": 1},
    ],
)
def test_invalid_requests_get_422_without_echoing_the_input(payload):
    if "text" in payload and len(payload["text"]) > 2000:
        payload = {"text": MARKER * 100}
    response = make_client().post("/api/score", json=payload)

    assert response.status_code == 422
    assert MARKER not in response.text
    assert all(set(item) == {"loc", "msg", "type"} for item in response.json()["detail"])


def test_text_length_boundaries():
    client = make_client()
    assert client.post("/api/score", json={"text": "a"}).status_code == 200
    assert client.post("/api/score", json={"text": "a" * 2000}).status_code == 200
    assert client.post("/api/score", json={"text": "a" * 2001}).status_code == 422


def test_text_never_appears_in_logs_responses_or_files(caplog, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    failing = FakeScorer("distilbert", fail=True)
    client = make_client(make_service(distilbert=failing))

    with caplog.at_level(logging.DEBUG):
        ok = client.post("/api/score", json={"text": MARKER})
        bad = client.post("/api/score", json={"text": MARKER * 200})
        wrong = client.post("/api/score", json={"text": MARKER, "systems": ["nope"]})

    assert ok.status_code == 200 and bad.status_code == 422 and wrong.status_code == 422
    errors = [i for i in ok.json()["issues"] if i["system"] == "distilbert"]
    assert errors == [{"system": "distilbert", "status": "error", "reason": "scoring failed"}]
    assert MARKER not in ok.text and MARKER not in bad.text and MARKER not in wrong.text
    assert MARKER not in caplog.text
    assert "RuntimeError" in caplog.text
    assert list(tmp_path.iterdir()) == []


def test_thresholds_endpoint_encodes_infinity_as_null_and_lists_primary_rows_only():
    body = make_client().get("/api/thresholds").json()

    assert len(body) == 3 * len(LABELS)
    assert all(set(row) == {"system", "label", "low", "high", "action_enabled"} for row in body)
    threat = [r for r in body if r["system"] == "jev" and r["label"] == "threat"][0]
    assert threat == {"system": "jev", "label": "threat", "low": 0.2, "high": None, "action_enabled": False}
    toxic = [r for r in body if r["label"] == "toxic"][0]
    assert toxic["high"] == 0.8 and toxic["action_enabled"] is True
    assert "Infinity" not in json.dumps(body)


def results_frames():
    rows = []
    for label in LABELS + ["mean"]:
        row = {"label": label, "positives": float("nan") if label == "mean" else 100.0}
        for s in ("logreg_word", "distilbert", "jev", "jev_stack"):
            row.update({f"ap_{s}": 0.5, f"ap_{s}_lo": 0.4, f"ap_{s}_hi": 0.6})
        for a, b in (("jev", "logreg_word"), ("jev", "distilbert"), ("jev_stack", "jev")):
            row.update({f"diff_ap_{a}_minus_{b}": -0.1, f"diff_ap_{a}_minus_{b}_lo": float("nan"), f"diff_ap_{a}_minus_{b}_hi": 0.0})
        rows.append(row)
    summary = pd.DataFrame(
        [
            {
                "system": s, "n_rows": 10, "n_failed_rows": 0, "n_allow": 5, "n_review": 3, "n_action": 2,
                "allow_share": 0.5, "m1_review_share": 0.3, "m2_action_precision": float("nan"),
                "m3_miss_rate": 0.1, "m4_rare_miss_rate": 0.1, "action_share": 0.2, "action_coverage": 0.2,
                "any_positive_comments": 4, "rare_positive_comments": 1,
            }
            for s in ("logreg_word", "distilbert", "jev")
        ]
    )
    contrasts = pd.DataFrame(
        [{"contrast": "jev - distilbert", "metric": "m1_review_share", "difference": 0.1, "ci_low": 0.0,
          "ci_high": 0.2, "resamples_used": 1000, "resamples_left_out": 0}]
    )
    budget = pd.DataFrame(
        [{"budget_pct": 1, "metric": "recall_rare", "comparison": "jev", "value": 0.3, "lo": 0.2, "hi": 0.4, "status": "descriptive"}]
    )
    per_label = pd.DataFrame(
        [
            {
                "system": "jev", "label": "toxic", "positives": 50, "action_enabled": True,
                "action_precision_label": 0.8, "miss_rate_label": 0.25,
            },
            {
                "system": "jev", "label": "threat", "positives": 5, "action_enabled": False,
                "action_precision_label": float("nan"), "miss_rate_label": float("nan"),
            },
        ]
    )
    thresholds = pd.DataFrame(
        [
            {
                "system": "jev", "label": label, "is_primary": primary, "recall_target": recall,
                "precision_target": 0.9, "validation_recall_at_low": 0.91,
                "validation_precision_at_high": high,
            }
            for label, primary, recall, high in (
                ("toxic", True, 0.9, 0.93), ("toxic", False, 0.8, 0.5), ("threat", True, 0.9, float("nan")),
            )
        ]
    )
    return {
        "comparison": pd.DataFrame(rows), "policy_summary": summary, "policy_contrasts": contrasts,
        "budget": budget, "policy_per_label": per_label, "policy_thresholds": thresholds,
    }


def test_results_payload_has_only_numbers_and_known_identifiers():
    payload = api_results.build_results_payload(results_frames())
    client = make_client(provider=lambda: api_results.build_results_payload(results_frames()))
    response = client.get("/api/results")

    assert response.status_code == 200
    body = json.loads(response.text)
    assert body == json.loads(json.dumps(payload))
    assert set(body) == {
        "mean_ap", "per_label_ap", "ap_differences", "policy_summary", "policy_contrasts",
        "policy_per_label", "review_budget", "policy_fit", "missing_sections",
    }
    assert body["missing_sections"] == []
    allowed = (
        set(LABELS) | {"mean", "logreg_word", "distilbert", "jev", "jev_stack", "jev - logreg_word",
                       "jev - distilbert", "jev_stack - jev", "m1_review_share", "recall_rare", "descriptive"}
    )

    def walk(value):
        if isinstance(value, dict):
            for key, item in value.items():
                assert key in {
                    "system", "ap", "ap_lo", "ap_hi", "label", "positives", "contrast", "difference", "lo", "hi",
                    "n_rows", "n_failed_rows", "n_allow", "n_review", "n_action", "allow_share",
                    "m1_review_share", "m2_action_precision", "m3_miss_rate", "m4_rare_miss_rate",
                    "action_share", "action_coverage", "metric", "ci_low", "ci_high", "resamples_used",
                    "resamples_left_out", "budget_pct", "comparison", "value", "status", "mean_ap",
                    "per_label_ap", "ap_differences", "policy_summary", "policy_contrasts", "review_budget",
                    "missing_sections", "policy_per_label", "action_enabled", "recall_target",
                    "precision_target", "validation_recall_at_low", "test_recall_at_low",
                    "validation_precision_at_high", "test_action_precision", "policy_fit", "support_floor",
                }
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
        elif isinstance(value, str):
            assert value in allowed

    walk(body)
    assert "NaN" not in response.text


def test_results_payload_lists_missing_sections():
    payload = api_results.build_results_payload({"comparison": results_frames()["comparison"]})

    assert "mean_ap" in payload and "policy_summary" not in payload
    assert payload["missing_sections"] == [
        "policy_summary", "policy_contrasts", "policy_per_label", "review_budget",
    ]
    assert api_results.build_results_payload({})["missing_sections"][:3] == ["mean_ap", "per_label_ap", "ap_differences"]


def test_policy_label_rows_join_validation_and_test_values():
    rows = api_results.build_results_payload(results_frames())["policy_per_label"]
    by_label = {row["label"]: row for row in rows}

    assert [row["label"] for row in rows] == ["toxic", "threat"]
    toxic = by_label["toxic"]
    assert toxic["recall_target"] == 0.9 and toxic["precision_target"] == 0.9
    assert toxic["validation_recall_at_low"] == 0.91
    assert toxic["validation_precision_at_high"] == 0.93
    assert toxic["test_action_precision"] == 0.8
    assert toxic["test_recall_at_low"] == 0.75
    assert toxic["action_enabled"] is True
    # Only the primary threshold row is used: the 0.5 precision belongs to the non-primary row.
    assert by_label["threat"]["validation_precision_at_high"] is None
    assert by_label["threat"]["test_recall_at_low"] is None
    assert by_label["threat"]["test_action_precision"] is None
    assert by_label["threat"]["action_enabled"] is False


def test_policy_per_label_is_listed_as_missing_when_a_source_or_column_is_absent():
    frames = results_frames()
    without_thresholds = {k: v for k, v in frames.items() if k != "policy_thresholds"}
    assert "policy_per_label" in api_results.build_results_payload(without_thresholds)["missing_sections"]

    short = dict(frames, policy_per_label=frames["policy_per_label"].drop(columns=["miss_rate_label"]))
    payload = api_results.build_results_payload(short)
    assert "policy_per_label" in payload["missing_sections"] and "policy_per_label" not in payload


def test_committed_policy_per_label_covers_every_system_and_label():
    payload = api_results.load_results_payload()
    if "policy_per_label" not in payload:
        pytest.skip("committed policy files are not present")
    rows = payload["policy_per_label"]

    assert {(row["system"], row["label"]) for row in rows} == {
        (system, label) for system in ("logreg_word", "distilbert", "jev") for label in LABELS
    }
    for row in rows:
        assert row["recall_target"] == 0.9 and row["precision_target"] == 0.9
        assert 0.0 <= row["test_recall_at_low"] <= 1.0
        # Validation precision at HIGH exists exactly where automatic action is enabled.
        assert (row["validation_precision_at_high"] is not None) == row["action_enabled"]
        if not row["action_enabled"]:
            assert row["test_action_precision"] is None


def test_results_expose_the_support_floor_used_when_fitting_high():
    from policyjev.policy import thresholds as policy_thresholds

    payload = api_results.build_results_payload({})
    assert payload["policy_fit"] == {"support_floor": policy_thresholds.SUPPORT_FLOOR}
    client = make_client(provider=lambda: api_results.build_results_payload(results_frames()))
    assert client.get("/api/results").json()["policy_fit"]["support_floor"] == policy_thresholds.SUPPORT_FLOOR


def test_committed_high_thresholds_meet_the_exposed_support_floor():
    payload = api_results.load_results_payload()
    floor = payload["policy_fit"]["support_floor"]
    table = pd.read_csv(api_results.REPO_ROOT / "artifacts" / "policy" / "policy_thresholds.csv")
    primary = table[table["is_primary"].astype(str).str.lower() == "true"]
    enabled = primary[primary["action_enabled"].astype(str).str.lower() == "true"]
    assert len(enabled) > 0
    assert (enabled["validation_flagged_at_high"] >= floor).all()


def _imports(path):
    import ast

    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
            names.update(f"{node.module}.{a.name}" for a in node.names)
    return names


def test_the_live_jev_path_has_no_cache_and_no_logging_of_the_text():
    """Verification of the privacy notice by reading the code that /api/score runs with JEV configured.

    The batch pipeline caches JEV answers in artifacts/cache/jev.sqlite (policyjev.jev.cache, used by the
    runner). The API path is app.py, scoring.py, schemas.py and the client in policyjev/jev/client.py:
    none of them imports the cache or sqlite3, the client never logs, prints or opens a file to write,
    and the API logs only exception type names.
    """
    api = REPO_ROOT / "src" / "policyjev" / "api"
    client = REPO_ROOT / "src" / "policyjev" / "jev" / "client.py"
    for path in [*api.glob("*.py"), client]:
        found = _imports(path)
        assert "sqlite3" not in found, path
        assert not any(name.startswith("policyjev.jev.cache") or name == "policyjev.jev.cache" for name in found), path
    client_source = client.read_text(encoding="utf-8")
    for forbidden in ("logging", "logger", "print(", ".write(", "write_text", "open(", "sqlite"):
        assert forbidden not in client_source, forbidden
    for path in api.glob("*.py"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if "logger." in line:
                assert "text" not in line.replace("type(error)", "").lower() or "text length" in line, line


def test_a_failing_jev_call_that_echoes_the_text_leaves_no_trace(monkeypatch, caplog, tmp_path):
    from policyjev.jev import client as jev_client

    def echo(self, state, questions, on_attempt=None):
        raise jev_client.JevHTTPError(f"request failed (HTTP 422): bad input {state}", 422)

    monkeypatch.setattr(jev_client.JevClient, "evaluate", echo)
    monkeypatch.chdir(tmp_path)
    scorer = scoring.JevScorer(key_loader=lambda: "key-value")
    client = make_client(make_service(jev=scorer))
    with caplog.at_level(logging.DEBUG):
        response = client.post("/api/score", json={"text": MARKER, "systems": ["jev"]})
    assert response.status_code == 200
    assert response.json()["issues"] == [{"system": "jev", "status": "error", "reason": "scoring failed"}]
    assert MARKER not in response.text and MARKER not in caplog.text
    assert list(tmp_path.iterdir()) == []


def test_a_working_jev_call_writes_no_file_and_logs_no_text(monkeypatch, caplog, tmp_path):
    from types import SimpleNamespace

    from policyjev.jev import client as jev_client

    monkeypatch.setattr(
        jev_client.JevClient, "evaluate",
        lambda self, state, questions, on_attempt=None: SimpleNamespace(answers={label: 0.5 for label in LABELS}),
    )
    monkeypatch.chdir(tmp_path)
    cache = REPO_ROOT / "artifacts" / "cache"
    before = {p: p.stat().st_mtime_ns for p in cache.rglob("*") if p.is_file()} if cache.exists() else {}
    client = make_client(make_service(jev=scoring.JevScorer(key_loader=lambda: "key-value")))
    with caplog.at_level(logging.DEBUG):
        body = client.post("/api/score", json={"text": MARKER, "systems": ["jev"]}).json()
    assert [r["system"] for r in body["results"]] == ["jev"]
    assert MARKER not in caplog.text and MARKER not in json.dumps(body)
    after = {p: p.stat().st_mtime_ns for p in cache.rglob("*") if p.is_file()} if cache.exists() else {}
    assert after == before, "the live path must not write to artifacts/cache"
    assert list(tmp_path.iterdir()) == []


def test_cors_allows_only_the_two_local_console_origins():
    client = make_client()
    assert ALLOWED_ORIGINS == ["http://localhost:5173", "http://127.0.0.1:5173"]
    for origin in ALLOWED_ORIGINS:
        response = client.options(
            "/api/score",
            headers={"Origin": origin, "Access-Control-Request-Method": "POST", "Access-Control-Request-Headers": "content-type"},
        )
        assert response.headers["access-control-allow-origin"] == origin
    other = client.options(
        "/api/score",
        headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "POST"},
    )
    assert "access-control-allow-origin" not in other.headers
    simple = client.get("/api/health", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in simple.headers


def test_server_binds_to_loopback():
    from policyjev.api import __main__ as entry

    assert entry.HOST == "127.0.0.1" and entry.PORT == 8000


def test_scorers_without_model_files_are_unavailable_not_errors(tmp_path):
    assert scoring.LogregScorer(tmp_path / "missing.joblib").reason == "model file not found"
    assert scoring.DistilbertScorer(tmp_path / "missing").reason == "model files not found"


def test_every_unavailable_cause_has_a_stable_reason_code(tmp_path, monkeypatch):
    # model files missing
    assert scoring.LogregScorer(tmp_path / "missing.joblib").reason_code == "model_missing"
    assert scoring.DistilbertScorer(tmp_path / "missing").reason_code == "model_missing"

    # no API key
    def no_key():
        raise RuntimeError("no key")

    jev = scoring.JevScorer(key_loader=no_key)
    assert jev.available() is False and jev.reason_code == "no_api_key"

    # calibrator, thresholds and unknown system, as the service sees them
    scorers = {"logreg_word": FakeScorer("logreg_word", RAW)}
    no_calibrator = ScoringService(scorers, {}, policy_table())
    assert no_calibrator.reason_code("logreg_word") == "calibrator_missing"
    assert no_calibrator.reason_code("nothing") == "unknown_system"
    no_thresholds = ScoringService(scorers, {"logreg_word": {"labels": {}}}, None)
    assert no_thresholds.reason_code("logreg_word") == "thresholds_missing"

    # a scorer disabled without a code still reports a code, never null
    plain = FakeScorer("logreg_word", RAW, reason="something else")
    service = make_service(logreg_word=plain)
    assert service.reason_code("logreg_word") == "unavailable"

    # manifest failure and load failure
    scorer = FakeScorer("logreg_word", RAW)
    scorer.disable("frozen manifest check failed", scoring.MANIFEST_FAILED)
    assert make_service(logreg_word=scorer).reason_code("logreg_word") == "manifest_failed"

    # available and loading systems have no code
    assert make_service().reason_code("logreg_word") is None


def test_the_reason_code_is_in_health_and_in_the_issues_and_does_not_depend_on_the_text():
    jev = FakeScorer("jev", reason="some new wording that never mentions a key", code="no_api_key")
    client = make_client(make_service(jev=jev))
    health = client.get("/api/health").json()["systems"]
    assert health["jev"]["reason_code"] == "no_api_key"
    assert health["jev"]["reason"] == "some new wording that never mentions a key"
    issue = client.post("/api/score", json={"text": "x", "systems": ["jev"]}).json()["issues"][0]
    assert issue["reason_code"] == "no_api_key"

    other = FakeScorer("jev", reason="API key not configured", code="model_missing")
    assert make_client(make_service(jev=other)).get("/api/health").json()["systems"]["jev"]["reason_code"] == "model_missing"


def test_health_reason_code_is_null_for_available_and_loading_systems():
    health = make_client().get("/api/health").json()["systems"]
    assert health["logreg_word"]["reason_code"] is None
    assert health["distilbert"]["reason_code"] is None


def test_jev_scorer_availability_depends_on_the_key_and_the_key_is_not_exposed(caplog):
    calls = []

    def missing_key():
        calls.append(1)
        raise RuntimeError("no key")

    scorer = scoring.JevScorer(key_loader=missing_key)
    assert not scorer.available() and not scorer.available()
    assert scorer.reason == "API key not configured"
    assert len(calls) == 1

    secret = "sk-secret-value"
    with caplog.at_level(logging.DEBUG):
        present = scoring.JevScorer(key_loader=lambda: secret)
        assert present.available()
    assert secret not in caplog.text and secret not in repr(present.__dict__.get("reason"))


def test_jev_scorer_calls_the_client_without_any_cache(monkeypatch):
    from policyjev.jev import cache as cache_module
    from policyjev.jev import client as client_module

    seen = {}

    class FakeClient:
        def evaluate(self, text, questions):
            seen["questions"] = sorted(questions)
            seen["has_criteria"] = all("criteria" in q for q in questions.values())
            return SimpleNamespace(answers={label: 0.25 for label in LABELS})

    monkeypatch.setattr(client_module.JevClient, "from_config", classmethod(lambda cls, cfg, key, **kwargs: FakeClient()))
    monkeypatch.setattr(
        cache_module, "JevCache", lambda *a, **k: (_ for _ in ()).throw(AssertionError("cache used"))
    )
    scorer = scoring.JevScorer(key_loader=lambda: "key")

    assert scorer.available()
    assert scorer.raw("some text") == [0.25] * len(LABELS)
    assert seen["questions"] == sorted(LABELS) and seen["has_criteria"]


def test_manifest_mismatch_marks_only_the_affected_system_and_logs_the_file(monkeypatch, caplog):
    scorers = {n: FakeScorer(n) for n in ("logreg_word", "distilbert", "jev")}

    def verify(paths):
        for path in paths:
            if path.name == "distilbert.json":
                raise ManifestError("distilbert.json: hash does not match the frozen manifest")

    monkeypatch.setattr(scoring, "verify_files", verify)
    monkeypatch.setattr(scoring, "verify_model_folder", lambda prefix, folder: None)

    with caplog.at_level(logging.WARNING):
        failures = scoring.verify_frozen_inputs(scorers)

    assert failures == {"distilbert": "frozen manifest check failed"}
    assert "distilbert.json" in caplog.text


def test_thresholds_manifest_mismatch_marks_every_system(monkeypatch):
    scorers = {n: FakeScorer(n) for n in ("logreg_word", "distilbert", "jev")}

    def verify(paths):
        for path in paths:
            if path.name == "policy_thresholds.csv":
                raise ManifestError("policy_thresholds.csv: hash does not match the frozen manifest")

    monkeypatch.setattr(scoring, "verify_files", verify)
    monkeypatch.setattr(scoring, "verify_model_folder", lambda prefix, folder: None)

    assert set(scoring.verify_frozen_inputs(scorers)) == {"logreg_word", "distilbert", "jev"}


def test_scoring_does_not_keep_the_text():
    service = make_service()
    scorer = service.scorers["logreg_word"]
    service.score(MARKER)

    assert scorer.seen == [len(MARKER)]
    assert MARKER not in repr(vars(scorer)) and MARKER not in repr(vars(service))


def test_app_module_uses_no_file_writing_for_requests():
    import inspect

    source = inspect.getsource(api_app) + inspect.getsource(scoring.ScoringService)
    for forbidden in ("open(", "write_text", "to_csv", "pickle", "sqlite"):
        assert forbidden not in source
    assert isinstance(np.float64(1.0), float)


import threading


class SlowScorer(FakeScorer):
    """A local scorer whose warm-up blocks until released."""

    local = True

    def __init__(self, name, release, started=None, fail_warm=False):
        super().__init__(name, RAW)
        self.release = release
        self.started = started
        self.fail_warm = fail_warm
        self.warmed = False

    def warm(self):
        if self.started is not None:
            self.started.set()
        assert self.release.wait(timeout=5)
        if self.fail_warm:
            raise RuntimeError(f"cannot load: {MARKER}")
        self.warmed = True


def test_warmup_reports_loading_then_available_and_never_blocks_scoring(caplog):
    release, started = threading.Event(), threading.Event()
    slow = SlowScorer("logreg_word", release, started)
    service = make_service(logreg_word=slow)

    with TestClient(create_app(service, policy_table(), lambda: {})) as client:
        assert started.wait(timeout=5)
        health = client.get("/api/health").json()["systems"]
        assert health["logreg_word"] == {"status": "loading", "reason": None, "reason_code": None}
        assert health["distilbert"]["status"] == "available"

        response = client.post("/api/score", json={"text": "hello", "systems": ["logreg_word", "distilbert"]})
        body = response.json()
        assert response.status_code == 200
        assert body["issues"] == [{"system": "logreg_word", "status": "loading", "reason": "model is loading"}]
        assert [r["system"] for r in body["results"]] == ["distilbert"]
        assert slow.seen == []

        release.set()
        service._warm_thread.join(timeout=5)
        assert client.get("/api/health").json()["systems"]["logreg_word"]["status"] == "available"
        after = client.post("/api/score", json={"text": "hello", "systems": ["logreg_word"]}).json()
        assert after["results"][0]["system"] == "logreg_word" and slow.warmed


def test_only_local_available_scorers_are_warmed_and_jev_is_never_loading():
    release = threading.Event()
    local = SlowScorer("logreg_word", release)
    unavailable_local = SlowScorer("distilbert", release)
    unavailable_local.reason = "model files not found"
    service = make_service(logreg_word=local, distilbert=unavailable_local)

    thread = service.start_warmup()
    assert service.status("logreg_word") == ("loading", None)
    assert service.status("distilbert") == ("unavailable", "model files not found")
    assert service.status("jev")[0] == "unavailable"
    release.set()
    thread.join(timeout=5)
    assert local.warmed and not unavailable_local.warmed
    assert service.status("logreg_word") == ("available", None)


def test_failed_warmup_marks_the_system_unavailable_without_logging_details(caplog):
    release = threading.Event()
    release.set()
    failing = SlowScorer("logreg_word", release, fail_warm=True)
    service = make_service(logreg_word=failing)

    with caplog.at_level(logging.DEBUG):
        service.start_warmup().join(timeout=5)

    assert service.status("logreg_word") == ("unavailable", "model failed to load")
    assert "RuntimeError" in caplog.text and MARKER not in caplog.text


def test_start_warmup_with_nothing_to_load_returns_none():
    assert make_service().start_warmup() is None


def test_warmup_can_be_disabled():
    release = threading.Event()
    slow = SlowScorer("logreg_word", release)
    with TestClient(create_app(make_service(logreg_word=slow), policy_table(), lambda: {}, warm_up=False)) as client:
        assert client.get("/api/health").json()["systems"]["logreg_word"]["status"] == "available"
    assert not slow.warmed
