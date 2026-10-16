"""Local API for the end-to-end checks (console/e2e). Not a test module; the harness starts it.

It never uses port 8000, never contacts TypeSafe or any other host, and never reads a key file.

    python tests/e2e_api.py --port 8123 [--mode real|light] [--jev off|stub]
                            [--warm-delay SECONDS] [--jev-delay SECONDS]

--mode real   the real calibrated logistic regression and DistilBERT models, the frozen thresholds
              and calibrators, and the same manifest checks as the real server.
--mode light  fixed local scorers and no model loading, so the server starts in a second or two
              (used to kill the server and bring it back).
--jev off     JEV is unavailable because no key can be found (the key lookup is replaced, so no key
              file is opened); the app reports it as needing a key.
--jev stub    JEV is available and a stub scorer answers with fixed valid probabilities. Each call
              writes one line to stderr with only the length of the text.
--warm-delay  wait this long before each local model starts loading, so a test can watch the loading
              state for a known time.

The server is started the way python -m policyjev.api starts it: the same app factory, loopback only,
default logging.
"""

import argparse
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import pandas as pd  # noqa: E402
import uvicorn  # noqa: E402

from policyjev.api import scoring  # noqa: E402
from policyjev.api.app import create_app  # noqa: E402
from policyjev.api.scoring import JevScorer, Scorer, ScoringService  # noqa: E402
from policyjev.config import LABELS  # noqa: E402
from policyjev.evaluation.calibration import calibrator_path, load_calibrators  # noqa: E402

HOST = "127.0.0.1"
FORBIDDEN_PORT = 8000
STUB_RAW = [0.90, 0.05, 0.40, 0.02, 0.35, 0.03]
LIGHT_RAW = [0.05, 0.01, 0.04, 0.01, 0.03, 0.01]


class StubJevScorer(Scorer):
    """Fixed raw probabilities in place of the TypeSafe call; records only that it was called."""

    def __init__(self, delay: float = 0.0) -> None:
        super().__init__("jev")
        self.delay = delay

    def raw(self, text: str) -> list[float]:
        print(f"[e2e] jev stub called (text length {len(text)})", file=sys.stderr, flush=True)
        if self.delay:
            time.sleep(self.delay)
        return list(STUB_RAW)


class FixedLocalScorer(Scorer):
    """A local scorer that returns fixed raw probabilities at once (light mode)."""

    local = True

    def __init__(self, name: str) -> None:
        super().__init__(name)

    def raw(self, text: str) -> list[float]:
        return list(LIGHT_RAW)


def no_key() -> str:
    raise RuntimeError("no key in end-to-end runs")


def build_service(mode: str) -> ScoringService:
    if mode == "real":
        return scoring.build_default_service()
    calibrators = {
        system: load_calibrators(calibrator_path(system, scoring.JEV_VERSION if system == "jev" else None))
        for system in scoring.SYSTEMS
    }
    table = pd.read_csv(scoring.POLICY_THRESHOLDS)
    scorers = {
        "logreg_word": FixedLocalScorer("logreg_word"),
        "distilbert": FixedLocalScorer("distilbert"),
        "jev": JevScorer(key_loader=no_key),
    }
    return ScoringService(scorers, calibrators, table)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--mode", choices=["real", "light"], default="real")
    parser.add_argument("--jev", choices=["off", "stub"], default="off")
    parser.add_argument("--warm-delay", type=float, default=0.0)
    parser.add_argument("--jev-delay", type=float, default=0.0)
    args = parser.parse_args()
    if args.port == FORBIDDEN_PORT:
        sys.exit("port 8000 is for the real server; pick another port")

    service = build_service(args.mode)
    service.scorers["jev"] = StubJevScorer(args.jev_delay) if args.jev == "stub" else JevScorer(key_loader=no_key)

    if args.warm_delay > 0:
        for scorer in service.scorers.values():
            if getattr(scorer, "local", False):
                original = scorer.warm

                def slow(original=original) -> None:
                    time.sleep(args.warm_delay)
                    original()

                scorer.warm = slow

    assert len(LABELS) == len(STUB_RAW)
    uvicorn.run(create_app(service), host=HOST, port=args.port)


if __name__ == "__main__":
    main()
