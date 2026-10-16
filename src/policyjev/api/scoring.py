"""Scoring service for the console: scorers, frozen calibrators and the policy engine.

Typed text is only held in memory for the duration of a call. It is never written to disk,
logged, cached or placed in an error message.
"""

import logging
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from policyjev.config import LABELS, REPO_ROOT, load_config
from policyjev.evaluation.calibration import apply_calibrator, calibrator_path, load_calibrators
from policyjev.freeze_manifest import ManifestError, verify_files, verify_model_folder
from policyjev.policy.engine import decide
from policyjev.policy.thresholds import PRIMARY_PRECISION, PRIMARY_RECALL, thresholds_for

logger = logging.getLogger(__name__)

SYSTEMS = ["logreg_word", "distilbert", "jev"]
MODELS_DIR = REPO_ROOT / "artifacts" / "models"
LOGREG_MODEL = MODELS_DIR / "logreg_word.joblib"
DISTILBERT_DIR = MODELS_DIR / "distilbert"
POLICY_THRESHOLDS = REPO_ROOT / "artifacts" / "policy" / "policy_thresholds.csv"
QUESTIONS_V2 = REPO_ROOT / "configs" / "questions_v2.yaml"
JEV_VERSION = "v2"
MAX_LENGTH = 256

# Stable machine-readable reasons a system is unavailable (the human text may change; these do not).
NO_API_KEY = "no_api_key"
MODEL_MISSING = "model_missing"
MANIFEST_FAILED = "manifest_failed"
LOAD_FAILED = "load_failed"
CALIBRATOR_MISSING = "calibrator_missing"
THRESHOLDS_MISSING = "thresholds_missing"
UNKNOWN_SYSTEM = "unknown_system"
UNAVAILABLE = "unavailable"


class Scorer:
    """A system that turns a comment into six raw probabilities in LABELS order."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.reason: str | None = None
        self.reason_code: str | None = None

    def available(self) -> bool:
        """True if the system can score."""
        return self.reason is None

    def disable(self, reason: str, code: str = UNAVAILABLE) -> None:
        """Mark the system unavailable, with the text for people and the stable code for programs."""
        self.reason = reason
        self.reason_code = code

    def warm(self) -> None:
        """Load anything expensive ahead of the first request (no-op by default)."""

    def raw(self, text: str) -> list[float]:
        """Raw probabilities for one comment."""
        raise NotImplementedError


class LogregScorer(Scorer):
    """TF-IDF + logistic regression, loaded in the background at startup or on first use."""

    local = True

    def __init__(self, model_path: Path = LOGREG_MODEL) -> None:
        super().__init__("logreg_word")
        self.model_path = model_path
        self._saved = None
        self._lock = threading.Lock()
        if not model_path.is_file():
            self.disable("model file not found", MODEL_MISSING)

    def warm(self) -> None:
        self._load()

    def _load(self) -> dict:
        with self._lock:
            if self._saved is None:
                import joblib

                saved = joblib.load(self.model_path)
                if list(saved["labels"]) != LABELS:
                    raise ValueError("label order mismatch")
                self._saved = saved
            return self._saved

    def raw(self, text: str) -> list[float]:
        saved = self._load()
        proba = saved["classifier"].predict_proba(saved["vectorizer"].transform([text]))
        return [float(v) for v in proba[0]]


class DistilbertScorer(Scorer):
    """Fine-tuned DistilBERT on CPU, loaded lazily; same preprocessing as test inference."""

    local = True

    def __init__(self, model_dir: Path = DISTILBERT_DIR, max_length: int = MAX_LENGTH) -> None:
        super().__init__("distilbert")
        self.model_dir = model_dir
        self.max_length = max_length
        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()
        if not model_dir.is_dir():
            self.disable("model files not found", MODEL_MISSING)

    def warm(self) -> None:
        self._load()

    def _load(self):
        with self._lock:
            if self._model is None:
                from transformers import AutoModelForSequenceClassification, AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
                model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
                if model.config.num_labels != len(LABELS):
                    raise ValueError("unexpected number of outputs")
                self._model = model.to("cpu").eval()
            return self._model

    def raw(self, text: str) -> list[float]:
        import torch

        self._load()
        from policyjev.baselines.distilbert import tokenize

        ids = tokenize([text], self._tokenizer, self.max_length)[0]
        input_ids = torch.tensor([ids], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            logits = self._model(input_ids=input_ids, attention_mask=attention_mask).logits
        return [float(v) for v in torch.sigmoid(logits.float())[0].tolist()]


class JevScorer(Scorer):
    """TypeSafe JEV with question set v2; every call goes to the API and nothing is cached."""

    def __init__(
        self,
        key_loader: Callable[[], str] | None = None,
        questions_path: Path = QUESTIONS_V2,
    ) -> None:
        super().__init__("jev")
        if key_loader is None:
            from policyjev.jev.client import get_api_key

            key_loader = get_api_key
        self._key_loader = key_loader
        self._questions_path = questions_path
        self._client = None
        self._questions = None
        self._checked = False
        self._lock = threading.Lock()

    def available(self) -> bool:
        if self.reason is None and not self._checked:
            self._checked = True
            try:
                self._key_loader()
            except Exception:
                self.disable("API key not configured", NO_API_KEY)
        return self.reason is None

    def raw(self, text: str) -> list[float]:
        with self._lock:
            if self._client is None:
                from policyjev.jev.client import JevClient, build_questions

                cfg = load_config()["jev"]
                with self._questions_path.open("r", encoding="utf-8") as f:
                    self._questions = build_questions(yaml.safe_load(f)["questions"])
                # the live path records nothing, so no price is needed
                self._client = JevClient.from_config(cfg, self._key_loader(), price_per_million_input_tokens=0.0)
        result = self._client.evaluate(text, self._questions)
        return [float(result.answers[label]) for label in LABELS]


class ScoringService:
    """Combines scorers with the frozen calibrators and the primary-target policy thresholds."""

    def __init__(
        self,
        scorers: Mapping[str, Scorer],
        calibrators: Mapping[str, dict],
        policy_table: pd.DataFrame | None,
    ) -> None:
        self.scorers = dict(scorers)
        self.calibrators = dict(calibrators)
        self.thresholds: dict[str, pd.DataFrame] = {}
        self._loading: set[str] = set()
        self._state_lock = threading.Lock()
        self._warm_thread: threading.Thread | None = None
        if policy_table is not None:
            for system in self.scorers:
                try:
                    self.thresholds[system] = thresholds_for(
                        policy_table, system, PRIMARY_RECALL, PRIMARY_PRECISION
                    )
                except ValueError:
                    continue

    def _unavailable(self, system: str) -> tuple[str | None, str] | None:
        """Why a system cannot score, as (text, stable code); None if nothing stops it."""
        scorer = self.scorers.get(system)
        if scorer is None:
            return "unknown system", UNKNOWN_SYSTEM
        if system not in self.calibrators:
            return "calibrator not found", CALIBRATOR_MISSING
        if system not in self.thresholds:
            return "policy thresholds not found", THRESHOLDS_MISSING
        if not scorer.available():
            return scorer.reason, scorer.reason_code or UNAVAILABLE
        return None

    def reason_code(self, system: str) -> str | None:
        """Stable machine-readable reason a system is unavailable; None when it is available or loading."""
        problem = self._unavailable(system)
        return None if problem is None else problem[1]

    def status(self, system: str) -> tuple[str, str | None]:
        """State of a system (available, loading or unavailable) and the reason if unavailable."""
        problem = self._unavailable(system)
        if problem is not None:
            return "unavailable", problem[0]
        with self._state_lock:
            if system in self._loading:
                return "loading", None
        return "available", None

    def health(self) -> dict[str, tuple[str, str | None]]:
        """State of every known system."""
        return {system: self.status(system) for system in SYSTEMS if system in self.scorers}

    def start_warmup(self) -> threading.Thread | None:
        """Load the available local models in a background thread; they show as loading meanwhile."""
        pending = [
            s
            for s in SYSTEMS
            if s in self.scorers
            and getattr(self.scorers[s], "local", False)
            and self.status(s)[0] == "available"
        ]
        if not pending:
            return None
        with self._state_lock:
            self._loading.update(pending)

        def run() -> None:
            for system in pending:
                try:
                    self.scorers[system].warm()
                except Exception as error:
                    logger.error("model failed to load for %s (%s)", system, type(error).__name__)
                    self.scorers[system].disable("model failed to load", LOAD_FAILED)
                finally:
                    with self._state_lock:
                        self._loading.discard(system)

        thread = threading.Thread(target=run, name="model-warmup", daemon=True)
        self._warm_thread = thread
        thread.start()
        return thread

    def score(
        self, text: str, systems: list[str] | None = None, include_raw: bool = False
    ) -> tuple[list[dict], list[dict]]:
        """Score one comment; returns (results, issues). The text is never retained."""
        requested = systems if systems else [s for s in SYSTEMS if s in self.scorers]
        results, issues = [], []
        for system in requested:
            state, reason = self.status(system)
            if state == "loading":
                issues.append({"system": system, "status": "loading", "reason": "model is loading"})
                continue
            if state == "unavailable":
                issues.append(
                    {
                        "system": system,
                        "status": "unavailable",
                        "reason": reason,
                        "reason_code": self.reason_code(system),
                    }
                )
                continue
            try:
                started = time.perf_counter()
                raw = self.scorers[system].raw(text)
                saved = self.calibrators[system]
                calibrated = [
                    float(apply_calibrator(saved["labels"][label], np.array([raw[j]]))[0])
                    for j, label in enumerate(LABELS)
                ]
                outcome = decide(dict(zip(LABELS, calibrated)), self.thresholds[system])
                elapsed_ms = (time.perf_counter() - started) * 1000.0
            except Exception as error:
                logger.error("scoring failed for %s (%s)", system, type(error).__name__)
                issues.append({"system": system, "status": "error", "reason": "scoring failed"})
                continue
            entry = {
                "system": system,
                "probabilities": dict(zip(LABELS, calibrated)),
                "decision": outcome.decision,
                "triggering_labels": list(outcome.triggering_labels),
                "time_ms": elapsed_ms,
                "raw_probabilities": dict(zip(LABELS, raw)) if include_raw else None,
            }
            results.append(entry)
        return results, issues


def verify_frozen_inputs(
    scorers: Mapping[str, Scorer], jev_version: str = JEV_VERSION
) -> dict[str, str]:
    """Check the freeze manifest entries each system uses; returns {system: reason} for failures."""
    thresholds_file = POLICY_THRESHOLDS
    failures: dict[str, str] = {}

    def check(system: str, files: list[Path], model_folder: Path | None = None) -> None:
        try:
            verify_files(files)
            if model_folder is not None:
                verify_model_folder("artifacts/models/distilbert", model_folder)
        except ManifestError as error:
            logger.warning("frozen file check failed for %s: %s", system, error)
            failures[system] = "frozen manifest check failed"

    check("logreg_word", [LOGREG_MODEL, calibrator_path("logreg_word", None), thresholds_file])
    check("distilbert", [calibrator_path("distilbert", None), thresholds_file], DISTILBERT_DIR)
    check("jev", [calibrator_path("jev", jev_version), QUESTIONS_V2, thresholds_file])
    return {s: r for s, r in failures.items() if s in scorers}


def build_default_service() -> ScoringService:
    """The service for the real models: lazy scorers, frozen calibrators, manifest-checked."""
    scorers: dict[str, Scorer] = {
        "logreg_word": LogregScorer(),
        "distilbert": DistilbertScorer(),
        "jev": JevScorer(),
    }
    calibrators: dict[str, dict] = {}
    for system, path in (
        ("logreg_word", calibrator_path("logreg_word", None)),
        ("distilbert", calibrator_path("distilbert", None)),
        ("jev", calibrator_path("jev", JEV_VERSION)),
    ):
        try:
            calibrators[system] = load_calibrators(path)
        except FileNotFoundError:
            logger.warning("calibrator file missing for %s", system)
    policy_table = pd.read_csv(POLICY_THRESHOLDS) if POLICY_THRESHOLDS.is_file() else None

    for system, reason in verify_frozen_inputs(scorers).items():
        if scorers[system].reason is None:
            scorers[system].disable(reason, MANIFEST_FAILED)
    return ScoringService(scorers, calibrators, policy_table)
