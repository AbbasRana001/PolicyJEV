"""Request and response models for the moderation console API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

SystemName = Literal["logreg_word", "distilbert", "jev"]
ReasonCode = Literal[
    "no_api_key",
    "model_missing",
    "manifest_failed",
    "load_failed",
    "calibrator_missing",
    "thresholds_missing",
    "unknown_system",
    "unavailable",
]


class ScoreRequest(BaseModel):
    """A comment to score: 1 to 2000 characters, optionally restricted to some systems."""

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, max_length=2000)
    systems: list[SystemName] | None = Field(default=None, min_length=1)
    include_raw: bool = False


class SystemScore(BaseModel):
    """One system's calibrated probabilities and policy decision."""

    system: SystemName
    probabilities: dict[str, float]
    decision: str
    triggering_labels: list[str]
    time_ms: float
    raw_probabilities: dict[str, float] | None = None


class SystemIssue(BaseModel):
    """A requested system that could not score the comment."""

    system: SystemName
    status: Literal["unavailable", "loading", "error"]
    reason: str
    reason_code: ReasonCode | None = None


class ScoreResponse(BaseModel):
    """Scores for the available systems, and the systems that were skipped."""

    results: list[SystemScore]
    issues: list[SystemIssue]


class SystemHealth(BaseModel):
    """State of one system; the reason is given when it is unavailable."""

    status: Literal["available", "loading", "unavailable"]
    reason: str | None = None
    reason_code: ReasonCode | None = None


class HealthResponse(BaseModel):
    """API status and per-system availability."""

    status: Literal["ok"]
    systems: dict[str, SystemHealth]


class ThresholdRow(BaseModel):
    """Policy thresholds for one system and label (null HIGH means ACTION is disabled)."""

    system: str
    label: str
    low: float | None
    high: float | None
    action_enabled: bool
