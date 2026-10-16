"""Deterministic moderation decision engine: ALLOW, HUMAN REVIEW or ACTION.

Per label, p < LOW gives ALLOW, LOW <= p < HIGH gives HUMAN REVIEW and p >= HIGH gives ACTION.
A comment's decision is the most severe outcome across its labels. The engine does no file or
network access and keeps model scoring separate from the decision.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

ALLOW = "ALLOW"
HUMAN_REVIEW = "HUMAN REVIEW"
ACTION = "ACTION"
DECISIONS = {0: ALLOW, 1: HUMAN_REVIEW, 2: ACTION}


@dataclass(frozen=True)
class Decision:
    """A comment's decision and the labels that triggered it (empty for ALLOW)."""

    decision: str
    triggering_labels: tuple[str, ...]


def label_severity(probability: float, low: float, high: float) -> int:
    """Outcome for one label: 0 ALLOW, 1 HUMAN REVIEW, 2 ACTION."""
    if probability >= high:
        return 2
    if probability >= low:
        return 1
    return 0


def decide(probabilities: Mapping[str, float], thresholds: pd.DataFrame) -> Decision:
    """Decision for one comment from calibrated probabilities (label -> p).

    thresholds has one row per label with columns label, low and high, for one system at
    one grid point. Triggering labels are those whose own outcome equals the decision.
    """
    labels = thresholds["label"].tolist()
    severities = [
        label_severity(float(probabilities[label]), float(low), float(high))
        for label, low, high in zip(labels, thresholds["low"], thresholds["high"])
    ]
    overall = max(severities)
    triggering = tuple(label for label, s in zip(labels, severities) if overall > 0 and s == overall)
    return Decision(DECISIONS[overall], triggering)


def decide_frame(probabilities: pd.DataFrame, thresholds: pd.DataFrame) -> pd.DataFrame:
    """Vectorised decide: one row per comment with decision and triggering_labels columns."""
    labels = thresholds["label"].tolist()
    values = probabilities[labels].to_numpy(dtype=float)
    low = thresholds["low"].to_numpy(dtype=float)
    high = thresholds["high"].to_numpy(dtype=float)
    severity = np.where(values >= high, 2, np.where(values >= low, 1, 0))
    overall = severity.max(axis=1)
    triggered = (severity == overall[:, None]) & (overall[:, None] > 0)
    return pd.DataFrame(
        {
            "decision": [DECISIONS[int(s)] for s in overall],
            "triggering_labels": [
                tuple(label for label, hit in zip(labels, row) if hit) for row in triggered
            ],
        },
        index=probabilities.index,
    )
