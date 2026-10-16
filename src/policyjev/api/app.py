"""FastAPI backend for the moderation console (local use only).

Typed text is never written to disk, logged, cached or echoed in error messages.
"""

import logging
from collections.abc import Callable
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from policyjev.api.results import clean, load_results_payload
from policyjev.api.schemas import (
    HealthResponse,
    ScoreRequest,
    ScoreResponse,
    SystemHealth,
    ThresholdRow,
)
from policyjev.api.scoring import POLICY_THRESHOLDS, ScoringService, build_default_service

logger = logging.getLogger(__name__)

ALLOWED_ORIGINS = ["http://localhost:5173", "http://127.0.0.1:5173"]


def primary_threshold_rows(table: pd.DataFrame) -> list[dict]:
    """Primary-target threshold rows with infinity encoded as null."""
    primary = table[table["is_primary"].astype(str).str.lower() == "true"]
    return [
        {
            "system": str(row["system"]),
            "label": str(row["label"]),
            "low": clean(row["low"]),
            "high": clean(row["high"]),
            "action_enabled": str(row["action_enabled"]).lower() == "true",
        }
        for _, row in primary.iterrows()
    ]


def create_app(
    service: ScoringService | None = None,
    policy_table: pd.DataFrame | None = None,
    results_provider: Callable[[], dict] | None = None,
    warm_up: bool = True,
) -> FastAPI:
    """Build the app; defaults use the real models, frozen files and committed metrics.

    With warm_up, the available local models load in a background thread at startup.
    """
    service = service if service is not None else build_default_service()
    results_provider = results_provider if results_provider is not None else load_results_payload

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if warm_up:
            service.start_warmup()
        yield

    app = FastAPI(title="PolicyJEV moderation console API", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, error: RequestValidationError) -> JSONResponse:
        details = [
            {"loc": list(item.get("loc", ())), "msg": item.get("msg", ""), "type": item.get("type", "")}
            for item in error.errors()
        ]
        return JSONResponse(status_code=422, content={"detail": details})

    @app.exception_handler(Exception)
    async def unexpected_error(request: Request, error: Exception) -> JSONResponse:
        logger.error("unhandled error (%s)", type(error).__name__)
        return JSONResponse(status_code=500, content={"detail": "internal error"})

    @app.get("/api/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            systems={
                system: SystemHealth(status=state, reason=reason, reason_code=service.reason_code(system))
                for system, (state, reason) in service.health().items()
            },
        )

    @app.get("/api/thresholds", response_model=list[ThresholdRow])
    def thresholds() -> list[ThresholdRow]:
        table = policy_table
        if table is None:
            if not POLICY_THRESHOLDS.is_file():
                return []
            table = pd.read_csv(POLICY_THRESHOLDS)
        return [ThresholdRow(**row) for row in primary_threshold_rows(table)]

    @app.get("/api/results")
    def results() -> dict:
        return results_provider()

    @app.post("/api/score", response_model=ScoreResponse, response_model_exclude_none=True)
    def score(request: ScoreRequest) -> ScoreResponse:
        found, issues = service.score(request.text, request.systems, request.include_raw)
        return ScoreResponse(results=found, issues=issues)

    return app
