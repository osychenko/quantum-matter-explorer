"""Patch 33: bounded local-first API for the Patch 32 educational contract.

Run with: python3 -m uvicorn app.api:app --host 127.0.0.1 --port 8000
No arbitrary scripts, research samplers, filesystem paths, or job submissions
are exposed to HTTP clients.
"""
from __future__ import annotations

from typing import Any
from threading import BoundedSemaphore

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from qmc.educational_science import science_map
from qmc.educational_comparisons import comparison_catalog, comparison_requires_live, run_comparison
from qmc.educational_contract import (
    SCHEMA_VERSION, ENGINE_RELEASE, catalog, run_experiment, validate_result,
)

_LIVE_VMC_SLOT = BoundedSemaphore(1)  # local-first bounded computations

app = FastAPI(
    title="Quantum Matter Explorer — Educational API",
    version="0.1.0",
    description="Curated analytic controls and historical benchmark records. No live QMC jobs.",
)


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    parameters: dict[str, Any] = Field(default_factory=dict)


class CompareRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    parameters: dict[str, Any] = Field(default_factory=dict)


class SweepRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    beta: float = Field(ge=0.5, le=5.0, allow_inf_nan=False)
    cell_length: float = Field(ge=1.5, le=5.0, allow_inf_nan=False)
    variable: str = Field(pattern=r"^(beta|cell_length)$")
    count: int = Field(default=21, ge=2, le=31)


def _bad_input(exc: ValueError) -> HTTPException:
    return HTTPException(status_code=422, detail=str(exc))


@app.get("/api/v1/health")
def health():
    return {"status": "ok", "schema_version": SCHEMA_VERSION,
            "engine_release": ENGINE_RELEASE, "execution": "local_analytic_and_recorded_only"}


@app.get("/api/v1/science-map")
def science():
    """Curated questions, physics concepts, methods, and parameter metadata.

    Research-only cards are descriptive; they do not unlock simulations.
    """
    return science_map()


@app.get("/api/v1/experiments")
def experiments():
    return {"schema_version": SCHEMA_VERSION, "presets": catalog()}


@app.post("/api/v1/experiments/{preset_id}/run")
def run(preset_id: str, request: RunRequest):
    live = preset_id in {'harmonic-oscillator-vmc-v1', 'softcore-structure-vmc-v1'}
    if live and not _LIVE_VMC_SLOT.acquire(blocking=False):
        raise HTTPException(status_code=429, detail='The local VMC laboratory is busy; retry when it finishes')
    try:
        try:
            result = run_experiment(preset_id, request.parameters)
            validate_result(result)
            return result
        except ValueError as exc:
            raise _bad_input(exc) from exc
    finally:
        if live:
            _LIVE_VMC_SLOT.release()


@app.get("/api/v1/comparisons")
def comparisons():
    """Allowed comparisons. The server, not the client, chooses both models."""
    return {"comparison_schema_version": "1.0.0", "comparisons": comparison_catalog()}


@app.post("/api/v1/comparisons/{comparison_id}/run")
def compare(comparison_id: str, request: CompareRequest):
    """Bounded curated two-arm study; a live comparison reserves one VMC slot."""
    live = comparison_requires_live(comparison_id)
    if live and not _LIVE_VMC_SLOT.acquire(blocking=False):
        raise HTTPException(status_code=429, detail='The local VMC laboratory is busy; retry when it finishes')
    try:
        try:
            return run_comparison(comparison_id, request.parameters)
        except ValueError as exc:
            raise _bad_input(exc) from exc
    finally:
        if live:
            _LIVE_VMC_SLOT.release()


@app.post("/api/v1/experiments/ideal-boson-addition-v1/sweep")
def ideal_sweep(request: SweepRequest):
    """Small deterministic visualization dataset; does not run Monte Carlo."""
    low, high = ((0.5, 5.0) if request.variable == "beta" else (1.5, 5.0))
    points = []
    for i in range(request.count):
        x = low + (high - low) * i / (request.count - 1)
        pars = {"beta": request.beta, "cell_length": request.cell_length,
                request.variable: x}
        result = run_experiment("ideal-boson-addition-v1", pars)
        points.append({"x": x, "green": result["observables"][0]["value"]})
    return {"variable": request.variable, "points": points,
            "kind": "analytic_control", "units": "dimensionless",
            "caution": "Imaginary-time correlation; not a real-time particle trajectory"}
