"""Patch 33: bounded local-first API for the Patch 32 educational contract.

Run with: python3 -m uvicorn app.api:app --host 127.0.0.1 --port 8000
No arbitrary scripts, research samplers, filesystem paths, or job submissions
are exposed to HTTP clients.
"""
from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from qmc.educational_contract import (
    SCHEMA_VERSION, ENGINE_RELEASE, catalog, run_experiment, validate_result,
)

app = FastAPI(
    title="Quantum Matter Explorer — Educational API",
    version="0.1.0",
    description="Curated analytic controls and historical benchmark records. No live QMC jobs.",
)


class RunRequest(BaseModel):
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


@app.get("/api/v1/experiments")
def experiments():
    return {"schema_version": SCHEMA_VERSION, "presets": catalog()}


@app.post("/api/v1/experiments/{preset_id}/run")
def run(preset_id: str, request: RunRequest):
    try:
        result = run_experiment(preset_id, request.parameters)
        validate_result(result)
        return result
    except ValueError as exc:
        raise _bad_input(exc) from exc


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
