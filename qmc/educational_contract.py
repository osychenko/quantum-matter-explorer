"""Patch 32: versioned, curated experiment contract for the educational MVP.

This module deliberately does not expose arbitrary research samplers through the
public educational API. A recorded benchmark is NOT a live simulation.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import exp, isfinite, pi, sqrt
from typing import Any

SCHEMA_VERSION = '1.0.0'
ENGINE_RELEASE = '0.3.0.dev0'


@dataclass(frozen=True)
class Preset:
    id: str
    title: str
    status: str
    mode: str
    description: str
    adjustable: tuple[str, ...]


_PRESETS = (
    Preset('ideal-boson-addition-v1', 'Free bosons: particle addition', 'supported',
           'analytic_control', 'Canonical N=1 -> 2 ideal Bose gas, zero-momentum midpoint Green function.',
           ('beta', 'cell_length')),
    Preset('softcore-m8-baseline-recorded-v1', 'Interacting bosons: recorded benchmark',
           'candidate_recording', 'recorded_result',
           'Previously measured M=8 baseline worm result; not a validated live engine preset.', ()),
)
_PRESET_MAP = {p.id: p for p in _PRESETS}

# Frozen, explicitly attributed measurements, NOT generated on each invocation.
_RECORDED_M8 = {
    'beta': 2.4, 'M': 8, 'cell_length': 2.8, 'potential_height': 3.0,
    'potential_cutoff': 1.2, 'method': 'baseline', 'replicas': 8,
    'green': 1.685053810648142, 'sem': 0.04352458506961887,
    'spectral_k2': 1.6910255956087066,
    'source': 'Patch 30 trace experiment; Patch 31 readiness report; seeds 2380000–2380007',
    'screen_status': 'candidate',
}


def catalog() -> list[dict[str, Any]]:
    return [dict(id=p.id, title=p.title, status=p.status, mode=p.mode,
                 description=p.description, adjustable=list(p.adjustable)) for p in _PRESETS]


def _finite_real(x: Any, *, name: str, low: float, high: float) -> float:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise ValueError(f'{name} must be a real number')
    v = float(x)
    if not isfinite(v) or not low <= v <= high:
        raise ValueError(f'{name} must be finite in [{low}, {high}]')
    return v


def _free_thermal_sum(beta: float, L: float, cutoff: int) -> float:
    """Finite momentum cube sum for the one-boson partition function.

    E_n = 0.5 (2*pi/L)^2 |n|^2. The 3D sum factorizes.
    """
    a = beta * 0.5 * (2*pi/L)**2
    one_dim = 1.0 + 2.0 * sum(exp(-a*n*n) for n in range(1, cutoff+1))
    return one_dim**3


def _ideal_green(beta: float, length: float) -> tuple[float, float]:
    # Canonical N=1 thermal host: G^+_0(t)=1+exp(-beta*epsilon0)/Z1
    # epsilon0=0 for k=0, so this is independent of t at fixed beta.
    z_8 = _free_thermal_sum(beta, length, 8)
    z_10 = _free_thermal_sum(beta, length, 10)
    return 1.0 + 1.0/z_10, abs(1.0/z_8-1.0/z_10)


def _canon(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(',', ':'), allow_nan=False)


def run_experiment(preset_id: str, parameters: dict[str, Any] | None = None) -> dict[str, Any]:
    """Produce JSON-serializable, versioned output for curated MVP scenarios.

    No arbitrary Python class name, sample count, physical Hamiltonian, or
    unrestricted sampler option is permitted by this interface.
    """
    if preset_id not in _PRESET_MAP:
        raise ValueError('Unknown or research-only preset: '+str(preset_id))
    preset = _PRESET_MAP[preset_id]
    params = {} if parameters is None else parameters
    if not isinstance(params, dict):
        raise ValueError('parameters must be an object')
    unsupported = set(params) - set(preset.adjustable)
    if unsupported:
        raise ValueError('parameters not allowed for preset: '+', '.join(sorted(unsupported)))
    if preset.id == 'ideal-boson-addition-v1':
        beta = _finite_real(params.get('beta', 2.4), name='beta', low=0.5, high=5.0)
        length = _finite_real(params.get('cell_length', 2.8), name='cell_length', low=1.5, high=5.0)
        green, truncation_gap = _ideal_green(beta, length)
        input_params = dict(beta=beta, cell_length=length, kinetic_coefficient=0.5,
                            n_background=1, momentum_mode=[0, 0, 0], t_fraction=0.5,
                            interaction_height=0.0)
        observable = dict(id='green_addition_midpoint_k0', value=green, units='dimensionless',
                          uncertainty=None, uncertainty_kind='not_applicable_analytic',
                          numerical_truncation_indicator=truncation_gap)
        provenance = dict(method='canonical_ideal_bose_analytic_control',
                          model='3d periodic ideal Bose gas, N=1 thermal host',
                          numerical_note='One-boson partition sum evaluated to |n_i|<=10; last-shell sensitivity is provided')
        cautions = ['Numerical truncation indicator is not a rigorous tail bound.',
                    'This is an imaginary-time correlation, not a real-time particle trajectory.']
    else:
        r = _RECORDED_M8
        input_params = {k:r[k] for k in ('beta','M','cell_length','potential_height','potential_cutoff','method')}
        observable = dict(id='green_addition_midpoint_k0', value=r['green'], units='dimensionless',
                          uncertainty=r['sem'], uncertainty_kind='independent_replica_sem',
                          spectral_reference_k2=r['spectral_k2'])
        provenance = dict(method='recorded_worm_benchmark', source=r['source'],
                          replicas=r['replicas'], readiness_screen=r['screen_status'],
                          warning='Recorded result, NOT a new Monte Carlo simulation')
        cautions = ['Candidate only: readiness screening does not prove continuum convergence.',
                    'The K=2 spectral value is a finite-basis numerical reference.',
                    'No parameter variation or real-time particle dynamics is available.']
    fingerprint = sha256(_canon({'preset_id':preset_id,'parameters':input_params,
                                 'mode':preset.mode,'provenance':provenance}).encode()).hexdigest()
    return dict(schema_version=SCHEMA_VERSION, engine_release=ENGINE_RELEASE,
                experiment_id=preset_id, preset_status=preset.status,
                execution_mode=preset.mode, parameters=input_params,
                observables=[observable], provenance=provenance,
                cautions=cautions, fingerprint_sha256=fingerprint)


def validate_result(result: dict[str, Any]) -> None:
    """Validate the public response's essential versioned contract."""
    if not isinstance(result, dict) or result.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('unsupported response schema')
    if result.get('experiment_id') not in _PRESET_MAP:
        raise ValueError('unrecognized experiment')
    if result.get('execution_mode') != _PRESET_MAP[result['experiment_id']].mode:
        raise ValueError('misrepresented execution mode')
    if not isinstance(result.get('observables'), list) or not result['observables']:
        raise ValueError('missing observables')
    for obs in result['observables']:
        if not isinstance(obs.get('value'), (int, float)) or not isfinite(obs['value']):
            raise ValueError('invalid observable value')
        if obs.get('uncertainty') is not None and (
            not isinstance(obs['uncertainty'], (int,float)) or
            not isfinite(obs['uncertainty']) or obs['uncertainty'] < 0
        ):
            raise ValueError('invalid observable uncertainty')
    for key in ('parameters','provenance','cautions','fingerprint_sha256'):
        if key not in result:
            raise ValueError('missing '+key)
    _canon(result)
