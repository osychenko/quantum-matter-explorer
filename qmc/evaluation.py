from __future__ import annotations

from typing import Callable

import numpy as np

from .configuration import configuration_norm2, require_positive_kinetic

Array = np.ndarray
Potential = Callable[[Array], Array]


def evaluate_potential(potential: Potential, walkers: Array) -> Array:
    values = np.asarray(potential(walkers), dtype=np.float64)
    expected = (len(walkers),)
    if values.shape != expected:
        raise ValueError(
            f"Potential must return shape {expected}; got {values.shape} "
            f"for walkers.shape={walkers.shape}."
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("Potential returned NaN or infinity.")
    return values


def evaluate_local_energy(
    potential: Potential, trial, walkers: Array, *, kinetic_coefficient: float = 0.5
) -> Array:
    """Evaluate E_L=(H Psi_T)/Psi_T, H=-D*Laplacian+V.

    Existing specialized trial.local_energy() implementations assume D=0.5.
    For other D, use the analytic log-derivatives instead, avoiding silently
    applying the wrong kinetic energy.
    """
    D = require_positive_kinetic(kinetic_coefficient)
    if D == 0.5 and hasattr(trial, "local_energy"):
        e_local = np.asarray(trial.local_energy(walkers), dtype=np.float64)
    else:
        potential_values = evaluate_potential(potential, walkers)
        grad = np.asarray(trial.grad_log_psi(walkers), dtype=np.float64)
        lap = np.asarray(trial.laplacian_log_psi(walkers), dtype=np.float64)
        if grad.shape != walkers.shape:
            raise ValueError(
                f"trial.grad_log_psi() returned shape {grad.shape}, "
                f"expected {walkers.shape}"
            )
        grad2 = configuration_norm2(grad)
        e_local = potential_values - D * (lap + grad2)

    expected = (len(walkers),)
    if e_local.shape != expected:
        raise ValueError(f"Local energy must return shape {expected}; got {e_local.shape}.")
    if not np.all(np.isfinite(e_local)):
        raise ValueError("Local energy contains NaN or infinity.")
    return e_local


def evaluate_trial_quantities(
    potential: Potential, trial, walkers: Array, *, kinetic_coefficient: float = 0.5
):
    """Return log(Psi_T), grad log(Psi_T), and E_L with one validation path."""
    n = len(walkers)
    log_psi = np.asarray(trial.log_psi(walkers), dtype=np.float64)
    drift = np.asarray(trial.grad_log_psi(walkers), dtype=np.float64)

    if log_psi.shape != (n,):
        raise ValueError(f"trial.log_psi() must return {(n,)}, got {log_psi.shape}.")
    if drift.shape != walkers.shape:
        raise ValueError(
            f"trial.grad_log_psi() must return {walkers.shape}, got {drift.shape}."
        )
    if not np.all(np.isfinite(log_psi)) or not np.all(np.isfinite(drift)):
        raise ValueError("Trial wavefunction returned NaN or infinity.")

    e_local = evaluate_local_energy(
        potential, trial, walkers, kinetic_coefficient=kinetic_coefficient
    )
    return log_psi, drift, e_local
