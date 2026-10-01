from __future__ import annotations

from typing import Callable

import numpy as np

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


def evaluate_local_energy(potential: Potential, trial, walkers: Array) -> Array:
    """Evaluate E_L=(H Psi_T)/Psi_T for all walkers."""
    if hasattr(trial, "local_energy"):
        e_local = np.asarray(trial.local_energy(walkers), dtype=np.float64)
    else:
        potential_values = evaluate_potential(potential, walkers)
        grad = np.asarray(trial.grad_log_psi(walkers), dtype=np.float64)
        lap = np.asarray(trial.laplacian_log_psi(walkers), dtype=np.float64)
        grad2 = np.sum(grad * grad, axis=1)
        e_local = potential_values - 0.5 * (lap + grad2)

    expected = (len(walkers),)
    if e_local.shape != expected:
        raise ValueError(f"Local energy must return shape {expected}; got {e_local.shape}.")
    if not np.all(np.isfinite(e_local)):
        raise ValueError("Local energy contains NaN or infinity.")
    return e_local


def evaluate_trial_quantities(potential: Potential, trial, walkers: Array):
    """Return log(Psi_T), grad log(Psi_T), and E_L with one validation path."""
    n = len(walkers)
    log_psi = np.asarray(trial.log_psi(walkers), dtype=np.float64)
    drift = np.asarray(trial.grad_log_psi(walkers), dtype=np.float64)

    if log_psi.shape != (n,):
        raise ValueError(f"trial.log_psi() must return {(n,)}, got {log_psi.shape}.")
    if drift.shape != (n, 3):
        raise ValueError(f"trial.grad_log_psi() must return {(n, 3)}, got {drift.shape}.")
    if not np.all(np.isfinite(log_psi)) or not np.all(np.isfinite(drift)):
        raise ValueError("Trial wavefunction returned NaN or infinity.")

    e_local = evaluate_local_energy(potential, trial, walkers)
    return log_psi, drift, e_local
