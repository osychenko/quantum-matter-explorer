"""Fast, deterministic scientific reference tests for the current 3D engine."""

import numpy as np
import pytest

from qmc import (
    DMCConfig,
    GaussianTrial,
    VMCConfig,
    YukawaCuspTrial,
    YukawaExponentialTrial,
    harmonic_oscillator,
    run_dmc_importance,
    run_vmc,
)
from qmc.evaluation import evaluate_trial_quantities
from qmc.potentials import yukawa_potential


def test_harmonic_oscillator_exact_local_energy():
    """For alpha=omega=1, the Gaussian is the exact 3D ground state."""
    walkers = np.array([
        [0.0, 0.0, 0.0],
        [0.25, -0.5, 0.75],
        [-2.0, 1.0, 0.5],
    ])
    log_psi, drift, local_energy = evaluate_trial_quantities(
        harmonic_oscillator, GaussianTrial(alpha=1.0), walkers
    )
    assert log_psi.shape == (3,)
    assert drift.shape == walkers.shape
    np.testing.assert_allclose(local_energy, 1.5, rtol=0, atol=1e-14)


def test_harmonic_oscillator_vmc_smoke():
    result = run_vmc(
        harmonic_oscillator,
        GaussianTrial(alpha=1.0),
        VMCConfig(
            n_walkers=48,
            n_steps=35,
            burn_in=10,
            sample_every=2,
            adaptive_proposal=False,
            seed=20,
        ),
    )
    assert len(result.energy_history) > 0
    np.testing.assert_allclose(result.energy_history, 1.5, rtol=0, atol=1e-12)
    assert result.local_energy_variance >= 0.0


@pytest.mark.parametrize("scheme", ["metropolis", "second_order"])
def test_harmonic_oscillator_dmc_smoke(scheme):
    """Exact local energy makes this a precise DMC/branching regression."""
    config = DMCConfig(
        n_walkers=48,
        n_steps=32,
        adaptation_steps=0,
        equilibration_steps=10,
        dt=0.02,
        adaptive_dt=False,
        scheme=scheme,
        log_interval=0,
        seed=21,
    )
    result = run_dmc_importance(
        harmonic_oscillator,
        GaussianTrial(alpha=1.0),
        config,
    )
    assert result.walkers.shape == (48, 3)
    np.testing.assert_allclose(result.local_energy_history, 1.5, rtol=0, atol=1e-12)
    np.testing.assert_allclose(result.reference_energy_history, 1.5, rtol=0, atol=1e-12)
    if scheme == "second_order":
        assert np.isnan(result.production_acceptance)
    else:
        assert 0.0 <= result.production_acceptance <= 1.0


def test_yukawa_exponential_cusp_is_finite():
    trial = YukawaExponentialTrial(g=1.0, mu=0.5, alpha=1.0)
    walkers = np.array([[1e-8, 0, 0], [1e-4, 0, 0], [1.0, 0, 0]])
    local = trial.local_energy(walkers)
    assert np.all(np.isfinite(local))
    # Limit r -> 0 is g*mu - g*g/2 = 0 for this parameter set.
    assert abs(local[0]) < 1e-7


def test_yukawa_cusp_log_derivative():
    trial = YukawaCuspTrial(g=1.0, mu=0.5, alpha=0.66, beta=0.185)
    r = np.array([[1e-5, 0, 0], [2e-5, 0, 0]], dtype=float)
    log_values = trial.log_psi(r)
    derivative = (log_values[1] - log_values[0]) / 1e-5
    assert derivative == pytest.approx(-1.0, abs=1e-5)
    assert np.all(np.isfinite(trial.grad_log_psi(r)))


def test_yukawa_potential_is_vectorized():
    xyz = np.array([[0.5, 0.0, 0.0], [2.0, 0.0, 0.0]])
    energy = yukawa_potential(xyz, g=1.0, mu=0.5)
    expected = -np.exp(-0.5 * np.array([0.5, 2.0])) / np.array([0.5, 2.0])
    np.testing.assert_allclose(energy, expected)
