"""Shape- and coefficient-aware VMC/DMC regression tests (v0.2)."""

import numpy as np
import pytest

from qmc import DMCConfig, VMCConfig, run_dmc_importance, run_vmc
from qmc.configuration import (
    configuration_norm2,
    initial_configuration_walkers,
    require_positive_kinetic,
    walker_broadcast_mask,
)
from qmc.evaluation import evaluate_local_energy, evaluate_trial_quantities
from qmc.propagators import drift_diffusion_metropolis, drift_diffusion_second_order


class ProductGaussian:
    """Exact ground-state trial for independent harmonic oscillators."""

    def __init__(self, n_particles, alpha):
        self.n_particles = n_particles
        self.alpha = alpha

    def log_psi(self, R):
        return -0.5 * self.alpha * configuration_norm2(R)

    def grad_log_psi(self, R):
        return -self.alpha * R

    def laplacian_log_psi(self, R):
        return np.full(len(R), -3 * self.n_particles * self.alpha)


def many_oscillators(R):
    return 0.5 * configuration_norm2(R)


@pytest.mark.parametrize("D", [0.5, 0.2, 1.1])
def test_exact_manybody_local_energy(D):
    n = 4
    alpha = 1.0 / np.sqrt(2.0 * D)
    R = np.random.default_rng(25).normal(size=(12, n, 3))
    _, drift, local = evaluate_trial_quantities(
        many_oscillators, ProductGaussian(n, alpha), R,
        kinetic_coefficient=D,
    )
    assert drift.shape == R.shape
    np.testing.assert_allclose(local, 3.0 * n * D * alpha, atol=1e-12)


@pytest.mark.parametrize("D", [0.5, 0.2])
def test_manybody_vmc_smoke(D):
    n = 3
    alpha = 1.0 / np.sqrt(2.0 * D)
    r = run_vmc(
        many_oscillators,
        ProductGaussian(n, alpha),
        VMCConfig(
            n_walkers=32, n_steps=28, burn_in=10, sample_every=2,
            adaptive_proposal=False, configuration_shape=(n, 3),
            kinetic_coefficient=D, seed=173,
        ),
    )
    assert r.walkers.shape == (32, n, 3)
    np.testing.assert_allclose(r.energy_history, 3.0*n*D*alpha, atol=1e-12)


@pytest.mark.parametrize("scheme", ["metropolis", "second_order"])
@pytest.mark.parametrize("D", [0.5, 0.2])
def test_manybody_dmc_smoke(scheme, D):
    n = 3
    alpha = 1.0 / np.sqrt(2.0 * D)
    r = run_dmc_importance(
        many_oscillators,
        ProductGaussian(n, alpha),
        DMCConfig(
            n_walkers=32, n_steps=28, equilibration_steps=9,
            adaptation_steps=0, dt=0.03, adaptive_dt=False,
            scheme=scheme, configuration_shape=(n, 3),
            kinetic_coefficient=D, seed=197, log_interval=0,
        ),
    )
    assert r.walkers.shape == (32, n, 3)
    np.testing.assert_allclose(r.local_energy_history, 3.0*n*D*alpha, atol=1e-12)
    np.testing.assert_allclose(r.reference_energy_history, 3.0*n*D*alpha, atol=1e-12)


def test_configuration_helpers_and_validation():
    values = np.arange(24).reshape((2, 3, 4))
    np.testing.assert_array_equal(configuration_norm2(values), np.sum(values**2, axis=(1,2)))
    mask = walker_broadcast_mask(np.array([True, False]), 3)
    assert mask.shape == (2, 1, 1)
    np.testing.assert_array_equal(np.broadcast_to(mask, values.shape)[:,0,0], [True,False])
    with pytest.raises(ValueError):
        require_positive_kinetic(0.0)
    with pytest.raises(ValueError):
        require_positive_kinetic(float('nan'))
    with pytest.raises(ValueError):
        initial_configuration_walkers(
            n_walkers=3, configuration_shape=(3,3), initial_sigma=1.0,
            rng=np.random.default_rng(1), initial_walkers=np.ones((3,3)),
        )


def test_nondefault_kinetic_ignores_legacy_local_energy_shortcut():
    class WithLegacyShortcut(ProductGaussian):
        def local_energy(self, R):
            return np.full(len(R), -999.)
    trial = WithLegacyShortcut(2, 1.0)
    R = np.ones((3, 2, 3))
    e_legacy = evaluate_local_energy(many_oscillators, trial, R)
    np.testing.assert_array_equal(e_legacy, -999.)
    e_new = evaluate_local_energy(
        many_oscillators, trial, R, kinetic_coefficient=0.2
    )
    assert np.all(np.isfinite(e_new))
    assert not np.allclose(e_new, -999.)


@pytest.mark.parametrize('scheme', ['metropolis', 'second_order'])
def test_free_diffusion_variance_scaling(scheme):
    class FreeTrial:
        def log_psi(self, R): return np.zeros(len(R))
        def grad_log_psi(self, R): return np.zeros_like(R)
        def laplacian_log_psi(self, R): return np.zeros(len(R))
    def zero_potential(R): return np.zeros(len(R))
    R = np.zeros((8, 2, 3))
    trial = FreeTrial()
    D, dt, seed = 0.2, 0.05, 54321
    rng = np.random.default_rng(seed)
    rng_check = np.random.default_rng(seed)
    if scheme == 'metropolis':
        moved, *_ = drift_diffusion_metropolis(
            R, trial.log_psi(R), trial.grad_log_psi(R),
            np.zeros(len(R)), zero_potential, trial, dt, rng,
            kinetic_coefficient=D,
        )
        expected = np.sqrt(2*D*dt) * rng_check.normal(size=R.shape)
    else:
        moved, *_ = drift_diffusion_second_order(
            R, zero_potential, trial, dt, rng, kinetic_coefficient=D,
        )
        expected = np.sqrt(D*dt) * (
            rng_check.normal(size=R.shape) + rng_check.normal(size=R.shape)
        )
    np.testing.assert_allclose(moved, expected, atol=1e-14)
