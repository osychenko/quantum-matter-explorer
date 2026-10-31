"""Periodic soft-core pair Hamiltonian and bosonic Jastrow regression tests."""

import numpy as np
import pytest

from qmc import (
    BCCCell,
    CubicCell,
    SoftCoreJastrowTrial,
    SoftCorePairPotential,
    VMCConfig,
    run_vmc,
)
from qmc.bosons import pair_displacements
from qmc.evaluation import evaluate_local_energy


def test_repulsion_pair_counting_and_translation():
    cell = CubicCell(10.0)
    potential = SoftCorePairPotential(cell, height=4.0, cutoff=4.0)
    R = np.array([[[0.2, 0.0, 0.0], [9.8, 0.0, 0.0], [5.0, 5.0, 5.0]]])
    f = (1.0 - (0.4 / 4.0)**2)**3
    np.testing.assert_allclose(potential(R), [4.0 * f])
    shifted = R.copy()
    shifted[0, 1, :] += [20.0, -10.0, 0.0]
    np.testing.assert_allclose(potential(shifted), potential(R))
    i, j, dr = pair_displacements(R, cell)
    assert len(i) == 3 and len(j) == 3 and dr.shape == (1, 3, 3)


def test_symmetric_trial_and_fast_particle_ratio():
    rng = np.random.default_rng(90)
    cell = CubicCell(7.0)
    trial = SoftCoreJastrowTrial(cell, amplitude=0.65, cutoff=3.0)
    R = rng.uniform(0.0, 7.0, size=(7, 5, 3))
    np.testing.assert_allclose(trial.log_psi(R[:, ::-1]), trial.log_psi(R))
    translations = np.array([7.0, -14.0, 21.0])
    R2 = R.copy()
    R2[:, 2, :] += translations
    np.testing.assert_allclose(trial.log_psi(R2), trial.log_psi(R))
    old = trial.log_psi(R)
    new_positions = R[:, 2, :] + rng.normal(0, 0.3, size=(len(R), 3))
    fast_delta = trial.log_ratio_particle_move(R, 2, new_positions)
    R2 = R.copy()
    R2[:, 2, :] = new_positions
    np.testing.assert_allclose(fast_delta, trial.log_psi(R2) - old, atol=1e-12)


def test_jastrow_analytic_gradient_and_laplacian():
    cell = CubicCell(12.0)
    trial = SoftCoreJastrowTrial(cell, amplitude=0.78, cutoff=5.0)
    R = np.array([[[1.1, 2.0, 3.0], [2.4, 2.3, 2.8], [3.0, 4.0, 1.2]]])
    eps = 1e-4
    analytic_grad = trial.grad_log_psi(R)[0]
    analytic_lap = trial.laplacian_log_psi(R)[0]
    grad_fd = np.zeros_like(R[0])
    lap_fd = 0.0
    psi0 = trial.log_psi(R)[0]
    for j in range(R.shape[1]):
        for d in range(3):
            plus, minus = R.copy(), R.copy()
            plus[0, j, d] += eps
            minus[0, j, d] -= eps
            qp, qm = trial.log_psi(plus)[0], trial.log_psi(minus)[0]
            grad_fd[j, d] = (qp - qm) / (2.0 * eps)
            lap_fd += (qp - 2.0 * psi0 + qm) / eps**2
    np.testing.assert_allclose(analytic_grad, grad_fd, rtol=2e-6, atol=2e-7)
    np.testing.assert_allclose(analytic_lap, lap_fd, rtol=2e-5, atol=3e-6)


def test_one_particle_zero_energy_and_no_gradients():
    cell = CubicCell(6.0)
    potential = SoftCorePairPotential(cell, height=1.0, cutoff=2.0)
    trial = SoftCoreJastrowTrial(cell, amplitude=0.9, cutoff=2.0)
    R = np.ones((3, 1, 3))
    np.testing.assert_array_equal(potential(R), np.zeros(3))
    np.testing.assert_array_equal(trial.log_psi(R), np.zeros(3))
    np.testing.assert_array_equal(trial.grad_log_psi(R), np.zeros_like(R))
    np.testing.assert_array_equal(trial.laplacian_log_psi(R), np.zeros(3))


def test_bcc_periodic_pair_evaluation():
    cell = BCCCell(8.0)
    cutoff = 3.0  # sqrt(3)*8/4 = 3.464...
    potential = SoftCorePairPotential(cell, height=2.0, cutoff=cutoff)
    trial = SoftCoreJastrowTrial(cell, amplitude=0.3, cutoff=cutoff)
    R = np.array([[[0.0, 0.0, 0.0], [3.9, 3.9, 3.9]]])
    assert potential(R)[0] > 0.0  # nearest BCC translate is at (4,4,4)
    assert trial.log_psi(R)[0] < 0.0


def test_cutoff_validated_and_parameter_errors():
    cell = CubicCell(6.0)
    with pytest.raises(ValueError, match="minimum-image"):
        SoftCorePairPotential(cell, height=1.0, cutoff=3.1)
    with pytest.raises(ValueError):
        SoftCoreJastrowTrial(cell, amplitude=-0.2, cutoff=2.0)
    with pytest.raises(ValueError):
        SoftCorePairPotential(cell, height=-1.0, cutoff=2.0)
    with pytest.raises(ValueError):
        SoftCoreJastrowTrial(cell, amplitude=1.0, cutoff=0.0)


def test_pairwise_vmc_updates_and_no_legacy_behavior_change():
    cell = CubicCell(5.0)
    n_particles, n_walkers = 4, 24
    trial = SoftCoreJastrowTrial(cell, amplitude=0.6, cutoff=2.3)
    potential = SoftCorePairPotential(cell, height=2.0, cutoff=2.3)
    initial = np.random.default_rng(96).uniform(0, 5, size=(n_walkers, n_particles, 3))
    r = run_vmc(
        potential, trial,
        VMCConfig(
            n_walkers=n_walkers,
            configuration_shape=(n_particles, 3),
            proposal_mode="particle",
            n_steps=36,
            burn_in=8,
            sample_every=2,
            proposal_sigma=0.5,
            adaptive_proposal=False,
            seed=72,
        ),
        initial_walkers=initial,
    )
    assert r.walkers.shape == initial.shape
    assert len(r.energy_history) == 14
    assert np.all(np.isfinite(r.energy_history))
    assert np.all(r.walkers >= -1e-12)
    assert np.all(r.walkers < 5.0 + 1e-12)
    assert 0 < r.production_acceptance <= 1
    assert np.isfinite(evaluate_local_energy(potential, trial, r.walkers)).all()
    with pytest.raises(ValueError, match="proposal_mode"):
        run_vmc(potential, trial, VMCConfig(proposal_mode="bad", n_steps=4, burn_in=1))
    with pytest.raises(ValueError, match="requires"):
        run_vmc(potential, trial, VMCConfig(proposal_mode="particle", n_steps=4, burn_in=1))


def test_particle_vmc_fallback_without_incremental_ratio():
    """Particle mode also works for trials without the optional fast API."""
    from qmc.configuration import configuration_norm2

    class ProductGaussian:
        def log_psi(self, R):
            return -0.5 * configuration_norm2(R)

        def grad_log_psi(self, R):
            return -R

        def laplacian_log_psi(self, R):
            return np.full(len(R), -6.0)

    def harmonic(R):
        return 0.5 * configuration_norm2(R)

    result = run_vmc(
        harmonic, ProductGaussian(),
        VMCConfig(
            configuration_shape=(2, 3), proposal_mode="particle",
            n_walkers=16, n_steps=12, burn_in=4, sample_every=1,
            adaptive_proposal=False, seed=123,
        ),
    )
    assert result.walkers.shape == (16, 2, 3)
    np.testing.assert_allclose(result.energy_history, 3.0, atol=1e-12)


@pytest.mark.parametrize("scheme", ["metropolis", "second_order"])
def test_small_bosonic_dmc_smoke(scheme):
    """Integration test for the existing DMC driver with the new pair model."""
    from qmc import DMCConfig, run_dmc_importance

    cell = CubicCell(5.0)
    potential = SoftCorePairPotential(cell, height=2.0, cutoff=2.1)
    trial = SoftCoreJastrowTrial(cell, amplitude=0.4, cutoff=2.1)
    start = np.random.default_rng(25).uniform(0, 5, size=(32, 4, 3))
    result = run_dmc_importance(
        potential, trial,
        DMCConfig(
            n_walkers=32, configuration_shape=(4, 3),
            n_steps=20, equilibration_steps=5, adaptation_steps=0,
            dt=0.01, adaptive_dt=False, scheme=scheme,
            log_interval=0, seed=42,
        ),
        initial_walkers=start,
    )
    assert result.walkers.shape == (32, 4, 3)
    assert np.all(np.isfinite(result.local_energy_history))
    assert np.all(np.isfinite(result.reference_energy_history))
