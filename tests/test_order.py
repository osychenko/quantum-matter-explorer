"""Analytic and symmetry checks for exchange-symmetric collective trial states."""

import numpy as np
import pytest

from qmc.bosons import SoftCoreJastrowTrial, SoftCorePairPotential
from qmc.cells import BCCCell, CubicCell, PeriodicCell
from qmc.evaluation import evaluate_local_energy
from qmc.order import DensityWaveJastrowTrial, canonical_modes


MODES = ((2, 0, 0), (0, 2, 0), (0, 0, 2))


def trial(cell, a=0.4, order=0.15, cutoff=2.5, modes=MODES):
    return DensityWaveJastrowTrial(
        cell, jastrow_amplitude=a, order_amplitude=order,
        cutoff=cutoff, indices=modes,
    )


def test_zero_order_exactly_matches_soft_core_jastrow():
    rng = np.random.default_rng(12)
    cell = CubicCell(6.0)
    R = rng.uniform(0, 6, size=(5, 4, 3))
    base = SoftCoreJastrowTrial(cell, amplitude=0.5, cutoff=2.5)
    ordered = trial(cell, a=0.5, order=0.0)
    np.testing.assert_allclose(ordered.log_psi(R), base.log_psi(R), atol=1e-14)
    np.testing.assert_allclose(ordered.grad_log_psi(R), base.grad_log_psi(R), atol=1e-14)
    np.testing.assert_allclose(ordered.laplacian_log_psi(R),
                               base.laplacian_log_psi(R), atol=1e-14)


def test_exchange_and_translation_invariance():
    rng = np.random.default_rng(42)
    cell = BCCCell(8.0)
    R = rng.uniform(0, 8, size=(6, 5, 3))
    t = trial(cell, a=0.3, order=0.17, cutoff=3.0, modes=((1, 0, 0), (0, 1, 0)))
    np.testing.assert_allclose(t.log_psi(R[:, [2, 0, 4, 1, 3]]), t.log_psi(R), atol=1e-12)
    for shift in (cell.vectors[0], cell.vectors[1] - cell.vectors[2]):
        shifted = R.copy()
        shifted[:, 1] += shift
        np.testing.assert_allclose(t.log_psi(shifted), t.log_psi(R), atol=2e-12)
    shifted = R + np.array([0.17, 0.21, -1.93])
    np.testing.assert_allclose(t.log_psi(shifted), t.log_psi(R), atol=2e-12)


def test_analytic_gradient_laplacian_against_finite_differences():
    cell = CubicCell(8.0)
    t = trial(cell, a=0.6, order=0.14, cutoff=3.5, modes=((1, 0, 0), (0, 1, 1)))
    R = np.array([[[1.2, 1.4, 0.8], [2.0, 2.2, 1.5], [4.0, 2.8, 0.4]]])
    eps = 1e-4
    base = t.log_psi(R)[0]
    fd_grad = np.zeros((3, 3))
    fd_lap = 0.0
    for i in range(3):
        for d in range(3):
            plus, minus = R.copy(), R.copy()
            plus[0, i, d] += eps
            minus[0, i, d] -= eps
            value_plus = t.log_psi(plus)[0]
            value_minus = t.log_psi(minus)[0]
            fd_grad[i, d] = (value_plus - value_minus) / (2 * eps)
            fd_lap += (value_plus - 2 * base + value_minus) / eps**2
    np.testing.assert_allclose(t.grad_log_psi(R)[0], fd_grad, atol=1e-7, rtol=2e-6)
    np.testing.assert_allclose(t.laplacian_log_psi(R)[0], fd_lap, atol=1e-6, rtol=2e-5)
    potential = SoftCorePairPotential(cell, height=2.0, cutoff=3.5)
    for d in (0.5, 0.2):
        energies = evaluate_local_energy(potential, t, R, kinetic_coefficient=d)
        assert energies.shape == (1,) and np.isfinite(energies).all()


def test_incremental_particle_move_is_exact():
    rng = np.random.default_rng(77)
    cell = CubicCell(8.0)
    t = trial(cell)
    R = rng.uniform(0, 8, size=(8, 6, 3))
    new = R[:, 4, :] + rng.normal(size=(8, 3)) * 0.2
    moved = R.copy()
    moved[:, 4, :] = new
    np.testing.assert_allclose(t.log_ratio_particle_move(R, 4, new),
                               t.log_psi(moved) - t.log_psi(R), atol=1e-12)
    with pytest.raises(IndexError):
        t.log_ratio_particle_move(R, -1, new)


@pytest.mark.parametrize("modes", [[], [(0, 0, 0)], [(1, 0, 0), (-1, 0, 0)],
                                   [(1, 0, 0), (1, 0, 0)], [(0.5, 0, 0)]])
def test_bad_reciprocal_mode_sets_rejected(modes):
    with pytest.raises(ValueError):
        canonical_modes(modes)


def test_parameter_validation():
    cell = CubicCell(6.0)
    with pytest.raises(ValueError):
        trial(cell, order=-0.2)
    with pytest.raises(ValueError):
        trial(cell, order=float("nan"))
    with pytest.raises(TypeError):
        trial("not cell")


@pytest.mark.parametrize("scheme", ["metropolis", "second_order"])
def test_density_wave_trial_dmc_smoke(scheme):
    """Integration check; requires repository's real DMC driver."""
    from qmc import DMCConfig, run_dmc_importance
    cell = CubicCell(5.0)
    t = DensityWaveJastrowTrial(
        cell, jastrow_amplitude=0.5, order_amplitude=0.05,
        cutoff=2.2, indices=((1, 0, 0), (0, 1, 0)),
    )
    potential = SoftCorePairPotential(cell, height=2.0, cutoff=2.2)
    initial = np.random.default_rng(12).uniform(0, 5, size=(24, 4, 3))
    result = run_dmc_importance(
        potential, t,
        DMCConfig(n_walkers=24, configuration_shape=(4, 3),
                  n_steps=18, equilibration_steps=6, adaptation_steps=0,
                  dt=0.01, adaptive_dt=False, scheme=scheme,
                  log_interval=0, seed=15),
        initial_walkers=initial,
    )
    assert result.walkers.shape == initial.shape
    assert np.all(np.isfinite(result.local_energy_history))
    assert np.all(np.isfinite(result.reference_energy_history))
