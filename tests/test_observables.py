"""Finite-size-normalized g(r), reciprocal S(k), and production hooks."""

import numpy as np
import pytest

from qmc import CubicCell, BCCCell, VMCConfig, run_vmc
from qmc.observables import (
    PairCorrelation,
    StaticStructureFactor,
    reciprocal_index_grid,
    safe_radial_limit,
)
from qmc.visualization import _bcc_wireframe, plot_cell_snapshot, plot_pair_correlation, plot_structure_factor


def test_uniform_ideal_gas_gr_normalization():
    cell = CubicCell(10.0)
    obs = PairCorrelation(cell, 8, n_bins=12, r_max=4.5)
    rng = np.random.default_rng(16)
    # Accumulating in batches must give the same result as one large batch.
    a = rng.uniform(0, 10, size=(9000, 8, 3))
    obs.accumulate(a[:3000])
    obs.accumulate(a[3000:])
    r = obs.result()
    assert r.n_configurations == 9000
    assert r.pair_counts.sum() > 0
    np.testing.assert_allclose(r.g_r[1:], 1.0, atol=0.08, rtol=0)
    assert abs(r.g_r[0] - 1.0) < 0.25  # sparse innermost shell


def test_gr_unordered_pair_count_and_pbc():
    cell = CubicCell(10.0)
    obs = PairCorrelation(cell, 3, n_bins=5, r_max=5)
    R = np.array([[[0., 0., 0.], [9.8, 0., 0.], [5., 5., 5.]]])
    obs.accumulate(R)
    result = obs.result()
    assert result.pair_counts.sum() == 1
    assert result.pair_counts[0] == 1
    with pytest.raises(ValueError, match="safe radial"):
        PairCorrelation(cell, 3, r_max=5.01)


def test_static_structure_factor_uncorrelated_baseline_and_invariance():
    cell = CubicCell(10.0)
    indices = np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [1, 1, 0]])
    rng = np.random.default_rng(7)
    R = rng.uniform(0, 10, size=(5000, 6, 3))
    a = StaticStructureFactor(cell, 6, indices)
    b = StaticStructureFactor(cell, 6, indices)
    a.accumulate(R)
    shifted = R + np.array([1.3, 2.4, -2.6])
    shifted[:, 1, :] += np.array([10., -20., 30.])
    b.accumulate(shifted)
    np.testing.assert_allclose(a.result().s_k, b.result().s_k, atol=1e-13)
    np.testing.assert_allclose(a.result().s_k, 1.0, atol=0.07, rtol=0)
    assert a.result().s_k[0] == pytest.approx(a.result().s_k[2], abs=1e-13)


def test_bragg_peak_simple_example():
    cell = CubicCell(10)
    R = np.array([[[0., 1., 2.], [5., 1., 2.]]])
    s = StaticStructureFactor(cell, 2, [[1, 0, 0], [2, 0, 0]])
    s.accumulate(R)
    np.testing.assert_allclose(s.result().s_k, [0., 2.], atol=1e-13)


def test_bcc_safe_limit_and_reciprocal_modes():
    cell = BCCCell(8.)
    assert safe_radial_limit(cell) == pytest.approx(np.sqrt(3) * 2)
    assert reciprocal_index_grid(1).shape == (26, 3)
    v, edges = _bcc_wireframe(cell)
    assert v.shape == (24, 3)
    assert len(edges) == 36
    R = np.random.default_rng(9).uniform(0, 4, size=(8, 3, 3))
    g = PairCorrelation(cell, 3)
    s = StaticStructureFactor(cell, 3, reciprocal_index_grid(1))
    g.accumulate(R)
    s.accumulate(R)
    assert np.isfinite(g.result().g_r).all()
    assert np.isfinite(s.result().s_k).all()


def test_observable_validations():
    cell = CubicCell(6)
    with pytest.raises(ValueError):
        PairCorrelation(cell, 1)
    with pytest.raises(ValueError):
        reciprocal_index_grid(0)
    with pytest.raises(ValueError):
        StaticStructureFactor(cell, 3, [[0, 0, 0]])
    with pytest.raises(ValueError):
        StaticStructureFactor(cell, 3, [[1, 0, 0], [1, 0, 0]])
    with pytest.raises(ValueError):
        StaticStructureFactor(cell, 3, [[0.5, 0, 0]])
    g = PairCorrelation(cell, 3)
    with pytest.raises(ValueError, match="no configurations"):
        g.result()
    with pytest.raises(ValueError, match="positions"):
        g.accumulate(np.zeros((2, 4, 3)))


def test_vmc_production_hooks_do_not_change_sampling():
    cell = CubicCell(5.)

    class ZeroTrial:
        def log_psi(self, R): return np.zeros(len(R))
        def grad_log_psi(self, R): return np.zeros_like(R)
        def laplacian_log_psi(self, R): return np.zeros(len(R))

    def zero(R): return np.zeros(len(R))

    config = VMCConfig(
        n_walkers=10, configuration_shape=(3, 3), proposal_mode="particle",
        n_steps=18, burn_in=5, sample_every=3,
        adaptive_proposal=False, seed=123,
    )
    initial = np.random.default_rng(1).uniform(0, 5, (10, 3, 3))
    a = run_vmc(zero, ZeroTrial(), config, initial_walkers=initial)
    g = PairCorrelation(cell, 3, n_bins=5)
    s = StaticStructureFactor(cell, 3, [[1, 0, 0]])
    b = run_vmc(zero, ZeroTrial(), config, initial_walkers=initial,
                observables=(g, s))
    np.testing.assert_array_equal(a.walkers, b.walkers)
    np.testing.assert_array_equal(a.energy_history, b.energy_history)
    assert g.n_configurations == 10 * 5  # steps 5, 8, 11, 14, 17
    assert s.n_configurations == g.n_configurations


@pytest.mark.parametrize("scheme", ["metropolis", "second_order"])
def test_dmc_production_hook_and_same_random_stream(scheme):
    from qmc import DMCConfig, run_dmc_importance

    cell = CubicCell(5)

    class ZeroTrial:
        def log_psi(self, R): return np.zeros(len(R))
        def grad_log_psi(self, R): return np.zeros_like(R)
        def laplacian_log_psi(self, R): return np.zeros(len(R))

    def zero(R): return np.zeros(len(R))

    cfg = DMCConfig(
        n_walkers=12, configuration_shape=(3, 3), scheme=scheme,
        n_steps=14, equilibration_steps=5, adaptation_steps=0,
        adaptive_dt=False, log_interval=0, seed=72,
    )
    initial = np.random.default_rng(6).uniform(0, 5, (12, 3, 3))
    a = run_dmc_importance(zero, ZeroTrial(), cfg, initial_walkers=initial)
    g = PairCorrelation(cell, 3, n_bins=6)
    b = run_dmc_importance(zero, ZeroTrial(), cfg, initial_walkers=initial,
                           observables=(g,), observable_stride=3)
    np.testing.assert_array_equal(a.walkers, b.walkers)
    np.testing.assert_array_equal(a.local_energy_history, b.local_energy_history)
    assert g.n_configurations == 12 * 3  # production steps 5,8,11
    with pytest.raises(ValueError, match="observable_stride"):
        run_dmc_importance(zero, ZeroTrial(), cfg, initial_walkers=initial,
                           observable_stride=0)


def test_headless_image_generation(tmp_path):
    cell = BCCCell(8)
    positions = np.random.default_rng(3).uniform(0, 8, (4, 3))
    g = PairCorrelation(cell, 4, n_bins=5)
    s = StaticStructureFactor(cell, 4, reciprocal_index_grid(1))
    g.accumulate(positions[None, :, :])
    s.accumulate(positions[None, :, :])
    paths = [
        plot_cell_snapshot(positions, cell, tmp_path / "cell.png"),
        plot_pair_correlation(g.result(), tmp_path / "gr.png"),
        plot_structure_factor(s.result(), tmp_path / "sk.png"),
    ]
    assert all(p.is_file() and p.stat().st_size > 4000 for p in paths)
