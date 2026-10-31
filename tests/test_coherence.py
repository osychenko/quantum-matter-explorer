"""Finite-cell variational one-body coherence tests (Patch 9)."""

from dataclasses import replace

import numpy as np
import pytest

from qmc.bosons import SoftCoreJastrowTrial
from qmc.cells import BCCCell, CubicCell, PeriodicCell
from qmc.coherence import (
    OneBodyDensityMatrix,
    VariationalCondensateOccupation,
    one_body_ratio,
)
from qmc.order import DensityWaveJastrowTrial


class UniformTrial:
    def log_psi(self, R):
        return np.zeros(len(R))


class PeriodicOrbital:
    """Analytic one-particle reference: f(x)=1+a cos(2pi x/L)."""

    def __init__(self, cell, a):
        self.cell, self.a = cell, a

    def log_psi(self, R):
        return np.log(1 + self.a * np.cos(2 * np.pi * R[:, 0, 0] / self.cell.length))


def _free(cell, n=5, w=13):
    return np.random.default_rng(11).uniform(-15, 20, (w, n, 3))


def test_fast_ratio_matches_direct_trial_difference():
    rng = np.random.default_rng(81)
    cell = CubicCell(5)
    trial = DensityWaveJastrowTrial(
        cell, jastrow_amplitude=0.8, order_amplitude=0.13, cutoff=2,
        indices=((1, 0, 0), (0, 1, 0)),
    )
    R = rng.uniform(-4, 9, (17, 4, 3))
    ids = rng.integers(4, size=17)
    candidates = cell.wrap(R[np.arange(17), ids] + rng.normal(size=(17, 3)))
    old_positions = R.copy()
    proposals = R.copy()
    proposals[np.arange(17), ids] = candidates
    expected = np.exp(trial.log_psi(proposals) - trial.log_psi(R))
    np.testing.assert_allclose(one_body_ratio(trial, R, ids, candidates), expected,
                               rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(R, old_positions)  # no walker mutation


def test_fallback_wavefunction_ratio():
    cell = CubicCell(5)
    R = _free(cell, 1, 4)
    proposal = R[:, 0] + np.array([1., 0, 0])
    trial = PeriodicOrbital(cell, 0.7)
    ratio = one_body_ratio(trial, R, np.zeros(len(R), dtype=int), proposal)
    np.testing.assert_allclose(ratio, np.exp(trial.log_psi(proposal[:, None]) - trial.log_psi(R)))


@pytest.mark.parametrize("cell", [CubicCell(5), BCCCell(6), PeriodicCell([[4,0,0],[1,5,0],[0.5,0.2,6]])])
def test_constant_wavefunction_has_exact_full_coherence(cell):
    R = _free(cell)
    trial = UniformTrial()
    observer = OneBodyDensityMatrix(trial, cell, 5,
                                    np.linspace(0, 0.9, 5), seed=10)
    occupancy = VariationalCondensateOccupation(trial, cell, 5, seed=11)
    for _ in range(3):
        observer.accumulate(R)
        occupancy.accumulate(R)
    g1 = observer.result()
    n0 = occupancy.result()
    np.testing.assert_allclose(g1.g1, 1, rtol=0, atol=0)
    assert n0.n0_over_n == 1.0
    assert g1.n_configurations == 3 * len(R)
    assert g1.n_displacement_samples_per_radius == 3 * len(R)
    assert n0.n_displacement_samples == 3 * len(R)
    assert np.array_equal(g1.radius, np.linspace(0, .9, 5))


def test_zero_radius_exactly_one_correlated_trial():
    cell = CubicCell(5)
    trial = SoftCoreJastrowTrial(cell, amplitude=0.8, cutoff=2.3)
    observer = OneBodyDensityMatrix(trial, cell, 4, (0., 0.3, 0.8, 1.4), seed=5)
    observer.accumulate(_free(cell, 4, 12))
    assert observer.result().g1[0] == 1.0
    assert np.all(np.isfinite(observer.result().g1))
    assert np.all(observer.result().g1 > 0)


def test_displacement_estimator_reproducible_and_incremental():
    cell = CubicCell(5)
    trial = SoftCoreJastrowTrial(cell, amplitude=1.0, cutoff=2.0)
    R = _free(cell, 4, 25)
    def sample():
        g = OneBodyDensityMatrix(trial, cell, 4, (0., 0.2, 0.9, 1.6),
                                 draws_per_configuration=2, seed=113)
        n = VariationalCondensateOccupation(trial, cell, 4,
                                           draws_per_configuration=2, seed=114)
        for _ in range(2):
            g.accumulate(R)
            n.accumulate(R)
        return g.result(), n.result()
    a, c = sample()
    b, d = sample()
    np.testing.assert_array_equal(a.g1, b.g1)
    assert c == d
    assert a.n_displacement_samples_per_radius == 100
    assert c.n_displacement_samples == 100


def test_analytic_k_zero_occupation_for_modulated_one_particle():
    # For f(x)=1+a cos(2*pi*x/L), the exact occupancy is
    # |integral f|²/(V integral f²)=1/(1+a²/2). Tests the 3D volume integral.
    cell = CubicCell(4)
    a = 0.8
    trial = PeriodicOrbital(cell, a)
    rng = np.random.default_rng(17)
    candidates = rng.uniform(0, cell.length, (40000, 1, 3))
    accept_prob = ((1 + a*np.cos(2*np.pi*candidates[:, 0, 0]/4)) / (1+a))**2
    R = candidates[rng.random(len(candidates)) < accept_prob][:5000]
    assert len(R) == 5000
    observer = VariationalCondensateOccupation(trial, cell, 1, seed=112)
    for batch in np.array_split(R, 5):
        observer.accumulate(batch)
    assert observer.result().n0_over_n == pytest.approx(1/(1+a*a/2), abs=0.035)


@pytest.mark.parametrize("bad_radii", [(), (0.,), (0.1, 0.2), (0., 0.),
                                       (0., 0.5, 0.4), (0., float('nan')),
                                       (0., 4.)])
def test_reject_invalid_radial_grids(bad_radii):
    with pytest.raises(ValueError):
        OneBodyDensityMatrix(UniformTrial(), CubicCell(5), 2, bad_radii)


@pytest.mark.parametrize("kwargs", [dict(n_particles=0),dict(draws_per_configuration=0),
                                      dict(seed=-3),dict(n_particles=2.1)])
def test_reject_bad_configuration(kwargs):
    params = dict(trial=UniformTrial(), cell=CubicCell(5), n_particles=3)
    params.update(kwargs)
    with pytest.raises(ValueError):
        VariationalCondensateOccupation(**params)


def test_reject_bad_input_move_and_empty_result():
    trial, cell = UniformTrial(), CubicCell(5)
    obs = OneBodyDensityMatrix(trial, cell, 3, (0., 0.3))
    with pytest.raises(ValueError, match='no configurations'):
        obs.result()
    with pytest.raises(ValueError, match='no configurations'):
        VariationalCondensateOccupation(trial, cell, 3).result()
    with pytest.raises(ValueError, match='positions'):
        obs.accumulate(np.ones((5, 3)))
    R = _free(cell, 3, 4)
    with pytest.raises(ValueError, match='particle_indices'):
        one_body_ratio(trial, R, np.array([0, 1, 2, 3]), np.ones((4, 3)))
    with pytest.raises(ValueError, match='candidate_positions'):
        one_body_ratio(trial, R, np.zeros(4,dtype=int), np.ones((4, 4)))


def test_observers_are_read_only():
    trial, cell = SoftCoreJastrowTrial(CubicCell(5), amplitude=.7, cutoff=2), CubicCell(5)
    R = _free(cell, 3, 9)
    old = R.copy()
    OneBodyDensityMatrix(trial, cell, 3, [0,.3,.9], seed=3).accumulate(R)
    VariationalCondensateOccupation(trial, cell, 3, seed=4).accumulate(R)
    np.testing.assert_array_equal(R, old)


def test_finite_experiment_smoke(tmp_path):
    from examples.bosons.run_coherence import CoherenceExperimentConfig, run_coherence_experiment
    cfg = CoherenceExperimentConfig(n_particles=3, cell_length=4, cutoff=1.8,
                                    n_walkers=8, n_steps=14, burn_in=4, sample_every=2,
                                    n_radii=4, draws_per_configuration=1, replicas=2)
    rows, summary, obdm = run_coherence_experiment(cfg, tmp_path)
    assert len(rows) == 6
    assert len(summary) == 3
    assert len(obdm) == 12
    for file in ['replicas.csv','summary.csv','obdm.csv','protocol.json',
                 'obdm.png','occupation.png']:
        assert (tmp_path/file).stat().st_size > 20
    assert next(r for r in summary if r['case']=='ideal_reference')['n0_over_n'] == 1.
    from examples.bosons.run_coherence import _validate
    with pytest.raises(ValueError):
        _validate(replace(cfg, replicas=1))


def test_mismatched_trial_and_observer_cell_rejected():
    trial = SoftCoreJastrowTrial(CubicCell(4), amplitude=.6, cutoff=1.8)
    with pytest.raises(ValueError, match='observer cell'):
        VariationalCondensateOccupation(trial, CubicCell(5), 3)
