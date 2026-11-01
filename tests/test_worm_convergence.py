"""Patch 22: independently diagonalized interacting G and conditional worm statistics."""

from dataclasses import replace

import numpy as np
import pytest

from qmc.bosons import SoftCorePairPotential
from qmc.cells import CubicCell
from qmc.continuum_spacetime_worm import exact_free_addition_green
from qmc.green_spectral import TwoBosonAdditionReference
from qmc.reliable_spacetime_worm import (ReliableContinuumSpacetimeWorm,
                                          ReliableSpacetimeConfig)
from examples.bosons.run_worm_convergence import _check_grid, _mean_sem, run_experiment

CELL = CubicCell(2.8)


def make(*, height=0.0, background=1, **params):
    cfg = replace(ReliableSpacetimeConfig(n_links=4, time_step=0.6,
        n_steps=110, burn_in=20, thin=2, seed=123), **params)
    p = SoftCorePairPotential(CELL, cutoff=1.2, height=height)
    return ReliableContinuumSpacetimeWorm(CELL, p, background, cfg)


@pytest.mark.parametrize("k_max", [1, 2])
def test_spectral_free_matches_analytic_one_boson(k_max):
    r = TwoBosonAdditionReference(height=0, k_max=k_max)
    times = np.array([0., .3, .9, 1.7])
    modes = np.asarray(((0, 0, 0), (1, 0, 0), (0, 0, 1)))
    spectral = r.measure(2.4, times, modes)
    exact = exact_free_addition_green(CELL, 1, 2.4, times, modes)
    np.testing.assert_allclose(spectral.green, exact, rtol=2e-9, atol=2e-9)
    assert spectral.n_one_body_modes == (2*k_max+1)**3


def test_spectral_equal_time_commutator_and_repulsion():
    beta = 2.4
    interacting = TwoBosonAdditionReference(height=3, k_max=1)
    free = TwoBosonAdditionReference(height=0, k_max=1)
    g = interacting.measure(beta, [0., .6, 1.2], ((0,0,0),)).green[:, 0]
    f = free.measure(beta, [0., .6, 1.2], ((0,0,0),)).green[:, 0]
    assert g[0] == pytest.approx(f[0], abs=1e-12)
    assert g[1] < f[1] and g[2] < f[2]
    assert np.all(g > 0)


def test_spectral_basis_convergence_for_test_case():
    a = TwoBosonAdditionReference(height=3., k_max=1, n_grid=16)
    b = TwoBosonAdditionReference(height=3., k_max=2, n_grid=16)
    g1 = a.measure(2.4,[.6,1.2],((0,0,0),(1,0,0))).green
    g2 = b.measure(2.4,[.6,1.2],((0,0,0),(1,0,0))).green
    assert np.max(np.abs(g1-g2)) < .01  # test parameters only, not general certificate


@pytest.mark.parametrize("bad", [
    {"cell_length":-1}, {"kinetic_coefficient":0}, {"height":-1},
    {"cutoff":1.5}, {"k_max":0}, {"k_max":3}, {"n_grid":4},
])
def test_spectral_bad_config(bad):
    with pytest.raises(ValueError):
        TwoBosonAdditionReference(**bad)


@pytest.mark.parametrize("beta,times,modes", [
    (0.,[.2],((0,0,0),)),
    (1.,[1.],((0,0,0),)),
    (1.,[-.1],((0,0,0),)),
    (1.,[.3],((3,0,0),)),
    (1.,[.3],((.2,0,0),)),
    (1.,[],((0,0,0),)),
])
def test_spectral_bad_measurement(beta,times,modes):
    with pytest.raises(ValueError):
        TwoBosonAdditionReference().measure(beta,times,modes)


def test_conditional_probability_matches_log_sector_weight():
    sampler=make(height=2)
    log_r = sampler.log_relative_weight(sampler.host,sampler.cut,sampler.length,sampler.extra)
    p = sampler.conditional_open_probability()
    assert p/(1-p) == pytest.approx(np.exp(log_r), rel=1e-12)
    assert not sampler.open
    # Gibbs keeps the augmented auxiliary variables; the open/closed
    # probability ratio is R even though the ghost reference remains fixed.
    for _ in range(5):
        sampler.move_sector_gibbs()
        assert sampler.conditional_open_probability() == pytest.approx(p)


def test_vacuum_gibbs_probability_is_exact_and_rb_k0_is_one():
    sampler=make(background=0, height=0, sector_fugacity=.8,
                 n_steps=400, burn_in=80)
    assert sampler.conditional_open_probability() == pytest.approx(.8/1.8)
    result=sampler.run()
    assert np.mean(result.green[:, 0]) == pytest.approx(1.0, abs=1e-12)
    # Individual length ratios fluctuate with the sampled ghost length.
    assert np.all(np.isfinite(result.green_raw))
    assert result.n_closed > 10
    assert result.mean_closed_probability == pytest.approx(1/1.8, abs=1e-12)


def test_refresh_keeps_closed_ghost_law_and_open_validity():
    sampler=make(height=2.5)
    sampler.move_ghost_refresh()
    sampler.move_global_refresh()
    assert sampler.extra.shape==(sampler.length+1,3)
    assert np.isfinite(sampler.log_relative_weight(sampler.host,sampler.cut,sampler.length,sampler.extra))
    sampler.open=True
    with pytest.raises(RuntimeError):
        sampler.move_ghost_refresh()
    sampler.move_global_refresh()
    assert sampler.extra.shape==(sampler.length+1,3)
    assert np.isfinite(sampler.current_log_relative_weight())


def test_reproducible_gibbs_and_raw_estimates():
    a,b=make(height=3.,seed=27,n_steps=280),make(height=3.,seed=27,n_steps=280)
    ra, rb=a.run(),b.run()
    np.testing.assert_array_equal(ra.green,rb.green)
    np.testing.assert_array_equal(ra.green_raw,rb.green_raw)
    np.testing.assert_array_equal(ra.n_open_by_length,rb.n_open_by_length)
    assert ra.closed_effective_samples > 0
    assert ra.conditional_effective_samples > 0
    assert ra.effective_samples_by_length.shape==(3,)


def test_short_free_one_boson_check():
    sampler=make(height=0., seed=87,n_steps=1400,burn_in=200)
    got=sampler.run()
    exact=exact_free_addition_green(CELL,1,got.beta,got.imaginary_times,got.indices)
    assert np.max(np.abs(got.green[:,0]-exact[:,0])) < .65
    assert got.mean_closed_probability > 0
    assert np.isfinite(got.green).all()


def test_config_and_grid_validation():
    with pytest.raises(ValueError):
        make(ghost_refresh_probability=1.5)
    with pytest.raises(TypeError):
        from qmc.continuum_spacetime_worm import ContinuumSpacetimeConfig
        ReliableContinuumSpacetimeWorm(CELL,SoftCorePairPotential(CELL,height=0,cutoff=1.2),1,
                                       ContinuumSpacetimeConfig())
    for args in [((0,), (4,), 2, 300, 40), ((2.4,), (2,), 2, 300, 40),
                 ((2.4,), (4,), 1, 300, 40), ((2.4,), (4,), 2, 10, 2)]:
        with pytest.raises(ValueError):
            _check_grid(*args)
    assert _mean_sem([1,3]) == pytest.approx((2,1))


def test_quick_example_file_contract(tmp_path):
    _, summary, reference=run_experiment(tmp_path,quick=True,n_steps=180,replicas=2)
    assert len(summary)==2*3*2  # two physical cases, 3 intervals, two modes
    assert len(reference)==2*2*3*2 # 2 cases, K=1/2, three lengths, two modes
    for f in ("replicas.csv", "summary.csv", "spectral_reference.csv",
              "protocol.json", "green_spectral.png", "bias_diagnostic.png"):
        assert (tmp_path/f).stat().st_size > 100
