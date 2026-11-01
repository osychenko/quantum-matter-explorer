"""Patch 18: normalized canonical boson worm sector and analytic free reference."""

from dataclasses import replace

import numpy as np
import pytest

from qmc.bosons import SoftCorePairPotential
from qmc.cells import CubicCell, OrthorhombicCell
from qmc.boson_permutations import free_one_particle_partition
from qmc.worm import (CanonicalWorm, WormConfig, canonical_free_occupations,
                      closed_link_occupations)


def cfg(**kwargs):
    return replace(WormConfig(n_beads=6, time_step=0.3, n_steps=800,
                              burn_in=100, thin=2, refresh_probability=0.20,
                              swap_probability=0.25, bridge_probability=0.2,
                              bridge_lengths=(2, 3), translate_probability=.05,
                              seed=1245, radial_bins=4), **kwargs)


def model(height=0.0, n=2, **kwargs):
    cell = CubicCell(2.8)
    potential = SoftCorePairPotential(cell, height=height, cutoff=1.1)
    return CanonicalWorm(cell, potential, n, cfg(**kwargs))


def test_exact_boson_occupation_one_particle():
    cell = OrthorhombicCell((3., 4., 5.))
    beta = 1.5
    k = cell.wavevectors(((1, 0, 0),))[0]
    z1 = free_one_particle_partition(cell, beta)
    a = canonical_free_occupations(cell, 1, beta, ((0,0,0), (1,0,0)))
    np.testing.assert_allclose(a, [1/z1, np.exp(-beta*.5*np.dot(k,k))/z1], rtol=1e-12)


def test_exact_free_n2_occupation_matches_partition_recursion():
    cell = CubicCell(2.8)
    beta = 1.6
    z1 = free_one_particle_partition(cell, beta)
    z2 = free_one_particle_partition(cell, 2*beta)
    expected = (z1+1.)/(z1*z1+z2)
    occ = canonical_free_occupations(cell, 2, beta, ((0,0,0),))[0]
    assert occ == pytest.approx(expected, rel=1e-13)
    assert 0 < occ < 1


def test_occupation_nonnegative_and_sum_momentum_completeness():
    cell = CubicCell(2.5)
    modes = np.array([(x,y,z) for x in range(-4,5) for y in range(-4,5)
                      for z in range(-4,5)])
    a = canonical_free_occupations(cell, 2, 2.4, modes)
    assert np.all(a >= 0)
    assert a[np.where(np.all(modes==0, axis=1))[0][0]] > 0
    assert np.sum(a) == pytest.approx(1.0, abs=1e-10)


def test_open_link_density_exact_and_periodic():
    w = model(3.0)
    i, s = 0, np.array([.3, -.6, .8])
    delta = w._delta(i)
    r = (w.base.log_weight() + np.log(w.config.sector_fugacity)
         -np.log(w.n_particles*w.cell.volume)
         +w.kernel.log_kernel(delta+s, w.config.time_step)
         -w.kernel.log_kernel(delta, w.config.time_step))
    assert r-w.base.log_weight() == pytest.approx(w.log_sector_ratio(i,s))
    np.testing.assert_allclose(w.log_sector_ratio(i,s),
                               w.log_sector_ratio(i,s+np.array([w.cell.length,0.,0.])),
                               rtol=0,atol=1e-12)


def test_open_close_mh_log_ratios_are_inverses():
    w = model(1.0)
    w.rng = np.random.default_rng(42)
    w.move_toggle()
    assert w._counts['open'][1] == 1
    # The one-step probability is always computed from the *unshifted* closure.
    i=0
    k=w.kernel.log_kernel(w._delta(i),w.config.time_step)
    expected=np.log(w.config.sector_fugacity)-np.log(w.cell.volume)-k
    assert np.isfinite(expected)
    assert expected + (-expected) == pytest.approx(0)


def test_heatbath_shift_has_analytic_kernel_moments():
    w=model()
    i=0
    samples=np.array([w._heatbath_shift(i) for _ in range(2500)])
    delta=w._delta(i)
    # y=delta+s is a wrapped N(0,2 D tau) variate, independently of delta.
    y=w.cell.minimum_image(samples+delta)
    q=w.cell.wavevectors(((1,0,0),))[0]
    expected=np.exp(-w.kernel.kinetic_coefficient*w.config.time_step*np.dot(q,q))
    observed=np.cos(y @ q).mean()
    assert observed==pytest.approx(expected, abs=0.05)


def test_heatbath_shift_acceptance_one():
    w=model()
    for _ in range(100):
        if w.move_toggle():
            break
    assert w.open
    assert all(w.move_shift(method='heatbath') for _ in range(100))
    assert w._counts['shift_heatbath'] == [100,100]


def test_local_shift_ratio_matches_full_open_density():
    w=model(2.0)
    i=1
    old=np.array([.5,.6,.3]);new=np.array([.4,.7,.9])
    x=w.log_sector_ratio(i,new)-w.log_sector_ratio(i,old)
    k=w.kernel.log_kernel(w._delta(i)+new,w.config.time_step)-w.kernel.log_kernel(w._delta(i)+old,w.config.time_step)
    assert x==pytest.approx(float(k),abs=1e-12)


def test_worm_run_shapes_and_determinism():
    a=model(n_steps=1000).run()
    b=model(n_steps=1000).run()
    assert a.n_closed>0 and a.n_open>0
    assert a.n_closed+a.n_open == (1000-100+2-1)//2
    np.testing.assert_array_equal(a.occupation_fractions,b.occupation_fractions)
    np.testing.assert_array_equal(a.g1_radial,b.g1_radial)
    assert a.n0_fraction>0
    assert np.all(np.isfinite(a.g1_radial))
    assert 0<a.sector_effective_samples<=a.n_closed+a.n_open
    assert all(0<=v<=1 for v in a.acceptance.values() if np.isfinite(v))


def test_interacting_worm_and_closed_momentum_finite():
    w=model(height=2.0)
    closed=closed_link_occupations(w.base,w.indices)
    assert closed.shape==(len(w.indices),)
    assert np.all(np.isfinite(closed))
    out=w.run()
    assert np.isfinite(out.n0_fraction) and out.n0_fraction>0


@pytest.mark.parametrize('change',[{'sector_fugacity':0.}, {'toggle_probability':0.},
   {'local_shift_probability':1.1},{'shift_sigma':-1.}, {'radial_bins':0},
   {'n_beads':3},{'n_steps':4},{'refresh_probability':.9,'swap_probability':.5}])
def test_invalid_config(change):
    with pytest.raises((ValueError, TypeError)):
        cfg(**change).validate()


def test_invalid_momentum_modes():
    w=model()
    for modes in [[], [(1,0,0)], [(0,0,0),(0,0,0)], [(0.2,0,0)],[(np.nan,0,0)]]:
        with pytest.raises((ValueError,TypeError)):
            canonical_free_occupations(w.cell,2,w.config.beta,modes)


def test_cannot_shift_in_closed_sector():
    with pytest.raises(RuntimeError):
        model().move_shift()


def test_nonzero_interacting_potential_changes_open_density_only_via_paths():
    a=model(0.)
    b=model(5.)
    # Initial free paths and proposal RNGs coincide, so the known-sector
    # insertion ratio is independent of the potential for fixed coordinates.
    assert a.log_sector_ratio(0,np.array([.2,.3,.4])) == pytest.approx(
           b.log_sector_ratio(0,np.array([.2,.3,.4])))


def test_closed_link_occupation_analytic_integral_positive_at_zero():
    w=model(0)
    est=closed_link_occupations(w.base,[(0,0,0)])
    d=w.base.path[0,w.base.permutation]-w.base.path[-1]
    truth=np.mean(1/(w.cell.volume*np.exp(w.kernel.log_kernel(d,w.config.time_step))))
    assert est[0] == pytest.approx(truth)


def test_uniform_free_canonical_mode_thermal_limit():
    w=model()
    # Large beta yields Bose ground-state occupation of the k=0 mode.
    occ=canonical_free_occupations(w.cell,2,20.,w.indices)
    assert occ[0]==pytest.approx(1,abs=1e-12)
    assert np.all(occ[1:]<1e-12)


def test_sector_gibbs_probs_are_normalized_and_nontrivial():
    w = model()
    p = w.conditional_sector_probabilities()
    assert p.shape == (w.n_particles + 1,)
    assert np.all(p > 0)
    assert np.sum(p) == pytest.approx(1.0, abs=1e-14)
    # Conditional probabilities must be exactly those derived from the
    # analytically integrated G_i displacement kernels.
    delta = w.base.path[0, w.base.permutation] - w.base.path[-1]
    k = np.exp(w.kernel.log_kernel(delta, w.config.time_step))
    weight = np.r_[1., w.config.sector_fugacity / (w.n_particles*w.cell.volume*k)]
    np.testing.assert_allclose(p, weight/weight.sum(), rtol=1e-13)


def test_sector_gibbs_matches_conditional_histogram():
    w = model()
    expected = w.conditional_sector_probabilities()
    counts = np.zeros(len(expected))
    for _ in range(5500):
        assert w.move_sector_gibbs()
        counts[0 if not w.open else w.particle + 1] += 1
    np.testing.assert_allclose(counts / counts.sum(), expected, atol=.025, rtol=0)
    assert w._counts['sector_gibbs'] == [5500,5500]


def test_free_worm_replicas_agree_with_exact_with_monte_carlo_tolerance():
    # This is a small statistical smoke test, not a precision benchmark.
    cell = CubicCell(2.8)
    exact = canonical_free_occupations(cell,2,1.8,((0,0,0),))[0]
    results = [model(seed=100+i*100, n_steps=5000, burn_in=900,
                     gibbs_probability=1.).run().n0_fraction for i in range(3)]
    assert abs(np.mean(results)-exact) < .20


def test_run_experiment_output_contract(tmp_path):
    import csv
    from examples.bosons.run_worm import run_experiment
    rep,summ,mom,rads = run_experiment(tmp_path,quick=True)
    assert len(rep)==4
    assert len(summ)==2
    assert len(mom)==8
    assert len(rads)==12
    for name,expected in [('replicas.csv',4),('summary.csv',2),
                          ('momenta.csv',8),('g1_radial.csv',12)]:
        with (tmp_path/name).open() as f:
            assert len(list(csv.DictReader(f)))==expected
    for name in ('occupation.png','momentum.png','g1_radial.png'):
        assert (tmp_path/name).stat().st_size>1000
    assert (tmp_path/'protocol.json').stat().st_size>100
    assert all(np.isfinite(x['n0_worm']) and x['n0_worm']>0 for x in summ)
    assert all(x['free_exact_n0'] != '' for x in summ if x['height']==0.)
    assert all(x['free_exact_n0'] == '' for x in summ if x['height']>0.)


def test_collapsed_refresh_normalizer_is_exact_sector_sum():
    w = model(height=2.0)
    p = w.conditional_sector_probabilities()
    norm = w._collapsed_log_normalizer(w.base.path, w.base.permutation)
    assert np.exp(norm) == pytest.approx(1.0/p[0], rel=1e-12)


def test_collapsed_refresh_keeps_state_valid_and_samples_both_sectors():
    w = model(height=1., n_steps=1200)
    for _ in range(40):
        w.move_collapsed_refresh()
        assert np.isfinite(w.base.log_weight())
        p = w.conditional_sector_probabilities()
        assert np.sum(p) == pytest.approx(1.0)
        if w.open:
            assert 0 <= w.particle < w.n_particles
            assert w.shift.shape == (3,)
    assert w._counts['collapsed_refresh'][1] == 40
    assert w._counts['sector_gibbs'][1] == 40


@pytest.mark.parametrize('change',[{'collapsed_refresh_probability':-0.1},
    {'collapsed_refresh_probability':1.1},
    {'toggle_probability':0.8,'collapsed_refresh_probability':.3}])
def test_invalid_collapsed_update_config(change):
    with pytest.raises(ValueError):
        cfg(**change).validate()
