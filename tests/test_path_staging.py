"""Patch 16: exact free bridges, MH reversibility, statistical diagnostics."""

from dataclasses import replace

import numpy as np
import pytest

from qmc.bosons import SoftCoreJastrowTrial, SoftCorePairPotential
from qmc.cells import CubicCell, OrthorhombicCell
from qmc.continuum_path import PeriodicFreePropagator
from qmc.path_staging import (
    AutocorrelationEstimate, StagedInteractingPath, StagedPathConfig,
    autocorrelation_estimate, periodic_bridge,
)
from qmc.two_boson_reference import TwoBosonRelativeReference


class UniformTrial:
    def log_psi(self, R):
        return np.zeros(len(R))
    def grad_log_psi(self, R):
        return np.zeros_like(R)
    def laplacian_log_psi(self, R):
        return np.zeros(len(R))


class ConstantPotential:
    def __init__(self, value=0.):
        self.value = value
    def __call__(self, R):
        return np.full(len(R), self.value)


def sampler(*, method="mixed", length=8, seed=21, interacting=False):
    cell = CubicCell(3.5)
    cfg = StagedPathConfig(n_links=length, time_step=0.2, n_steps=42,
                           burn_in=8, thin=2, seed=seed,
                           bridge_lengths=(2, 4), bridge_method=method)
    if interacting:
        potential = SoftCorePairPotential(cell, height=4., cutoff=1.5)
        trial = SoftCoreJastrowTrial(cell, amplitude=.5, cutoff=1.5)
    else:
        potential, trial = ConstantPotential(), UniformTrial()
    return StagedInteractingPath(cell, potential, trial, 2, cfg)


@pytest.mark.parametrize("method", ["staging", "bisection"])
@pytest.mark.parametrize("length", [2, 3, 4, 8])
def test_bridge_endpoints_wrapping_shapes(method, length):
    cell = OrthorhombicCell((3., 4., 5.))
    kernel = PeriodicFreePropagator(cell, .5)
    rng = np.random.default_rng(91)
    start = np.array([[.2,3.8,1.],[2.95,.01,4.98]])
    end = np.array([[2.9,.1,1.],[.1,3.9,.02]])
    bridge = periodic_bridge(kernel,rng,start,end,length,.15,method=method)
    assert bridge.shape == (length+1,2,3)
    np.testing.assert_allclose(bridge[0],start)
    np.testing.assert_allclose(bridge[-1],end)
    assert np.all(np.isfinite(bridge))
    assert np.all(bridge >= 0)
    assert np.all(bridge < cell.lengths)


@pytest.mark.parametrize("method", ["staging", "bisection"])
def test_bridge_brownian_midpoint_mean_and_variance(method):
    cell = CubicCell(100.)
    kernel = PeriodicFreePropagator(cell, .5)
    rng = np.random.default_rng(990)
    samples = 3000
    base = np.full((samples,3), 50.)
    output = periodic_bridge(kernel,rng,base,base,4,.1,method=method)
    delta = output[2,:,0]-50
    assert abs(delta.mean()) < 0.026
    # Var = (2D*tau)* j*(M-j)/M = .1, for j=2,M=4,D=.5.
    assert delta.var() == pytest.approx(.1,rel=.095)


def test_bridge_reproducibility_and_invalid_input():
    cell = CubicCell(3.5)
    kernel = PeriodicFreePropagator(cell)
    p = np.ones((1,3))
    a = periodic_bridge(kernel,np.random.default_rng(3),p,p,4,.1)
    b = periodic_bridge(kernel,np.random.default_rng(3),p,p,4,.1)
    np.testing.assert_array_equal(a,b)
    with pytest.raises(ValueError): periodic_bridge(kernel,np.random.default_rng(1),p,p,1,.1)
    with pytest.raises(ValueError): periodic_bridge(kernel,np.random.default_rng(1),p,p,2,0)
    with pytest.raises(ValueError): periodic_bridge(kernel,np.random.default_rng(1),p,p,2,.1,method="unknown")
    with pytest.raises(ValueError): periodic_bridge(kernel,np.random.default_rng(1),p,np.zeros((2,3)),2,.1)


@pytest.mark.parametrize("bad", [
    {"bridge_probability":-0.1}, {"bridge_probability":1.1},
    {"bridge_probability":float('nan')}, {"bridge_lengths":()},
    {"bridge_lengths":(1,)}, {"bridge_lengths":(2, 9)},
    {"bridge_lengths":(2, 2)}, {"bridge_lengths":(2.5,)},
    {"bridge_method":"none"},
])
def test_bad_staging_config(bad):
    with pytest.raises(ValueError):
        replace(StagedPathConfig(n_links=8), **bad).validate()


def test_bridge_moves_always_accept_zero_or_constant_potential():
    s = sampler()
    for method in ("staging","bisection"):
        for length in (2,4,6):
            for start in range(s.config.n_links-length+1):
                assert s.move_bridge(method=method,length=length,start=start,particle=1)
    assert s._counts["staging"][0]==s._counts["staging"][1]
    assert s._counts["bisection"][0]==s._counts["bisection"][1]
    s2 = sampler(interacting=False)
    s2.potential = ConstantPotential(1.7)
    for _ in range(15):
        assert s2.move_bridge()


@pytest.mark.parametrize("method", ["staging","bisection"])
def test_bridge_log_accept_ratio_is_exact_interaction_difference(method):
    s = sampler(interacting=True)
    old = s.path.copy()
    record = []
    original = s._accept
    try:
        s._accept = lambda kind, proposal, ratio: record.append((kind, proposal, ratio))
        s.move_bridge(length=4,start=1,particle=0,method=method)
    finally:
        s._accept = original
    assert len(record)==1
    kind, proposed, delta = record[0]
    assert kind == method
    np.testing.assert_array_equal(proposed[:2],old[:2])
    np.testing.assert_array_equal(proposed[5:],old[5:])
    np.testing.assert_array_equal(proposed[:,1],old[:,1])
    interaction_delta = -s.config.time_step*(np.sum(s.potential(proposed[2:5]))-
                                             np.sum(s.potential(old[2:5])))
    assert delta == pytest.approx(interaction_delta,abs=1e-12)
    # Conditional proposal ratio exactly removes the free-link kinetic delta.
    log_free_diff = float(np.sum(s.kernel.log_kernel(
        proposed[2:6,0]-proposed[1:5,0],s.config.time_step))-
        np.sum(s.kernel.log_kernel(old[2:6,0]-old[1:5,0],s.config.time_step)))
    assert s.log_weight(proposed)-s.log_weight(old)-log_free_diff == pytest.approx(delta,abs=1e-10)


def test_staged_run_and_step_counts_reproducible():
    a,b=sampler(seed=77,interacting=True),sampler(seed=77,interacting=True)
    result1,result2=a.run(),b.run()
    np.testing.assert_array_equal(result1.midpoint_potential,result2.midpoint_potential)
    np.testing.assert_array_equal(result1.final_path,result2.final_path)
    assert sum(result1.attempted.values()) == a.config.n_steps
    assert result1.attempted["staging"] > 0
    assert result1.attempted["bisection"] > 0
    assert len(result1.midpoint_potential) == 17
    assert np.all(np.isfinite(result1.midpoint_potential))


def test_autocorrelation_constant_and_reproducible():
    d=autocorrelation_estimate(np.full(30,3.))
    assert isinstance(d,AutocorrelationEstimate)
    assert d.mean == 3
    assert d.effective_samples == 30
    assert d.stderr == 0
    assert d.tau_integrated == 0.5
    arr=np.random.default_rng(2).normal(size=3000)
    x,y=autocorrelation_estimate(arr),autocorrelation_estimate(arr)
    assert x==y
    assert .5<=x.tau_integrated<2.5
    assert 0 < x.effective_samples<=len(arr)


def test_autocorrelation_correlated_ar1():
    rng=np.random.default_rng(42)
    x=np.zeros(6000)
    for i in range(1,len(x)):
        x[i]=.83*x[i-1]+rng.normal()
    a=autocorrelation_estimate(x)
    b=autocorrelation_estimate(rng.normal(size=6000))
    assert a.tau_integrated > 2.5
    assert a.effective_samples < b.effective_samples/3


@pytest.mark.parametrize("bad", [np.ones((2,2)),[1,2,3], [1,np.nan,2,3]])
def test_autocorrelation_invalid(bad):
    with pytest.raises(ValueError):autocorrelation_estimate(bad)


def test_reference_interacting_hermiticity_and_basis_convergence():
    kwargs=dict(cell_length=3.5,kinetic_coefficient=.5,height=4.0,
                cutoff=1.5,jastrow_amplitude=.5,n_grid=24)
    a=TwoBosonRelativeReference(**kwargs,k_max=1)
    b=TwoBosonRelativeReference(**kwargs,k_max=2)
    np.testing.assert_allclose(b.hamiltonian,b.hamiltonian.conj().T,rtol=0,atol=1e-12)
    assert len(a.modes)==27 and len(b.modes)==125
    x,y=a.measure(.8),b.measure(.8)
    assert b.measure(np.inf).energy <= a.measure(np.inf).energy + 1e-9  # Rayleigh-Ritz for ground-state energies
    assert abs(x.midpoint_potential-y.midpoint_potential)<.015
    assert .05<y.midpoint_potential<.3


def test_reference_exact_free_constant_state():
    r=TwoBosonRelativeReference(cell_length=3.5,kinetic_coefficient=.5,
                                height=0,cutoff=1.5,jastrow_amplitude=0,k_max=1,n_grid=16)
    for beta in (0.,.7,np.inf):
        out=r.measure(beta)
        assert out.energy==pytest.approx(0,abs=1e-12)
        assert out.midpoint_potential==pytest.approx(0,abs=1e-12)


def test_reference_invalid_parameters():
    params=dict(cell_length=3.5,kinetic_coefficient=.5,height=4.0,cutoff=1.5,
                jastrow_amplitude=.5,k_max=1,n_grid=16)
    for extra in ({"cell_length":-3},{"cutoff":2.0},{"height":-1},
                  {"k_max":0},{"k_max":3,"n_grid":8},{"jastrow_amplitude":-1}):
        with pytest.raises(ValueError):TwoBosonRelativeReference(**(params|extra))


def test_quick_experiment_outputs(tmp_path):
    from examples.bosons.run_path_staging import run_experiment
    replicas,summary,references=run_experiment(tmp_path,quick=True)
    assert len(replicas)==8 and len(summary)==4 and len(references)==2
    assert all(np.isfinite(r["midpoint_v"]) for r in replicas)
    assert all(0<r["effective_measurements"]<=r["n_measurements"] for r in replicas)
    for f in ("replicas.csv","summary.csv","fourier_reference.csv","protocol.json",
              "midpoint_convergence.png","effective_samples.png"):
        assert (tmp_path/f).stat().st_size>10
