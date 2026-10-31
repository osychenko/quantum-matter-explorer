"""Periodic continuum free open-path projector and pure coherence tests."""

import csv
import json
from dataclasses import replace

import numpy as np
import pytest

from qmc.cells import CubicCell, OrthorhombicCell
from qmc.continuum_path import (
    ContinuumPathConfig, ContinuumPathResult, FreeOpenPathSampler,
    PeriodicCosineOrbital, PeriodicFreePropagator, measure_free_coherence,
)
from examples.bosons.run_continuum_path import ExperimentConfig, run_experiment


def reference():
    cell = CubicCell(4)
    return PeriodicFreePropagator(cell, kinetic_coefficient=0.5), PeriodicCosineOrbital(cell, .7)


@pytest.mark.parametrize("D", [0.2, 0.5, 1.4])
def test_periodic_heat_kernel_positive_even_and_invariant(D):
    kernel = PeriodicFreePropagator(OrthorhombicCell((3., 4., 5.)), D)
    disp = np.array([[.7, -.8, .9], [1.3, 1.9, -2.]] )
    log = kernel.log_kernel(disp, .11)
    np.testing.assert_allclose(log, kernel.log_kernel(-disp, .11), atol=2e-12)
    np.testing.assert_allclose(log, kernel.log_kernel(disp + [6, -4, 15], .11), atol=2e-12)
    assert np.all(np.isfinite(log))


@pytest.mark.parametrize("time", [.015, .15, 1.0, 4.])
def test_heat_kernel_numeric_one_axis_normalization(time):
    kernel, _ = reference()
    grid = np.linspace(0, 4, 1001)[:-1]
    vals = np.zeros((len(grid), 3))
    vals[:, 0] = grid
    # Other two coordinates are zero, so K3d(x,0,0)=K1d(x)*K1d(0)^2.
    base = np.exp(kernel.log_kernel([0,0,0], time))
    raw = np.exp(kernel.log_kernel(vals, time))
    # Integrate ratio: mean K3d(x,0,0)/K3d(0,0) is a normalized
    # one-dimensional reference if multiplied by K1d(0); independently
    # compute K1d(0) with the same image formulation.
    sigma = np.sqrt(2*kernel.kinetic_coefficient*time)
    shifts = np.arange(-18, 19)*4
    k1_zero = np.exp(-.5*(shifts/sigma)**2).sum()/(np.sqrt(2*np.pi)*sigma)
    integral = (raw/base).mean()*4*k1_zero
    np.testing.assert_allclose(integral, 1.0, atol=1e-7)


def test_short_time_gaussian_reference_and_step_variance():
    kernel,_=reference()
    d=np.array([.06,-.13,.02]);t=0.012
    expected = -.5*np.sum(d*d)/(2*.5*t) - 1.5*np.log(4*np.pi*.5*t)
    assert kernel.log_kernel(d,t)==pytest.approx(expected,abs=1e-12)
    rng=np.random.default_rng(9)
    x=kernel.sample_step(np.zeros((12000,3)),.015,rng)
    dx=kernel.cell.wrap(x + 2) - 2
    assert np.var(dx[:,0])==pytest.approx(.015,rel=.04)


def test_semigroup_by_fourier_mode():
    kernel,orbital=reference()
    points=np.zeros((15000,3))
    points[:,0]=np.linspace(0,4,len(points),endpoint=False)
    t=0.8
    K=np.exp(kernel.log_kernel(points,t))
    # Uniform x integral of K(x,0,0) cos(k x) / integral K
    projected=(K*np.cos(orbital.k*points[:,0])).sum()/K.sum()
    assert projected==pytest.approx(np.exp(-.5*orbital.k**2*t),abs=2e-9)


@pytest.mark.parametrize("amplitude", [0., .25, -.6, .8])
def test_exact_radial_and_zero_momentum(amplitude):
    cell=CubicCell(4); orbital=PeriodicCosineOrbital(cell,amplitude,mode=2)
    r=np.array([0., .3, .8]); g,n=orbital.exact_radial(.4,r,.3)
    assert g[0]==pytest.approx(1)
    assert n==pytest.approx(1/(1+.5*amplitude**2*np.exp(-2*.3*orbital.k**2*.4)))
    if amplitude==0:
        np.testing.assert_array_equal(g,np.ones(3))
        assert n==1.
    assert 0<n<=1
    shifts=np.array([[0,0,0],[2.,0,0]])
    f,_=orbital.exact_coherence(.4,shifts,.3)
    assert f[0]==pytest.approx(1)


def test_sampler_reproducible_refresh_reptation_and_path_shape():
    kernel,orbital=reference()
    config=ContinuumPathConfig(n_links=10,n_steps=150,burn_in=40,thin=4,seed=903)
    a=FreeOpenPathSampler(kernel,orbital,config).run()
    b=FreeOpenPathSampler(kernel,orbital,config).run()
    np.testing.assert_array_equal(a.paths,b.paths)
    assert a.paths.shape==(28,11,3)
    assert 0<a.refresh_acceptance<=1
    assert 0<a.reptation_acceptance<=1
    assert np.all(a.paths>=0) and np.all(a.paths<4)
    assert a.beta==pytest.approx(config.beta)


def test_reptation_and_refresh_mh_log_balance_exact():
    kernel,orbital=reference()
    cfg=ContinuumPathConfig(n_links=4,n_steps=10,burn_in=2)
    sampler=FreeOpenPathSampler(kernel,orbital,cfg)
    rng=np.random.default_rng(23)
    path=sampler._free_path()
    y=kernel.sample_step(path[-1],cfg.time_step,rng)
    shifted=np.concatenate([path[1:],y[None]],axis=0)
    # Forward reptation q=K(last,y); reverse q=K(x1,x0).
    target=lambda x: (sampler._end_log(x)+sum(float(kernel.log_kernel(x[i+1]-x[i],cfg.time_step)) for i in range(cfg.n_links)))
    logqforward=float(kernel.log_kernel(y-path[-1],cfg.time_step))
    logqbackward=float(kernel.log_kernel(path[0]-path[1],cfg.time_step))
    mh=target(shifted)-target(path)+logqbackward-logqforward
    assert mh==pytest.approx(sampler._end_log(shifted)-sampler._end_log(path),abs=2e-12)
    candidate=sampler._free_path()
    logqratio = (target(path)-sampler._end_log(path))-(target(candidate)-sampler._end_log(candidate))
    mh=target(candidate)-target(path)+logqratio
    assert mh==pytest.approx(sampler._end_log(candidate)-sampler._end_log(path),abs=2e-12)


def test_sampler_nonuniform_orbital_endpoint_density():
    kernel,orbital=reference()
    cfg=ContinuumPathConfig(n_links=10,n_steps=9000,burn_in=1500,thin=3,
                            refresh_probability=.55,seed=202)
    result=FreeOpenPathSampler(kernel,orbital,cfg).run()
    mean_cos=np.cos(orbital.k*result.paths[:,0,orbital.axis]).mean()
    a=orbital.amplitude; decay=np.exp(-kernel.kinetic_coefficient*orbital.k**2*2*cfg.beta)
    expected=a*(1+decay)/(2*(1+.5*a*a*decay))
    assert mean_cos==pytest.approx(expected,abs=.05)


def test_endpoint_pure_coherence_matches_exact_nonuniform_trial():
    kernel,orbital=reference()
    cfg=ContinuumPathConfig(time_step=.12,n_links=12,n_steps=18000,burn_in=3500,
                            thin=3,refresh_probability=.65,seed=93)
    run=FreeOpenPathSampler(kernel,orbital,cfg).run()
    radii=np.linspace(0,2,5)
    result=measure_free_coherence(kernel,run,radii,draws=4,seed=94)
    exact,n0=orbital.exact_radial(cfg.beta,radii,kernel.kinetic_coefficient)
    np.testing.assert_allclose(result.g1,exact,atol=.07)
    assert result.n0_fraction==pytest.approx(n0,abs=.07)
    assert result.g1[0]==1.


def test_free_uniform_trial_matches_exact_with_sampling_error():
    kernel,_=reference()
    orbital=PeriodicCosineOrbital(kernel.cell,0)
    cfg=ContinuumPathConfig(time_step=.1,n_links=18,n_steps=6500,burn_in=1000,thin=2,seed=82)
    run=FreeOpenPathSampler(kernel,orbital,cfg).run()
    assert run.refresh_acceptance==1
    assert run.reptation_acceptance==1
    estimated=measure_free_coherence(kernel,run,[0,.5,1,1.7,2],draws=5,seed=83)
    np.testing.assert_allclose(estimated.g1,1,atol=.065)
    assert estimated.n0_fraction==pytest.approx(1,abs=.065)


def test_midpoint_and_endpoint_estimators_share_free_expectation():
    kernel,orbital=reference()
    cfg=ContinuumPathConfig(time_step=.2,n_links=18,n_steps=12000,burn_in=2000,thin=2,seed=11)
    run=FreeOpenPathSampler(kernel,orbital,cfg).run()
    exact,n0=orbital.exact_radial(cfg.beta,[0,.3],kernel.kinetic_coefficient)
    for method in ['endpoint','midpoint']:
        res=measure_free_coherence(kernel,run,[0,.3],draws=4,seed=42,estimator=method)
        assert res.g1[1]==pytest.approx(exact[1],abs=.075)
        assert res.n0_fraction==pytest.approx(n0,abs=.075)


@pytest.mark.parametrize('invalid', [
    dict(n_links=1),dict(n_links=5),dict(n_links=-2),dict(n_steps=0),
    dict(burn_in=6000),dict(thin=0),dict(refresh_probability=-.1),
    dict(refresh_probability=1.1),dict(seed=-1),dict(time_step=0),
])
def test_reject_bad_path_config(invalid):
    with pytest.raises(ValueError):
        replace(ContinuumPathConfig(),**invalid).validate()


@pytest.mark.parametrize('invalid', [-1,1,np.nan,np.inf])
def test_reject_bad_orbital_amplitude(invalid):
    with pytest.raises(ValueError):PeriodicCosineOrbital(CubicCell(4),invalid)


def test_reject_bad_kernel_and_measure_inputs():
    kernel,orbital=reference()
    with pytest.raises(ValueError):kernel.log_kernel([0,0,0],0)
    with pytest.raises(ValueError):kernel.log_kernel([0,0],.1)
    with pytest.raises(TypeError):PeriodicFreePropagator(object())
    with pytest.raises(ValueError):FreeOpenPathSampler(kernel,PeriodicCosineOrbital(CubicCell(5),.2))
    cfg=ContinuumPathConfig(n_steps=8,burn_in=2,n_links=4)
    r=FreeOpenPathSampler(kernel,orbital,cfg).run()
    with pytest.raises(ValueError):measure_free_coherence(kernel,r,[0,2.1])
    with pytest.raises(ValueError):measure_free_coherence(kernel,r,[1,2])
    with pytest.raises(ValueError):measure_free_coherence(kernel,r,[0,1],draws=0)
    with pytest.raises(ValueError):measure_free_coherence(kernel,r,[0,1],estimator='bad')
    with pytest.raises(ValueError):measure_free_coherence(kernel,
        ContinuumPathResult(r.paths[:,:-1],0.5,0.5,r.beta),[0,1])


def test_quick_demo_output_and_determinism(tmp_path):
    cfg=ExperimentConfig(half_projection_times=(.75,),n_links=(8,),
         n_steps=200,burn_in=70,thin=5,replicas=2,n_radii=4,direction_draws=2)
    _,summary,g=run_experiment(cfg,tmp_path/'one')
    assert len(summary)==2
    assert len(g)==2*cfg.n_radii
    for name in ['replicas.csv','summary.csv','g1.csv','protocol.json',
                 'occupation_convergence.png','g1_comparison.png']:
        assert (tmp_path/'one'/name).stat().st_size>0
    with (tmp_path/'one'/'replicas.csv').open() as f:
        assert len(list(csv.DictReader(f)))==4
    with (tmp_path/'one'/'protocol.json').open() as f:
        assert json.load(f)['model'].startswith('3D periodic free')
    _,again,_=run_experiment(cfg,tmp_path/'two')
    assert summary==again


def test_invalid_experiment_configuration(tmp_path):
    for kwargs in (dict(replicas=1),dict(n_links=(3,)),dict(n_steps=10,burn_in=11),
                   dict(amplitudes=(1.2,)),dict(half_projection_times=(0.,))):
        with pytest.raises(ValueError):
            run_experiment(replace(ExperimentConfig(),**kwargs),tmp_path/'bad')
