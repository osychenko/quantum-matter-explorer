"""Patch 24: exact MTM balance, regression, rare-observation diagnostics."""
from dataclasses import replace

import numpy as np
import pytest

from qmc.bosons import SoftCorePairPotential
from qmc.cells import CubicCell
from qmc.rare_worm import (RareWormConfig, RareEventContinuumWorm,
                           contribution_concentration, independent_multiple_try)
from qmc.worm_audit import audit_worm_chain

CELL=CubicCell(2.8)


def make(*, seed=123, n=1, height=2., **overrides):
    cfg=replace(RareWormConfig(n_links=4,time_step=.6,n_steps=240,burn_in=40,
                               thin=2,seed=seed, multiple_try_probability=1.,
                               multiple_try_candidates=3),**overrides)
    return RareEventContinuumWorm(CELL,SoftCorePairPotential(CELL,height=height,cutoff=1.2),n,cfg)


def test_candidate_validation():
    for k in (0,-1,1.5,True,65):
        with pytest.raises(ValueError):
            independent_multiple_try(0,lambda x: 0.,lambda: 0,np.random.default_rng(1),candidates=k)
    for p in (-.1,1.01,float('nan')):
        with pytest.raises(ValueError):
            replace(RareWormConfig(),multiple_try_probability=p).validate()
    for k in (0,65,True):
        with pytest.raises(ValueError):
            replace(RareWormConfig(),multiple_try_candidates=k).validate()


def test_one_candidate_exact_independence_mh_ratio():
    draw=lambda: 1
    rng=np.random.default_rng(15)
    _,accepted,ratio=independent_multiple_try(0,lambda x: 2*x,draw,rng,candidates=1)
    assert ratio == pytest.approx(2.)
    assert accepted


def test_constant_weights_accept_all_k():
    rng=np.random.default_rng(13)
    for k in (1,2,4,8):
        for _ in range(8):
            x,a,lr=independent_multiple_try(0,lambda _: 16.,lambda: int(rng.integers(5)),rng,candidates=k)
            assert a and lr==pytest.approx(0.,abs=1e-12)
            assert 0<=x<5


def test_discrete_toy_stationary_distribution_multiple_try():
    # Reference proposal is nonuniform. Target must be q_i R_i / sum(q R).
    q=np.array([.52,.28,.20]);R=np.array([.1,2.,9.]);exact=q*R/(q@R)
    rng=np.random.default_rng(42)
    def draw(): return int(rng.choice(3,p=q))
    def w(x): return float(np.log(R[x]))
    x=0;counts=np.zeros(3,int)
    for j in range(16000):
        x,_,_=independent_multiple_try(x,w,draw,rng,candidates=4)
        if j>=1000:counts[x]+=1
    np.testing.assert_allclose(counts/counts.sum(),exact,atol=.035)


def test_log_weights_stay_stable_when_extreme():
    rng=np.random.default_rng(11)
    out,a,lr=independent_multiple_try(0,lambda x: 1200.+150.*x,lambda: 1,rng,candidates=4)
    assert out==1 and a and np.isfinite(lr)


def test_invalid_log_importance_rejected():
    rng=np.random.default_rng(1)
    with pytest.raises(ValueError):
        independent_multiple_try(0,lambda x: float('nan'),lambda: 1,rng)
    with pytest.raises(ValueError):
        independent_multiple_try(0,lambda x: float('inf') if x else 0.,lambda: 1,rng)


def test_closed_sector_rejects_mtm():
    s=make()
    with pytest.raises(RuntimeError):s.move_multiple_try()


def test_free_open_mtm_is_reversible_and_accepts():
    s=make(height=0.,n=0)
    s.open=True
    for _ in range(20):
        assert s.move_multiple_try()
        assert s._counts['multiple_try'][1]==s._counts['multiple_try'][0]
        assert s.current_log_relative_weight()==pytest.approx(np.log(s.config.sector_fugacity))


def test_deterministic_state_and_counts():
    a=make(seed=67);b=make(seed=67)
    for _ in range(125):a.step();b.step()
    assert a.open==b.open and a.cut==b.cut and a.length==b.length
    np.testing.assert_allclose(a.host,b.host)
    np.testing.assert_allclose(a.extra,b.extra)
    assert a._counts==b._counts


def test_zero_probability_baseline_never_invokes_mtm():
    s=make(multiple_try_probability=0.)
    s.open=True
    for _ in range(10):s.move_global_refresh()
    assert s._counts['multiple_try']==[0,0]


def test_legacy_audit_works_on_new_sampler():
    s=make(n=0,height=0.,n_steps=440,burn_in=80,thin=2,seed=16)
    result=audit_worm_chain(s,n_blocks=4)
    assert result.green.shape==(3,2)
    assert result.n_measurements==180
    assert np.all(np.isfinite(result.green))


def test_contribution_concentration():
    a=np.ones((100,2,3)); a[0,0,0]=99.
    conc=contribution_concentration(a,tail_fraction=.01)
    assert conc.shape==(2,3)
    assert conc[0,0]==pytest.approx(.5)
    assert conc[1,1]==pytest.approx(.01)
    assert np.all(contribution_concentration(np.zeros_like(a))==0)
    for bad in (np.zeros((3,2,2)), np.ones((4,3)), np.full((8,2,2),np.nan)):
        with pytest.raises(ValueError):contribution_concentration(bad)


def test_validating_sampler_configuration():
    s=make()
    assert s.config.multiple_try_probability==1.
    with pytest.raises(TypeError):
        RareEventContinuumWorm(CELL,SoftCorePairPotential(CELL,height=2,cutoff=1.2),1,config=None if False else object())


def test_independent_free_vacuum_zero_mode_is_exact():
    from qmc.rare_worm import independent_reference_green
    s=make(n=0,height=0.,seed=31,sector_fugacity=.65)
    r=independent_reference_green(s,draws_per_interval=24)
    assert r.green.shape==(3,2)
    np.testing.assert_allclose(r.green[:,0],1.,atol=1e-12)
    np.testing.assert_allclose(r.stderr[:,0],0.,atol=1e-12)


def test_independent_ref_is_reproducible():
    from qmc.rare_worm import independent_reference_green
    a,b=make(seed=85),make(seed=85)
    x=independent_reference_green(a,draws_per_interval=15)
    y=independent_reference_green(b,draws_per_interval=15)
    np.testing.assert_allclose(x.green,y.green)
    np.testing.assert_allclose(x.stderr,y.stderr)
    assert x.samples_per_interval==15
    with pytest.raises(ValueError):independent_reference_green(make(),draws_per_interval=1)
