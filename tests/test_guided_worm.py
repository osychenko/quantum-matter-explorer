"""Patch 25: guided proposals, normalized reweighting, detailed balance and limits."""
from dataclasses import replace
from math import exp

import numpy as np
import pytest

from qmc.bosons import SoftCorePairPotential
from qmc.cells import CubicCell
from qmc.continuum_spacetime_worm import exact_free_addition_green
from qmc.guided_worm import (
    GuidedExtraProposal, GuidedContinuumWorm, GuidedWormConfig,
    independent_guided_green,
)
from qmc.rare_worm import RareEventContinuumWorm, RareWormConfig

CELL = CubicCell(2.8)


def make(*, height=3., seed=14, **kwargs):
    config = replace(GuidedWormConfig(n_links=4,time_step=.6,n_steps=600,
                     burn_in=120,thin=3,seed=seed,
                     sector_fugacity=.9, guided_fraction=.65),**kwargs)
    return GuidedContinuumWorm(CELL,SoftCorePairPotential(CELL,height=height,cutoff=1.2),
                               1,config,indices=((0,0,0),(1,0,0)))


def test_guided_config_invalid():
    for field, value in [('guided_move_probability',-1),('guided_move_probability',1.1),
                         ('guided_fraction',1),('guided_fraction',-0.1),
                         ('drift_strength',-1),('drift_strength',float('nan')),
                         ('max_drift',0),('max_drift',float('inf')),('initial_tilt_strength',-1)]:
        with pytest.raises(ValueError):
            replace(GuidedWormConfig(),**{field:value}).validate()


def test_requires_soft_core_and_matching_cell():
    a=make()
    with pytest.raises(TypeError):
        GuidedExtraProposal(a.kernel,object())
    with pytest.raises(ValueError):
        GuidedExtraProposal(a.kernel,SoftCorePairPotential(CubicCell(3),height=3,cutoff=1.2))
    with pytest.raises(ValueError):
        GuidedContinuumWorm(CELL,a.potential,0,a.config)
    with pytest.raises(TypeError):
        GuidedContinuumWorm(CELL,a.potential,1,RareWormConfig())


def test_drift_points_away_from_repulsion_and_is_capped():
    a=make(drift_strength=4,max_drift=.2)
    host=np.array([1.,1.,1.])
    previous=host+np.array([.3,0,0])
    drift=a.guide._drift(previous,host,.6)
    assert drift[0]>0
    assert np.linalg.norm(drift)==pytest.approx(.2)
    assert np.all(a.guide._drift(host+np.array([1.3,0,0]),host,.6)==0)
    assert np.all(a.guide._drift(host,host,.6)==0)


def test_zero_drift_guide_identical_to_free_density():
    a=make(drift_strength=0,initial_tilt_strength=0)
    e,_=a.guide.sample(a.host,1,2,.6,a.rng)
    assert a.guide.log_guide_over_free(a.host,1,e,.6)==pytest.approx(0)
    assert a.guide.log_reference_over_mixture(a.host,1,e,.6)==pytest.approx(0)
    free=make(height=0)
    e,_=free.guide.sample(free.host,0,3,.6,free.rng)
    assert free.guide.log_reference_over_mixture(free.host,0,e,.6)==0


def test_guided_density_difference_matches_manual_kernel_ratio():
    a=make(guided_fraction=.75,drift_strength=1,max_drift=.4)
    h=a.host.copy()
    e,_=a.guide.sample(h,2,3,.6,a.rng)
    manual=a.guide._initial_log_guide_over_free(e[0],h[2,0],.6)
    for j in range(3):
        dr=e[j+1]-e[j]
        d=a.guide._drift(e[j],h[(2+j+1)%4,0],.6)
        manual+=float(a.kernel.log_kernel(dr-d,.6)-a.kernel.log_kernel(dr,.6))
    assert a.guide.log_guide_over_free(h,2,e,.6)==pytest.approx(manual)
    expected=-np.logaddexp(np.log(.25),np.log(.75)+manual)
    assert a.guide.log_reference_over_mixture(h,2,e,.6)==pytest.approx(expected)


def test_probability_mixture_bounds_importance_ratio():
    a=make(guided_fraction=.8)
    for _ in range(100):
        e,_=a.guide.sample(a.host,0,3,.6,a.rng)
        lr=a.guide.log_reference_over_mixture(a.host,0,e,.6)
        assert np.isfinite(lr) and lr <= -np.log(.2)+1e-12


def test_initial_tilt_normalizer_matches_uniform_monte_carlo():
    a=make(initial_tilt_strength=2)
    proposal=a.guide
    uniform=CELL.to_cartesian(a.rng.random((30000,3)))
    host=a.host[1,0]
    dr=CELL.minimum_image(uniform-host)
    rr=np.sum(dr*dr,axis=-1)
    shape=np.maximum(0,1-rr/proposal.potential.cutoff**2)**3
    weights=np.exp(-2*.5*.6*proposal.potential.height*shape)
    exact=np.exp(proposal._initial_log_normalizer(.6))
    assert abs(weights.mean()-exact)<5*weights.std(ddof=1)/np.sqrt(len(weights))+.002


def test_guide_integrates_to_one_one_link_by_reweighting():
    # For fixed host and first extra bead, E_qguide[K_free/K_guide]=1.
    # Use 3000 independent guide increments with a deliberately large drift.
    a=make(guided_fraction=.0,drift_strength=1.0,max_drift=.6)
    start=a.host[0,0]+np.array((.3,0,0))
    hostnext=a.host[1,0]
    drift=a.guide._drift(start,hostnext,.6)
    assert np.linalg.norm(drift)>0
    x=a.kernel.sample_step(start+drift,.6,a.rng)
    assert x.shape==(3,)
    generated=a.kernel.sample_step(np.broadcast_to(start+drift,(4000,3)),.6,a.rng)
    ratios=np.exp(a.kernel.log_kernel(generated-start,.6)-
                  a.kernel.log_kernel(generated-start-drift,.6))
    assert abs(ratios.mean()-1) < 5*ratios.std(ddof=1)/np.sqrt(len(ratios))+.02


def test_move_log_ratio_reverse_and_no_bias():
    a=make()
    a.open=True
    a.cut,a.length,a.extra=a._draw_ghost()
    old=(a.host,a.cut,a.length,a.extra)
    new=a._draw_guided_state()
    forward=a._corrected_log_weight(new)-a._corrected_log_weight(old)
    reverse=a._corrected_log_weight(old)-a._corrected_log_weight(new)
    assert forward==-reverse
    assert np.isfinite(forward)
    # Demonstrate the correction matters for at least one sampled candidate.
    assert abs(a.guide.log_reference_over_mixture(new[0],new[1],new[3],.6))>1e-8


def test_free_limit_independent_guided_matches_free():
    a=make(height=0,seed=12)
    result=independent_guided_green(a,draws_per_interval=350)
    exact=exact_free_addition_green(CELL,1,2.4,result.imaginary_times,result.indices)
    assert np.all(np.abs(result.green[:,0]-exact[:,0]) < .55)
    assert np.all(np.isfinite(result.stderr))
    assert np.all(result.weight_ess>0)


def test_importance_identity_qmix_to_qfree_for_general_state():
    a=make(guided_fraction=.7)
    for _ in range(3):
        h=a._new_host()
        e,_=a.guide.sample(h,1,2,.6,a.rng)
        log_mix=-a.guide.log_reference_over_mixture(h,1,e,.6)
        r=a.log_relative_weight(h,1,2,e)
        # Desired G density/qmix is R * q_free/qmix, not just R.
        assert a._corrected_log_weight((h,1,2,e))==pytest.approx(r-log_mix)


def test_periodic_shift_invariance_of_guide_density():
    a=make()
    e,_=a.guide.sample(a.host,2,3,.6,a.rng)
    delta=a.guide.log_reference_over_mixture(a.host,2,e,.6)
    # Shift a particle at any bead by a lattice vector: physical path unchanged.
    shifted=e.copy();shifted[1]+=np.array([2.8,0,0]);shifted[3]+=np.array([0,-2.8,0])
    assert a.guide.log_reference_over_mixture(a.host,2,shifted,.6)==pytest.approx(delta,abs=1e-11)


def test_guide_refresh_changes_state_and_is_repeatable():
    a=make(seed=23)
    b=make(seed=23)
    for s in (a,b):
        s.open=True
        s.cut,s.length,s.extra=s._draw_ghost()
    for _ in range(10):
        assert a.move_guided_refresh()==b.move_guided_refresh()
        np.testing.assert_allclose(a.host,b.host)
        np.testing.assert_allclose(a.extra,b.extra)
        assert a.cut==b.cut and a.length==b.length
    assert a._counts['guided_refresh'][1]==10


def test_closed_sector_global_refresh_unchanged():
    a=make()
    assert a.open is False
    assert a.move_global_refresh()
    assert a._counts['guided_refresh']==[0,0]


def test_zero_guided_probability_recovers_existing_scheduler():
    a=make(guided_move_probability=0,seed=19)
    b=RareEventContinuumWorm(CELL,a.potential,1,
                 replace(RareWormConfig(),n_links=4,time_step=.6,n_steps=600,
                         burn_in=120,thin=3,seed=19,sector_fugacity=.9,
                         multiple_try_probability=0),indices=((0,0,0),(1,0,0)))
    for _ in range(80):
        a.step(); b.step()
    # The configs are otherwise identical and the guide branch consumes no RNG.
    np.testing.assert_allclose(a.host,b.host)
    assert a.open==b.open and a.length==b.length
    np.testing.assert_allclose(a.extra,b.extra)


def test_independent_estimator_reproducible_and_input_validation():
    a,b=make(seed=39),make(seed=39)
    x=independent_guided_green(a,draws_per_interval=30)
    y=independent_guided_green(b,draws_per_interval=30)
    np.testing.assert_array_equal(x.green,y.green)
    np.testing.assert_array_equal(x.stderr,y.stderr)
    with pytest.raises(ValueError): independent_guided_green(a,draws_per_interval=3)
    with pytest.raises(TypeError): independent_guided_green(object())


def test_run_and_audit_contract():
    from qmc.worm_audit import audit_worm_chain
    a=make(n_steps=350,burn_in=50,thin=3)
    result=audit_worm_chain(a,n_blocks=4)
    assert result.green.shape==(3,2)
    assert np.isfinite(result.green).all()
    assert 'guided_refresh' in result.acceptance
    assert result.n_measurements==100


def test_reference_bounds_for_soft_core():
    a=make(guided_fraction=.8)
    r=independent_guided_green(a,draws_per_interval=50)
    assert np.all(r.top_one_pct>=0) and np.all(r.top_one_pct<=1)
    assert np.all(r.weight_ess>=1) and np.all(r.weight_ess<=50+1e-8)


def test_initial_tilt_without_drift_still_requires_reweighting():
    a=make(drift_strength=0,initial_tilt_strength=2,guided_fraction=.8)
    e,_=a.guide.sample(a.host,2,3,.6,a.rng)
    correction=a.guide.log_reference_over_mixture(a.host,2,e,.6)
    assert correction==pytest.approx(-np.logaddexp(np.log(.2),
        np.log(.8)+a.guide._initial_log_guide_over_free(e[0],a.host[2,0],.6)))
    assert np.isfinite(correction)
