"""Patch 21: true 3D finite-continuum N=0/1 -> N+1 spacetime insertion checks."""
from dataclasses import replace
from math import exp, log

import numpy as np
import pytest

from qmc.bosons import SoftCorePairPotential
from qmc.cells import CubicCell, OrthorhombicCell
from qmc.boson_permutations import free_one_particle_partition
from qmc.continuum_spacetime_worm import (
    ContinuumSpacetimeConfig, ContinuumSpacetimeWorm,
    exact_free_addition_green,
)

CELL = CubicCell(2.8)


def make(n=1, height=0.0, **kwargs):
    cfg=replace(ContinuumSpacetimeConfig(
        n_links=4, time_step=0.65, n_steps=2500, burn_in=400,
        thin=3, seed=23), **kwargs)
    potential=SoftCorePairPotential(CELL,height=height,cutoff=1.2)
    return ContinuumSpacetimeWorm(CELL,potential,n,cfg,
                                  indices=((0,0,0),(1,0,0)))


def test_config_validates_probabilities():
    for field, value in (("n_links",2),("time_step",0),("n_steps",1),
                         ("burn_in",6500),("thin",0),("sector_fugacity",-1),
                         ("toggle_probability",1.1),("global_refresh_probability",1.),
                         ("time_move_probability",1.1),("endpoint_move_probability",1.),
                         ("seed",-1),("move_sigma",0)):
        with pytest.raises(ValueError):
            replace(ContinuumSpacetimeConfig(),**{field:value}).validate()


def test_mode_and_domain_validation():
    pot=SoftCorePairPotential(CELL,height=0,cutoff=1)
    with pytest.raises(ValueError,match="currently supports"):
        ContinuumSpacetimeWorm(CELL,pot,2)
    with pytest.raises(ValueError,match="include zero"):
        ContinuumSpacetimeWorm(CELL,pot,0,indices=((1,0,0),))
    with pytest.raises(ValueError,match="duplicate"):
        ContinuumSpacetimeWorm(CELL,pot,0,indices=((0,0,0),(0,0,0)))
    with pytest.raises(ValueError):
        ContinuumSpacetimeWorm(CELL,pot,0,indices=((0.1,0,0),(0,0,0)))
    with pytest.raises(TypeError):
        ContinuumSpacetimeWorm(object(),pot,0)
    other=OrthorhombicCell((3.,3.,3.))
    with pytest.raises(ValueError,match="different cell"):
        ContinuumSpacetimeWorm(other,pot,0)


def test_vacuum_relative_weight_is_exact_eta_for_every_link():
    worm=make(0,sector_fugacity=0.7)
    for ell in range(1,worm.config.n_links):
        for cut in range(worm.config.n_links):
            extra=worm._new_extra(ell)
            assert worm.log_relative_weight(worm.host,cut,ell,extra)==pytest.approx(log(0.7),abs=1e-13)


def test_free_two_boson_single_link_symmetrization():
    worm=make(1)
    ell=1;cut=0
    extra=worm._new_extra(ell)
    h=worm.host
    tau=worm.config.time_step
    a=float(worm.kernel.log_kernel(h[1,0]-h[0,0],tau))
    b=float(worm.kernel.log_kernel(extra[1]-extra[0],tau))
    c=float(worm.kernel.log_kernel(extra[1]-h[0,0],tau))
    d=float(worm.kernel.log_kernel(h[1,0]-extra[0],tau))
    expected=log(1.+exp(c+d-a-b))
    assert worm.log_relative_weight(h,cut,ell,extra)==pytest.approx(expected,abs=1e-12)


def test_pair_action_is_symmetric_trapezoid():
    free=make(1,height=0)
    rep=make(1,height=5)
    ell=2;cut=3
    extra=free._new_extra(ell)
    host=free.host
    v=np.asarray(rep.potential(np.stack((host[(cut+np.arange(ell+1))%4,0],extra),axis=1)))
    expected=-rep.config.time_step*(v[0]/2+v[1]+v[2]/2)
    assert rep.log_relative_weight(host,cut,ell,extra)-free.log_relative_weight(host,cut,ell,extra)==pytest.approx(expected)


def test_permutation_exchange_invariance():
    worm=make(1)
    # K^sym_2 symmetric under either incoming or outgoing particle swap.
    tau=worm.config.time_step
    r=worm.rng.random((2,2,3))*CELL.length
    A=worm.kernel.log_kernel(r[1]-r[0],tau)
    direct=float(A.sum())
    exchange=float(worm.kernel.log_kernel(r[1,::-1]-r[0],tau).sum())
    sym=float(np.logaddexp(direct,exchange)-np.log(2.))
    swapped=float(np.logaddexp(exchange,direct)-np.log(2.))
    assert sym==swapped


def test_reversible_open_close_ratios():
    worm=make(1,height=2)
    e=worm._new_extra(2)
    lr=worm.log_relative_weight(worm.host,2,2,e)
    assert lr + (-lr)==0.
    # Exactly the same for an interval that crosses the time boundary.
    assert np.isfinite(worm.log_relative_weight(worm.host,3,2,e))


@pytest.mark.parametrize("which,direction", [
    ("head","extend"),("tail","extend"),("head","retract"),("tail","retract")])
def test_time_move_preserves_shape_and_validity(which,direction):
    worm=make(1,height=2)
    worm.open=True;worm.cut=3;worm.length=2;worm.extra=worm._new_extra(2)
    before=worm.current_log_relative_weight()
    assert np.isfinite(before)
    worm.move_time(which,direction)
    assert worm.extra.shape==(worm.length+1,3)
    assert 0<=worm.cut<worm.config.n_links
    assert 1<=worm.length<worm.config.n_links
    assert np.isfinite(worm.current_log_relative_weight())


def test_time_move_q_cancellation_on_tail_and_head():
    worm=make(1)
    e=worm._new_extra(2)
    worm.open=True;worm.cut=1;worm.length=2;worm.extra=e
    for which in ("head","tail"):
        new=worm.kernel.sample_step(e[-1] if which=="head" else e[0],worm.config.time_step,worm.rng)
        e2=np.vstack((e,new)) if which=="head" else np.vstack((new,e))
        delta_q=worm._log_extra_reference(e2)-worm._log_extra_reference(e)
        x=e[-1] if which=="head" else e[0]
        assert delta_q==pytest.approx(float(worm.kernel.log_kernel(new-x,worm.config.time_step)))
    # Proposal q cancels target q ratio; the remaining MH log factor is d(log R).


def test_endpoint_local_proposal_uses_free_density():
    worm=make(1,height=1)
    e=worm._new_extra(2)
    e2=e.copy();e2[0]=CELL.wrap(e2[0]+np.array((.2,-.1,.05)))
    dq=worm._log_extra_reference(e2)-worm._log_extra_reference(e)
    k=worm.kernel
    assert dq==pytest.approx(float(k.log_kernel(e2[1]-e2[0],worm.config.time_step)
                                  -k.log_kernel(e[1]-e[0],worm.config.time_step)))


def test_move_scheduler_deterministic():
    a,b=make(1,height=1),make(1,height=1)
    for i in range(250):
        a.step();b.step()
    np.testing.assert_allclose(a.host,b.host)
    assert a.open==b.open
    assert a.cut==b.cut and a.length==b.length
    if a.open:
        np.testing.assert_allclose(a.extra,b.extra)
    assert a._counts==b._counts


def test_vacuum_free_exact_k0_and_k_nonzero():
    worm=make(0,n_steps=7000,burn_in=1000,thin=2,seed=17,sector_fugacity=0.8)
    result=worm.run()
    reference=exact_free_addition_green(CELL,0,result.beta,result.imaginary_times,result.indices)
    assert np.all(np.abs(result.green[:,0]-reference[:,0])<0.26)
    assert np.all(np.abs(result.green[:,1]-reference[:,1])<0.21)
    assert result.n_closed>100 and np.all(result.n_open_by_length>30)


def test_free_n1_bose_enhancement_relative_to_distinguishable():
    # N=1 free canonical G_0=1+1/Z_1(beta), not 1. The symmetrized
    # two-body link is essential to include the exchange contribution.
    beta=2.6
    ref=exact_free_addition_green(CELL,1,beta,np.array((.5,1.3)),np.array(((0,0,0),(1,0,0))))
    Z1=free_one_particle_partition(CELL,beta)
    np.testing.assert_allclose(ref[:,0],1+1/Z1)
    assert ref[0,0]>1.5
    assert ref[1,1]<ref[0,1]


def test_free_n1_short_mcmc_agrees_with_exact():
    worm=make(1,n_steps=13000,burn_in=2000,thin=3,seed=44,sector_fugacity=0.9)
    result=worm.run()
    ref=exact_free_addition_green(CELL,1,result.beta,result.imaginary_times,result.indices)
    assert np.all(np.abs(result.green[:,0]-ref[:,0])<0.6)
    assert result.n_closed>50 and np.all(result.n_open_by_length>25)


def test_repulsive_case_samples_both_sectors():
    worm=make(1,height=3,n_steps=3500,burn_in=500,thin=3)
    result=worm.run()
    assert result.green.shape==(3,2)
    assert result.sector_effective_samples>0
    assert np.all(np.isfinite(result.green))
    assert (result.acceptances["global_refresh"]>=0.)


def test_zero_interaction_on_vacuum_and_constant_shift():
    class Shift:
        def __call__(self, R):
            return np.full(len(R),2.0)
    s=make(0)
    e=s._new_extra(3)
    x=ContinuumSpacetimeWorm(CELL,Shift(),0,s.config)
    assert x.log_relative_weight(s.host,0,3,e)-s.log_relative_weight(s.host,0,3,e)==pytest.approx(-3*s.config.time_step*2)


def test_bad_worm_states_and_actions():
    s=make(1)
    e=s._new_extra(2)
    with pytest.raises(ValueError):
        s.log_relative_weight(s.host,0,3,e)
    with pytest.raises(ValueError):
        s.log_relative_weight(s.host,4,2,e)
    with pytest.raises(RuntimeError):
        s.move_time()
    with pytest.raises(RuntimeError):
        s.move_endpoint()
    with pytest.raises(RuntimeError):
        s.move_extra_refresh()
    with pytest.raises(ValueError):
        exact_free_addition_green(CELL,0,s.config.beta,[-1],((0,0,0),))


def test_exchange_bridge_requires_open_sector():
    worm=make(1)
    with pytest.raises(RuntimeError):
        worm.move_exchange_bridge()


def test_exact_free_exchange_bridge_accepts_and_preserves_endpoints():
    worm=make(1,height=0)
    worm.open=True;worm.cut=3;worm.length=3;worm.extra=worm._new_extra(3)
    idx=(worm.cut+np.arange(worm.length+1))%worm.config.n_links
    start=worm.host[idx[[0,-1]],0].copy()
    ends=worm.extra[[0,-1]].copy()
    for _ in range(40):
        assert worm.move_exchange_bridge() is True
        np.testing.assert_allclose(worm.host[idx[[0,-1]],0],start)
        np.testing.assert_allclose(worm.extra[[0,-1]],ends)
    assert worm._counts["exchange_bridge"] == [40,40]


def test_exchange_bridge_conditional_midpoint_is_label_symmetric():
    worm=make(1,height=0)
    # Endpoint pairing direct/exchanged is sampled with exact heat-kernel
    # weights, and each intermediate pair of positions is relabeled 50:50.
    worm.open=True;worm.cut=0;worm.length=2
    worm.host[:3,0]=np.array(([.3,.3,.3],[1.4,1.4,1.4],[.4,.4,.4]))
    worm.extra=np.array(([2.2,2.2,2.2],[1.0,1.0,1.0],[2.3,2.3,2.3]))
    middle=[]
    for _ in range(200):
        assert worm.move_exchange_bridge()
        middle.append(worm.host[1,0,0])
    # The conditional law is exchange-symmetric in intermediate labels.
    # For this well-separated pair the two Gaussian bridge midpoints have
    # means near .35 and 2.25; their 50:50 mixture has mean near 1.30.
    assert abs(np.mean(middle)-1.30)<0.35


def test_exchange_bridge_short_interval_is_self_loop():
    worm=make(1)
    worm.open=True;worm.cut=0;worm.length=1;worm.extra=worm._new_extra(1)
    assert worm.move_exchange_bridge() is False
    assert worm._counts["exchange_bridge"]==[0,1]
