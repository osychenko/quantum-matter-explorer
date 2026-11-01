"""Patch 20 — normalized independent-endpoint lattice spacetime worm tests."""
from dataclasses import replace

import numpy as np
import pytest

from qmc.pure_projector import BoseHubbardRing
from qmc.spacetime_worm import ExactSpacetimeWorm, SpacetimeWormConfig
from examples.bosons.run_spacetime_worm import lattice_free_occupation, run_experiment


def small(onsite=2., seed=2026, **kwargs):
    cfg=replace(SpacetimeWormConfig(n_steps=1500,burn_in=250,thin=2,seed=seed),**kwargs)
    return ExactSpacetimeWorm(BoseHubbardRing(n_sites=3,n_bosons=2,hopping=1.,onsite_u=onsite),cfg)


def test_thermal_green_symmetries_and_positivity():
    s=small()
    assert s.f.shape == (s.M,s.L,s.L)
    assert np.all(s.f[1:] > 0)
    np.testing.assert_allclose(s.f,np.swapaxes(s.f,1,2),rtol=1e-13,atol=1e-13)
    for row in range(s.L):
        np.testing.assert_allclose(s.f[1:,row,:], np.roll(s.f[1:,0,:],row,axis=1),rtol=1e-12,atol=1e-12)
    assert 0 < s.exact_open_fraction() < 1


def test_exact_closed_partition_matches_transfer_trace():
    s=small()
    assert s.z==pytest.approx(np.trace(np.linalg.matrix_power(s.tn,s.M)),abs=1e-12)
    weights=s.config.fugacity*s.f.sum()/((s.M-1)*s.L**2)
    assert s.exact_open_fraction()==pytest.approx(weights/(s.z+weights))


def test_exact_green_independent_spectral_form():
    s=small()
    sidx,i,j=2,1,0
    N=s.n_projector;P=s.plus_projector
    beta=s.config.beta;t=sidx*s.config.time_step
    # Independent direct spectral propagators, separate from power recursion.
    def expH(projector,t):
        v=projector.eigenvectors
        return (v*np.exp(-t*projector.energies)[None,:])@v.T
    a=s.create[i].T
    b=s.create[j]
    numer=np.trace(expH(N,beta-t)@a@expH(P,t)@b)
    den=np.trace(expH(N,beta))
    assert s.exact_green()[sidx,i,j]==pytest.approx(numer/den,rel=2e-12)


def test_canonical_free_zero_mode_independent_of_eigenvectors():
    s=small(0.0)
    expected=lattice_free_occupation(2,3,1.,s.config.beta)
    assert s.exact_zero_momentum_fraction()==pytest.approx(expected,rel=2e-12)
    assert 0<=expected<=1


def test_equal_time_green_limit_matches_one_plus_mode_occupation():
    s=small()
    # The insertion order is a_i a_j^dagger; at tau=0 this equals I+rho.
    obdm=s.exact_obdm()
    np.testing.assert_allclose(obdm,obdm.T,atol=1e-12)
    assert np.trace(obdm)==pytest.approx(s.N,abs=1e-12)
    n0=np.sum(obdm)/s.L
    # Use the exact a a^dagger trace at zero time, rather than extrapolating dt.
    full=np.zeros((s.L,s.L))
    for i in range(s.L):
        for j in range(s.L):
            full[i,j]=np.trace(s.npowers[s.M]@s.create[i].T@s.create[j])/s.z
    np.testing.assert_allclose(full,np.eye(s.L)+obdm.T,rtol=1e-12,atol=1e-12)
    assert full.sum()/s.L==pytest.approx(1+n0)


def test_gamma_path_z_is_closed_and_has_fixed_N():
    s=small()
    p=s.sample_path()
    assert p.open is False and p.extra is None
    assert p.background.shape==(s.M+1,s.L)
    assert np.all(p.background.sum(axis=1)==s.N)
    np.testing.assert_array_equal(p.background[0],p.background[-1])


def test_g_path_has_real_particle_number_jumps():
    s=small()
    for _ in range(3000):
        s.step()
        if s.open: break
    assert s.open
    for _ in range(12):
        p=s.sample_path()
        assert p.open and p.tail_time==s.tail_time and p.head_time==s.head_time
        assert p.extra.shape==(s.separation+1,s.L)
        assert p.background.shape==(s.M-s.separation+1,s.L)
        assert np.all(p.extra.sum(axis=1)==s.N+1)
        assert np.all(p.background.sum(axis=1)==s.N)
        e_tail=np.eye(s.L,dtype=int)[p.tail_site]
        e_head=np.eye(s.L,dtype=int)[p.head_site]
        np.testing.assert_array_equal(p.extra[0],p.background[-1]+e_tail)
        np.testing.assert_array_equal(p.background[0],p.extra[-1]-e_head)


def test_time_endpoints_independent_and_reversible_ratios(monkeypatch):
    s=small()
    s.open=True;s.tail_time=1;s.separation=3;s.head_site=1;s.tail_site=2
    # Force accept and capture MH ratios without changing the physics code.
    calls=[]
    original=s._accept
    def accept(kind,ratio):
        calls.append((kind,ratio))
        return True
    monkeypatch.setattr(s,'_accept',accept)
    old_head=s.head_time
    target=s.f[4,1,2]/s.f[3,1,2]
    assert s.move_head_time(+1)
    assert s.separation==4 and s.head_time==(old_head+1)%s.M
    assert calls[-1][1]==pytest.approx(target)
    assert s.move_head_time(-1)
    assert calls[-1][1]==pytest.approx(1/target)
    old_head=s.head_time
    target=s.f[2,1,2]/s.f[3,1,2]
    assert s.move_tail_time(+1)
    assert s.separation==2 and s.head_time==old_head
    assert calls[-1][1]==pytest.approx(target)
    assert s.move_tail_time(-1)
    assert calls[-1][1]==pytest.approx(1/target)
    monkeypatch.setattr(s,'_accept',original)


def test_endpoints_site_moves_have_reciprocal_ratios(monkeypatch):
    s=small();s.open=True;s.tail_time=1;s.separation=3;s.head_site=1;s.tail_site=2
    calls=[]
    monkeypatch.setattr(s,'_accept',lambda kind,ratio: calls.append((kind,ratio)) or True)
    ratio=s.f[3,0,2]/s.f[3,1,2]
    s.move_head_site(0);assert calls[-1][1]==pytest.approx(ratio)
    s.move_head_site(1);assert calls[-1][1]==pytest.approx(1/ratio)
    ratio=s.f[3,1,0]/s.f[3,1,2]
    s.move_tail_site(0);assert calls[-1][1]==pytest.approx(ratio)
    s.move_tail_site(2);assert calls[-1][1]==pytest.approx(1/ratio)


def test_reversible_open_close_proposal_ratios(monkeypatch):
    s=small();s.open=True;s.tail_time=0;s.separation=3;s.head_site=1;s.tail_site=2
    ratio=s.config.fugacity*s.f[3,1,2]/s.z
    calls=[]
    monkeypatch.setattr(s,'_accept',lambda kind,r: calls.append((kind,r)) or False)
    s.move_toggle()
    assert calls[-1][0]=='close' and calls[-1][1]==pytest.approx(1/ratio)
    assert ratio>0


def test_invalid_endpoint_moves():
    s=small()
    with pytest.raises(RuntimeError):s.move_head_time()
    with pytest.raises(RuntimeError):s.move_tail_site()
    s.open=True;s.tail_time=0;s.separation=1;s.head_site=0;s.tail_site=0
    assert not s.move_tail_time(1)
    assert not s.move_head_time(-1)
    with pytest.raises(ValueError):s.move_head_time(2)
    with pytest.raises(ValueError):s.move_head_site(0)
    with pytest.raises(ValueError):s.move_tail_site(-1)


def test_step_scheduler_attempts_same_toggle_frequency_in_Z_and_G(monkeypatch):
    # Regression for a normalization error found during development: Z->G
    # must not be attempted more often than G->Z without an MH correction.
    s=small()
    s.config=replace(s.config,toggle_probability=.35)
    class RNG:
        def random(self): return .5  # outside toggle range
    monkeypatch.setattr(s,'rng',RNG())
    calls=[]
    monkeypatch.setattr(s,'move_toggle',lambda :calls.append('toggle'))
    s.open=False;s.step()
    assert calls==[]


def test_reproducible_chain_and_normalization():
    cfg=dict(n_steps=28000,burn_in=4000,thin=3,seed=21321)
    a=small(**cfg);b=small(**cfg)
    ra=a.run();rb=b.run()
    np.testing.assert_array_equal(ra.green,rb.green)
    assert ra.observed_open_fraction==rb.observed_open_fraction
    assert ra.n_closed>200 and ra.n_open>200
    assert abs(ra.observed_open_fraction-ra.exact_open_fraction)<.04
    # Translationally averaged G(d,tau) from worm sectors.
    gref=np.stack([a.exact_green()[s,:,0] for s in range(1,a.M)])
    assert abs(ra.green[0,0]-gref[0,0])<.28


def test_free_tight_and_interacting_case_distinct():
    free=small(0.); interacting=small(4.)
    assert free.exact_zero_momentum_fraction()>interacting.exact_zero_momentum_fraction()
    assert free.exact_open_fraction()!=pytest.approx(interacting.exact_open_fraction())


@pytest.mark.parametrize('bad',[
    dict(time_step=0.), dict(time_step=float('nan')),dict(n_slices=3),
    dict(n_steps=4),dict(burn_in=2000),dict(thin=0),
    dict(fugacity=0),dict(fugacity=float('inf')),dict(seed=-1),
    dict(toggle_probability=0),dict(space_move_probability=0),
    dict(toggle_probability=.4,time_move_probability=.4,space_move_probability=.4),
])
def test_bad_config(bad):
    with pytest.raises(ValueError):small(**bad)


def test_wrong_model_and_config_type():
    with pytest.raises(TypeError):ExactSpacetimeWorm('no')
    with pytest.raises(TypeError):ExactSpacetimeWorm(BoseHubbardRing(),config='no')


def test_run_quick_files_and_reproducibility(tmp_path):
    r,s,g=run_experiment(tmp_path/'one',quick=True)
    assert len(r)==4 and len(s)==2 and len(g)==2*(6-1)
    for name in ('replicas.csv','summary.csv','green.csv','protocol.json',
                 'green_vs_time.png','sector_normalization.png'):
        assert (tmp_path/'one'/name).stat().st_size>100
    assert all(np.isfinite(z['gk0']) for z in g)
    assert all(0<z['observed_open_fraction']<1 for z in s)
    assert all(abs(z['observed_open_fraction']-z['exact_open_fraction'])<.1 for z in s)
    r2,s2,g2=run_experiment(tmp_path/'two',quick=True)
    assert r==r2 and s==s2 and g==g2


def test_free_occupation_validation():
    with pytest.raises(ValueError):lattice_free_occupation(0,3,1.,2.)
    with pytest.raises(ValueError):lattice_free_occupation(1,2,1.,2.)
    with pytest.raises(ValueError):lattice_free_occupation(1,3,0.,2.)
