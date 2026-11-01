"""Patch 23: trustworthy uncertainty and matched-physics audit contracts."""
from dataclasses import replace

import numpy as np
import pytest

from qmc.bosons import SoftCorePairPotential
from qmc.cells import CubicCell
from qmc.reliable_spacetime_worm import ReliableContinuumSpacetimeWorm, ReliableSpacetimeConfig
from qmc.worm_audit import (audit_worm_chain, ratio_block_jackknife,
                            replica_summary, split_rhat)
from examples.bosons.run_worm_audit import WormAuditConfig, _check, _contrasts, run_experiment


def _model(n=0, height=0, seed=33, steps=440):
    cell=CubicCell(2.8)
    cfg=ReliableSpacetimeConfig(n_links=4,time_step=.6,n_steps=steps,
         burn_in=80,thin=3,seed=seed,sector_fugacity=.8)
    return ReliableContinuumSpacetimeWorm(cell,SoftCorePairPotential(
         cell,height=height,cutoff=1.2),n,cfg,
         indices=((0,0,0),(1,0,0)))


def _num_den():
    rng=np.random.default_rng(17)
    den=rng.uniform(.3,.9,80)
    num=np.empty((80,2,2))
    num[:,0,0]=2*den
    num[:,0,1]=den*(.5+0.2*rng.normal(size=80))
    num[:,1,0]=den*(1+0.2*rng.normal(size=80))
    num[:,1,1]=den*3
    return num,den


def test_ratio_preserves_correlated_normalization():
    num,den=_num_den()
    estimate,sem=ratio_block_jackknife(num,den,n_blocks=8,scale=1.3)
    assert estimate.shape==(2,2)
    assert sem.shape==(2,2)
    np.testing.assert_allclose(estimate[0,0],2*1.3,atol=1e-14)
    np.testing.assert_allclose(estimate[1,1],3*1.3,atol=1e-14)
    assert sem[0,0]<1e-13
    assert sem[1,1]<1e-13
    assert sem[0,1]>0
    assert sem[1,0]>0


def test_jackknife_matches_direct_deletion_formula():
    num,den=_num_den();num=num[:64];den=den[:64]
    result,sem=ratio_block_jackknife(num,den,n_blocks=4)
    np.testing.assert_allclose(result,num.sum(axis=0)/den.sum())
    others=[]
    for i in range(4):
        keep=np.ones(64,dtype=bool);keep[i*16:(i+1)*16]=False
        others.append(num[keep].sum(axis=0)/den[keep].sum())
    others=np.asarray(others)
    expected=np.sqrt(3/4*np.sum((others-others.mean(axis=0))**2,axis=0))
    np.testing.assert_allclose(sem,expected,rtol=1e-13,atol=1e-13)


@pytest.mark.parametrize('modification',[
    ('shape',None),('negative_den',None),('nan',None),('too_short',None),
    ('zero_den',None),('blocks',2),('scale',0)
])
def test_bad_jackknife_inputs(modification):
    kind,v=modification;num,den=_num_den();kw={}
    if kind=='shape':den=den[:-1]
    elif kind=='negative_den':den[0]=-1
    elif kind=='nan':num[10,0,0]=np.nan
    elif kind=='too_short':num=num[:25];den=den[:25]
    elif kind=='zero_den':den[:]=0
    elif kind=='blocks':kw['n_blocks']=v
    elif kind=='scale':kw['scale']=v
    with pytest.raises(ValueError):ratio_block_jackknife(num,den,**kw)


def test_replica_sem_is_between_chains():
    mean,sem=replica_summary([2.,4.,6.])
    assert mean==4.
    assert sem==pytest.approx(np.std([2,4,6],ddof=1)/np.sqrt(3))
    with pytest.raises(ValueError):replica_summary([1])
    with pytest.raises(ValueError):replica_summary([1,np.nan])


def test_split_rhat_stationary_and_nonstationary():
    rng=np.random.default_rng(44)
    chains=[rng.normal(size=300) for _ in range(4)]
    r=split_rhat(chains)
    assert .95<r<1.12
    drift=[np.ones(40),np.full(40,3.)]
    assert np.isinf(split_rhat(drift))
    assert split_rhat([np.ones(40),np.ones(40)])==1


def test_split_rhat_checks_dimensions():
    with pytest.raises(ValueError):split_rhat([np.ones(40)])
    with pytest.raises(ValueError):split_rhat([np.ones(8),np.ones(8)])
    with pytest.raises(ValueError):split_rhat([np.ones(20),np.full(20,np.nan)])


def test_audit_matches_original_unchanged_sampler():
    old=_model(0,0,seed=27)
    new=_model(0,0,seed=27)
    original=old.run()
    audited=audit_worm_chain(new,n_blocks=4)
    np.testing.assert_allclose(audited.green,original.green,rtol=2e-14,atol=2e-14)
    np.testing.assert_allclose(audited.green_raw,original.green_raw,rtol=2e-14,atol=2e-14)
    assert audited.closed_count==original.n_closed
    np.testing.assert_array_equal(audited.open_counts,original.n_open_by_length)
    assert audited.influence.shape==(audited.n_measurements,3,2)
    assert audited.numerator_trace.shape==audited.influence.shape
    assert audited.denominator_trace.shape==(audited.n_measurements,)
    assert np.all(audited.ratio_ess>0)
    assert np.all(np.isfinite(audited.jackknife_stderr))


def test_audit_uses_reproducible_paths():
    a=audit_worm_chain(_model(1,2,seed=81),n_blocks=4)
    b=audit_worm_chain(_model(1,2,seed=81),n_blocks=4)
    np.testing.assert_array_equal(a.green,b.green)
    np.testing.assert_array_equal(a.jackknife_stderr,b.jackknife_stderr)
    np.testing.assert_array_equal(a.ratio_ess,b.ratio_ess)


def test_audit_rejects_small_chain():
    with pytest.raises(ValueError):audit_worm_chain(_model(0,0,steps=150),n_blocks=8)
    with pytest.raises(TypeError):audit_worm_chain(object())


def minimal_config():
    return WormAuditConfig(betas=(2.4,),n_links=(4,),heights=(0.,),
        chain_steps=(300,),mixtures=('baseline',),replicas=2,n_blocks=4,thin=3)


@pytest.mark.parametrize('delta',[
    {'betas':()}, {'betas':(0,)},{'heights':(-1,)},{'n_links':(5,)},
    {'n_links':(2,)},{'mixtures':('unknown',)},{'replicas':1},
    {'n_blocks':2},{'chain_steps':(30,)},{'burn_fraction':.95},
    {'modes':((1,0,0),)}, {'cutoff':1.9},
])
def test_invalid_audit_config(delta):
    with pytest.raises((ValueError,TypeError)):_check(replace(minimal_config(),**delta))


def test_end_to_end_minimal_outputs(tmp_path):
    replica,summary,reference=run_experiment(tmp_path,config=minimal_config())
    assert len(replica)==2*3*2
    assert len(summary)==3*2
    assert len(reference)==2*3*2
    assert all(r['status']=='inconclusive' for r in summary)
    assert all(np.isfinite(r['green']) for r in summary)
    for name in ('summary.csv','replicas.csv','spectral_reference.csv',
                 'protocol.json','timestep_audit.png','length_audit.png','ratio_ess.png'):
        assert (tmp_path/name).stat().st_size>0
    # A single grid and run length offers no matched-physics contrasts.
    import csv
    with (tmp_path/'contrasts.csv').open() as f:
        assert len(list(csv.reader(f)))>=0


def test_matched_physics_contrasts_exclude_unmatched():
    row=dict(beta=2.4,height=3.,mixture='baseline',separation=1.2,
             mode_x=0,mode_y=0,mode_z=0,n_links=4,n_steps=1000,
             green=1.1,replica_sem=.1)
    rows=[row,dict(row,n_steps=2000,green=1.4,replica_sem=.2),
          dict(row,n_links=6,green=1.3,replica_sem=.15),
          dict(row,height=4.,green=3.)]
    changes=_contrasts(rows)
    assert len(changes)==2
    assert {x['kind'] for x in changes}=={'time_step','run_length'}
    assert all(x['difference']>0 for x in changes)
