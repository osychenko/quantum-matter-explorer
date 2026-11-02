from dataclasses import replace
import numpy as np
import pytest
from qmc.cells import CubicCell
from qmc.bosons import SoftCorePairPotential
from qmc.replica_exchange_worm import ReplicaExchangeConfig
from qmc.equal_cpu_worm import plan_budget,run_validation,summarize_validation,make_sampler

CELL=CubicCell(2.8)
POT=SoftCorePairPotential(CELL,height=2.,cutoff=1.2)
BASE=ReplicaExchangeConfig(n_links=4,time_step=.6,n_steps=180,burn_in=40,thin=2)

def test_methods_and_errors():
    assert make_sampler('baseline',CELL,POT,BASE).n_background==1
    assert make_sampler('exchange',CELL,POT,BASE).hot is not None
    with pytest.raises(ValueError):make_sampler('other',CELL,POT,BASE)
    with pytest.raises(ValueError):plan_budget('baseline',CELL,POT,BASE,target_seconds=-1)
    with pytest.raises(ValueError):plan_budget('baseline',CELL,POT,BASE,target_seconds=1,pilot_steps=2)

def test_pilot_and_held_out():
    p=plan_budget('baseline',CELL,POT,BASE,target_seconds=.2,pilot_steps=8,seed=13,min_validation_steps=160,max_validation_steps=160)
    assert p.validation_steps==160 and p.pilot_seconds>0
    with pytest.raises(ValueError):run_validation('baseline',CELL,POT,BASE,p,seed=13)
    row=run_validation('baseline',CELL,POT,BASE,p,seed=30)
    assert np.isfinite(row['green']) and row['cpu_seconds']>0 and row['ratio_ess']>0
    assert row['open']>=0

def test_exchange_held_out():
    p=plan_budget('exchange',CELL,POT,BASE,target_seconds=.2,pilot_steps=8,seed=15,min_validation_steps=160,max_validation_steps=160)
    r=run_validation('exchange',CELL,POT,BASE,p,seed=16)
    assert np.isfinite(r['green']) and r['ess_per_cpu']>0

def test_summary_flags():
    row=dict(method='baseline',seed=1,green=1.0,cpu_seconds=1.0,ratio_ess=2.0,
             ess_per_cpu=2.0,max_abs_influence_fraction=.2,block_stability=4.,cpu_ratio=1.)
    s=summarize_validation([row,{**row,'seed':2,'green':1.2}],1.1)
    assert s['status']=='inconclusive' and 'low_ess' in s['flags']
    with pytest.raises(ValueError):summarize_validation([row,row],1.)
