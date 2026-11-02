"""Patch 28.2: distinguish physical audit flags from CPU planning warnings."""
import numpy as np
import pytest
from dataclasses import replace
from qmc.cells import CubicCell
from qmc.bosons import SoftCorePairPotential
from qmc.replica_exchange_worm import ReplicaExchangeConfig
from qmc.equal_cpu_worm import BudgetPlan, run_validation, summarize_validation, plan_budget


def test_multiscale_blocks_real_chain():
    cell=CubicCell(2.8)
    pot=SoftCorePairPotential(cell,height=3.,cutoff=1.2)
    base=ReplicaExchangeConfig(n_links=4,time_step=.6,sector_fugacity=.9)
    plan=BudgetPlan('baseline',1.,600,1.,350,100)
    row=run_validation('baseline',cell,pot,base,plan,seed=101)
    assert set(row['block_scales'].split(','))=={'4','8','16'}
    assert np.isfinite(row['block_sem_4'])
    assert np.isfinite(row['block_sem_8'])
    assert np.isfinite(row['block_sem_16'])
    assert row['block_stability'] >= 1


def test_budget_mismatch_is_operational():
    rows=[{'method':'baseline','seed':i,'green':1.691+(i-2)*.003,'ratio_ess':100.,
           'max_abs_influence_fraction':.002,'block_stability':1.2,
           'cpu_ratio':1.75,'cpu_seconds':35.,'ess_per_cpu':3.} for i in (1,2,3)]
    result=summarize_validation(rows,1.691)
    assert result['status']=='screen_pass_not_proof'
    assert result['flags']=='no_flags'
    assert result['cpu_status']=='budget_mismatch'
    assert result['cpu_flags']=='cpu_budget_mismatch'


def test_block_instability_remains_statistical_flag():
    rows=[{'method':'baseline','seed':i,'green':1.69,'ratio_ess':100.,
           'max_abs_influence_fraction':.002,'block_stability':4.,
           'cpu_ratio':1.0,'cpu_seconds':20.,'ess_per_cpu':5.} for i in (1,2)]
    result=summarize_validation(rows,1.69)
    assert result['status']=='inconclusive'
    assert 'blocking_unstable' in result['flags']
    assert result['cpu_status']=='within_tolerance'


def test_calibration_factor_scales_step_plan(monkeypatch):
    import qmc.equal_cpu_worm as mod
    class Stub:
        def step(self): pass
    monkeypatch.setattr(mod,'make_sampler',lambda *a:Stub())
    durations=iter((0.,.6,0.,.6))
    monkeypatch.setattr(mod.time,'process_time',lambda:next(durations))
    cell=CubicCell(2.8);pot=SoftCorePairPotential(cell,height=3.,cutoff=1.2)
    cfg=ReplicaExchangeConfig(n_links=4,time_step=.6)
    a=plan_budget('baseline',cell,pot,cfg,target_seconds=12,pilot_steps=12,calibration_factor=1.)
    b=plan_budget('baseline',cell,pot,cfg,target_seconds=12,pilot_steps=12,calibration_factor=1.5)
    assert a.validation_steps==240
    assert b.validation_steps==160
    with pytest.raises(ValueError,match='calibration_factor'):
        plan_budget('baseline',cell,pot,cfg,target_seconds=12,calibration_factor=0)
