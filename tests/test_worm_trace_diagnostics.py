from dataclasses import replace
import numpy as np
import pytest

from qmc.cells import CubicCell
from qmc.bosons import SoftCorePairPotential
from qmc.equal_cpu_worm import BudgetPlan,run_validation
from qmc.replica_exchange_worm import ReplicaExchangeConfig
from qmc.worm_trace_diagnostics import (load_trace,autocorrelation_curve,
    blocking_curve,running_ratio,group_diagnostics)
from examples.bosons.run_equal_cpu_worm import _validation_tasks,_map_validations
from examples.bosons.run_worm_trace_diagnostics import run_analysis


def test_constant_and_white_noise_acf():
    x=np.ones(1024)
    assert autocorrelation_curve(x)['warning']=='constant_trace'
    r=autocorrelation_curve(np.random.default_rng(11).normal(size=10000))
    assert 0.5 <= r['tau_int'] < 3
    assert 1000 < r['ess'] <= 10000


def test_trace_serialization_and_analysis(tmp_path):
    cell=CubicCell(2.8)
    potential=SoftCorePairPotential(cell,height=0.0,cutoff=1.2)
    base=ReplicaExchangeConfig(n_links=4,time_step=.6,n_steps=200,burn_in=30,thin=2,
        sector_fugacity=.9)
    plan=BudgetPlan('baseline',1.,40,.1,340,110)
    output=tmp_path/'traces'
    files=[]
    for seed in (340000,340001,340002):
        path=output/f'M4_baseline_seed{seed}.npz'
        run=run_validation('baseline',cell,potential,base,plan,seed=seed,trace_path=path)
        trace=load_trace(path)
        assert np.isclose(run['green'],trace['green'])
        assert len(running_ratio(trace))==len(trace['numerator'])
        assert len(blocking_curve(trace))>=2
        files.append(trace)
    g=group_diagnostics(files)
    assert g['replicas']==3 and np.isfinite(g['split_rhat_influence'])
    result=tmp_path/'figs'
    summary,_,_=run_analysis(output,output=result)
    assert len(summary)==1
    for name in ('trace_summary.csv','blocking_curves.csv','autocorrelation.csv',
                 'running_green.png','blocking_curves.png','autocorrelation.png','protocol.json'):
        assert (result/name).stat().st_size>0


def test_parallel_tasks_write_unique_traces(tmp_path):
    cell=CubicCell(2.8)
    potential=SoftCorePairPotential(cell,height=0.0,cutoff=1.2)
    base=ReplicaExchangeConfig(n_links=4,time_step=.6,n_steps=200,burn_in=30,thin=2,sector_fugacity=.9)
    plan=BudgetPlan('baseline',1.,40,.1,320,110)
    path=str(tmp_path/'traces')
    tasks=_validation_tasks('baseline',cell,potential,base,plan,4,2.4,2.,2,path)
    shifted=_validation_tasks('baseline',cell,potential,base,plan,4,2.4,2.,2,path,2000000)
    assert shifted[0][5] == tasks[0][5] + 2000000
    serial=_map_validations(tasks,1)
    parallel=_map_validations(tasks,2)
    assert [r['green'] for r in serial]==[r['green'] for r in parallel]
    assert len(list((tmp_path/'traces').glob('*.npz')))==2


def test_invalid_and_incompatible(tmp_path):
    with pytest.raises(ValueError):
        group_diagnostics([])
    with pytest.raises(ValueError):
        autocorrelation_curve([1,2,3])
    with pytest.raises(ValueError):
        run_analysis(tmp_path,output=tmp_path/'out')
