import csv
import numpy as np
import pytest
from qmc.engine_readiness import assess_preset,compare_methods,apply_cross_checks
from examples.bosons.run_engine_readiness import run_readiness


def rows(M, method, values, start):
    return [dict(M=M,method=method,seed=start+i,green=x,ratio_ess=500)
            for i,x in enumerate(values)]


def test_reference_discrepancy_blocks():
    x=rows(12,'exchange',[2.10,2.15,2.11,2.22,2.17,2.12,2.09,2.18],100)
    r=assess_preset(x)
    assert r['status']=='blocked' and 'reference_discrepancy' in r['blocking_flags']
    assert r['trace_evidence']=='missing'


def test_trace_absence_is_not_certification():
    x=rows(8,'baseline',[1.65,1.69,1.70,1.66,1.68,1.70,1.67,1.69],100)
    r=assess_preset(x)
    assert r['status']=='needs_evidence'
    assert 'traces_missing' in r['evidence_warnings']


def test_duplicate_seed_is_invalid():
    x=rows(8,'baseline',[1.7]*8,100)
    x[-1]['seed']=100
    with pytest.raises(ValueError,match='duplicate'):
        assess_preset(x)


def test_cross_method_disagreement_blocks_both():
    a=dict(M=12,method='baseline',mean_green=1.69,replica_sem=.04,status='candidate',blocking_flags='none')
    b=dict(M=12,method='exchange',mean_green=2.13,replica_sem=.08,status='candidate',blocking_flags='none')
    checks=compare_methods([a,b]);assert checks[0]['disagreement_z']>2
    apply_cross_checks([a,b],checks)
    assert a['status']==b['status']=='blocked'


def test_cli_csv_only(tmp_path):
    data=rows(8,'baseline',[1.65,1.69,1.70,1.66,1.68,1.70,1.67,1.69],100)
    f=tmp_path/'replicas.csv'
    with f.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
    output=tmp_path/'report'
    result,checks,gate=run_readiness(f,output=output)
    assert result[0]['status']=='needs_evidence'
    assert not gate['ready_for_engine_freeze']
    assert (output/'decision.json').exists() and (output/'readiness_report.md').exists()


def test_realistic_trace_group_passes_shape_checks():
    from qmc.engine_readiness import chain_influence_report
    from qmc.worm_trace_diagnostics import group_diagnostics
    rng=np.random.default_rng(29)
    traces=[]; inputs=[]
    for j in range(8):
        n=1024
        d=np.ones(n)*.8
        # Same ratio target with moderate independent fluctuations.
        v=1.70 + rng.normal(scale=.5,size=n)
        num=d*v / 7
        G=7*num.sum()/d.sum()
        traces.append(dict(method='baseline',M=8,seed=100+j,beta=2.4,scale=7.,
                           numerator=num,denominator=d,green=G))
        inputs.append(dict(method='baseline',M=8,seed=100+j,green=G,ratio_ess=500))
    diagnostic=group_diagnostics(traces)
    assert diagnostic['replicas']==8
    r=assess_preset(inputs,traces=traces)
    assert r['trace_evidence']=='available'
    # High concentration of estimates can be more precise than reference,
    # but all flags are still computed and no proof is claimed.
    inf=chain_influence_report(inputs,traces=traces)
    assert len(inf)==8
    assert all(np.isfinite(x['second_minus_first']) for x in inf)
