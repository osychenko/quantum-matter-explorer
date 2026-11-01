"""Patch-19 reproducible end-to-end CSV/plot/output contract."""
import csv
import json
import numpy as np
from examples.bosons.run_mobile_worm import run_experiment


def test_quick_workflow_and_determinism(tmp_path):
    p1=tmp_path/'one';p2=tmp_path/'two'
    a,b,c=run_experiment(p1,quick=True)
    a2,b2,c2=run_experiment(p2,quick=True)
    assert a==a2 and b==b2 and c==c2
    assert len(a)==4 and len(b)==2
    assert len(c)==12
    assert {z['model'] for z in a}=={'free_bosons','soft_core'}
    assert all(np.isfinite(z['winding_response']) for z in a)
    assert all(0<z['n_closed'] for z in a)
    assert all(0<z['n_open'] for z in a)
    for f in ('winding_response.png','coherence.png','cut_mobility.png'):
        assert (p1/f).stat().st_size>1000
    for f,n in [('replicas.csv',4),('summary.csv',2),('time_cuts.csv',12)]:
        with (p1/f).open() as handle:
            assert len(list(csv.DictReader(handle)))==n
    with (p1/'protocol.json').open() as handle:
        protocol=json.load(handle)
    assert protocol['beta']==2.1
    assert 'finite-temperature' in protocol['interpretation']
