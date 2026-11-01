"""Patch 24: end-to-end benchmark output and basic scientific result integrity."""
import csv
import json
import pytest

from examples.bosons.run_rare_worm import run_experiment


def test_quick_outputs_and_methods(tmp_path):
    diagnostics,summary=run_experiment(tmp_path,quick=True,steps=160,replicas=2)
    assert len(diagnostics)==8
    assert len(summary)==2*2*3*2  # case, method, intervals, momenta
    assert {r['method'] for r in summary}=={'baseline','multiple_try_4'}
    for name in ('replicas.csv','summary.csv','diagnostics.csv','independent_reference.csv',
                 'spectral_reference.csv','protocol.json','green_reference.png','rare_efficiency.png'):
        path=tmp_path/name
        assert path.exists() and path.stat().st_size>50
    with (tmp_path/'summary.csv').open(newline='') as f:
        points=list(csv.DictReader(f))
    assert len(points)==len(summary)
    assert all(p['screening'] in ('inconclusive','reference_discrepancy') for p in points)
    with (tmp_path/'independent_reference.csv').open(newline='') as f:
        independents=list(csv.DictReader(f))
    assert len(independents)==2*2*3*2
    assert all(float(x['draws'])==400 for x in independents)
    protocol=json.loads((tmp_path/'protocol.json').read_text())
    assert protocol['beta']==2.4 and protocol['replicas']==2


@pytest.mark.parametrize('steps,replicas',[(159,2),(160,1),(True,2),(160,False)])
def test_invalid_input(tmp_path,steps,replicas):
    with pytest.raises(ValueError):run_experiment(tmp_path,quick=True,steps=steps,replicas=replicas)
