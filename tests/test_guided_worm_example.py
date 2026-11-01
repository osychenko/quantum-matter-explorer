"""Patch 25 example output and reproducibility smoke tests."""
from pathlib import Path
import csv
import json
import numpy as np
import pytest
from examples.bosons.run_guided_worm import run_experiment


def test_example_quick_outputs(tmp_path):
    replicas,summary,iid=run_experiment(tmp_path,quick=True,steps=200,iid_draws=20,replicas=2)
    assert len(replicas)==12
    assert len(summary)==6
    assert len(iid)==8
    for filename in ('replicas.csv','summary.csv','independent_reference.csv',
                     'spectral_reference.csv','protocol.json','green_accuracy.png',
                     'cpu_efficiency.png','importance_tails.png'):
        assert (tmp_path/filename).exists()
        assert (tmp_path/filename).stat().st_size>0
    records=list(csv.DictReader((tmp_path/'summary.csv').open()))
    assert {r['method'] for r in records}=={'baseline','guided','multiple_try_4'}
    assert all(float(r['green_k0'])>0 for r in records)
    meta=json.loads((tmp_path/'protocol.json').read_text())
    assert meta['number_of_iid_draws']==20
    assert meta['links_values']==[4]


@pytest.mark.parametrize('opts',[
    dict(steps=10),dict(replicas=1),dict(iid_draws=3),
    dict(links=[3]),dict(links=[4,5]),dict(links=[]),
])
def test_example_rejects_bad_args(tmp_path,opts):
    with pytest.raises(ValueError):
        run_experiment(tmp_path,quick=True,**opts)
