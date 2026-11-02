"""Patch 27 output contract (small, reproducible quick study)."""
import csv
import json
from pathlib import Path
import numpy as np
import pytest

from examples.bosons.run_replica_exchange_worm import run_experiment


def test_small_quick_example(tmp_path):
    rows,summary=run_experiment(tmp_path,quick=True,steps=360,replicas=2,links=(4,))
    assert len(rows)==4 and len(summary)==2
    assert {r['sampler'] for r in summary}=={'baseline','replica_exchange'}
    assert all(np.isfinite(r['green']) for r in rows)
    assert all(r['n_steps']==360 for r in rows)
    for name in ('replicas.csv','summary.csv','spectral_reference.csv',
                 'protocol.json','green_accuracy.png','cpu_efficiency.png'):
        assert (tmp_path/name).stat().st_size>100
    with (tmp_path/'replicas.csv').open() as f:
        assert len(list(csv.DictReader(f)))==4
    assert json.loads((tmp_path/'protocol.json').read_text())['steps']==360


@pytest.mark.parametrize('override',[
    dict(steps=150),dict(replicas=1),dict(links=(3,)),
    dict(height=-1.),dict(hot_fraction=-.1)])
def test_bad_input(tmp_path, override):
    with pytest.raises(ValueError):
        run_experiment(tmp_path,quick=True,**override)
