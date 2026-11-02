import csv
import json
from pathlib import Path
from examples.bosons.run_equal_cpu_worm import run_experiment

def test_quick_example(tmp_path):
    plans,rows,summaries=run_experiment(tmp_path,quick=True,cpu_seconds=.12,replicas=2)
    assert len(plans)==2 and len(rows)==4 and len(summaries)==2
    for filename in ('pilots.csv','replicas.csv','summary.csv','spectral_reference.csv','protocol.json','accuracy.png','efficiency.png'):
        assert (tmp_path/filename).stat().st_size>0
    assert json.loads((tmp_path/'protocol.json').read_text())['replicas']==2
    with (tmp_path/'replicas.csv').open() as stream:
        data=list(csv.DictReader(stream))
    assert {r['method'] for r in data}=={'baseline','exchange'}
    assert len({r['seed'] for r in data})==4
