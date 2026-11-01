"""Patch 21 independent-replica artifact and reproducibility contract."""
import csv
import json

import numpy as np

from examples.bosons.run_continuum_spacetime_worm import run_experiment


def _records(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_quick_csv_and_plots(tmp_path):
    replicas, summary=run_experiment(tmp_path,quick=True)
    assert len(replicas)==3*2*3*3  # cases * replicas * time separations * modes
    assert len(summary)==3*3*3
    assert (tmp_path/"green_vs_time.png").stat().st_size>1000
    assert (tmp_path/"nonzero_momentum.png").stat().st_size>1000
    assert len(_records(tmp_path/"replicas.csv"))==len(replicas)
    assert len(_records(tmp_path/"summary.csv"))==len(summary)
    protocol=json.loads((tmp_path/"protocol.json").read_text())
    assert protocol["beta"]==2.4
    assert protocol["n_links_grid"]==[4]
    assert any(x["potential_height"]>0 for x in replicas)
    assert all(np.isfinite(float(x["green"])) for x in replicas)
    assert all(np.isfinite(float(x["green_replica_sem"])) for x in summary)
    assert all(x["exact_free"]=="" for x in summary if x["potential_height"]>0)
    assert all(x["exact_free"]!="" for x in summary if x["potential_height"]==0)


def test_experiment_override_validation(tmp_path):
    import pytest
    with pytest.raises(ValueError,match="replicas_override"):
        run_experiment(tmp_path,quick=True,replicas_override=1)
    with pytest.raises(ValueError,match="steps_override"):
        run_experiment(tmp_path,quick=True,steps_override=20)
