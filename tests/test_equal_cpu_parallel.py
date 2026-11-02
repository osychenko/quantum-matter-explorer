"""Patch 28.1: deterministic multiprocessing and calibrated CPU pilots."""
from dataclasses import replace
import json

import numpy as np
import pytest

from qmc.bosons import SoftCorePairPotential
from qmc.cells import CubicCell
from qmc.equal_cpu_worm import plan_budget
from qmc.replica_exchange_worm import ReplicaExchangeConfig
from examples.bosons.run_equal_cpu_worm import (
    _validation_task, _validation_tasks, _map_validations, run_experiment,
)


def _tasks(replicas=3):
    cell = CubicCell(2.8)
    potential = SoftCorePairPotential(cell, height=3., cutoff=1.2)
    cfg = ReplicaExchangeConfig(n_links=4, time_step=.6,
                                n_steps=160, burn_in=40, thin=2, sector_fugacity=.9)
    plan = plan_budget('baseline', cell, potential, cfg,
                       target_seconds=.1, pilot_steps=8,
                       seed=111, min_validation_steps=160, max_validation_steps=160)
    return _validation_tasks('baseline', cell, potential, cfg,
                             plan, 4, 2.4, 1.69, replicas)


def test_seeds_are_stable_and_disjoint():
    tasks = _tasks(8)
    assert [t[5] for t in tasks] == [340000+i for i in range(8)]
    x = _validation_tasks('exchange', tasks[0][1], tasks[0][2],
                          tasks[0][3], tasks[0][4], 4, 2.4, 1.69, 8)
    assert set(t[5] for t in tasks).isdisjoint(t[5] for t in x)


def test_actual_serial_vs_parallel_physics_equal():
    tasks = _tasks(3)
    serial = _map_validations(tasks, 1)
    parallel = _map_validations(tasks, 2)
    for a, b in zip(serial, parallel):
        assert a.keys() == b.keys()
        for k in a:
            if k in ('cpu_seconds', 'cpu_ratio', 'ess_per_cpu'):
                continue
            assert a[k] == pytest.approx(b[k], nan_ok=True) if isinstance(a[k], float) else a[k] == b[k]
    assert len({r['seed'] for r in parallel}) == 3


def test_realistic_pilot_uses_audited_workload(monkeypatch):
    import qmc.equal_cpu_worm as core
    states = []

    class FakeSampler:
        def __init__(self, config):
            self.config = config

    monkeypatch.setattr(core, 'make_sampler', lambda method, cell, pot, cfg: FakeSampler(cfg))
    monkeypatch.setattr(core, 'audit_worm_chain', lambda sampler, n_blocks: states.append((sampler.config, n_blocks)))
    x = core.plan_budget('baseline', None, None,
                          ReplicaExchangeConfig(n_links=4, time_step=.6, n_steps=200,
                                                burn_in=50, thin=2),
                          target_seconds=2, pilot_steps=400, seed=55,
                          min_validation_steps=160, max_validation_steps=200000)
    assert len(states) == 1
    assert states[0][0].n_steps == 400
    assert states[0][0].burn_in == 100
    assert states[0][0].thin == 2
    assert states[0][1] == 4
    assert x.seed == 55 and x.pilot_steps == 400


def test_invalid_workers_or_pilot_steps(tmp_path):
    for bad in (0, -1, 1.5, True, '6', 65):
        with pytest.raises(ValueError, match='workers'):
            run_experiment(tmp_path, quick=True, workers=bad)
    for bad in (0, 7, 1.5, True, '400'):
        with pytest.raises(ValueError, match='pilot_steps'):
            run_experiment(tmp_path, quick=True, pilot_steps=bad)


def test_parallel_example_outputs_and_timings(tmp_path):
    pilots, replicas, summary = run_experiment(
        tmp_path, quick=True, cpu_seconds=.12, replicas=2,
        workers=2, pilot_steps=12)
    assert len(pilots) == 2 and len(replicas) == 4 and len(summary) == 2
    metadata = json.loads((tmp_path/'protocol.json').read_text())
    assert metadata['workers'] == 2 and metadata['pilot_steps'] == 12
    assert metadata['total_wall_seconds'] > 0
    assert metadata['total_validation_cpu_seconds'] > 0
    assert metadata['total_pilot_cpu_seconds'] > 0
    assert metadata['total_validation_cpu_seconds'] == pytest.approx(
        sum(r['cpu_seconds'] for r in replicas))
    for name in ('pilots.csv', 'replicas.csv', 'summary.csv',
                 'spectral_reference.csv', 'protocol.json',
                 'accuracy.png', 'efficiency.png'):
        assert (tmp_path/name).stat().st_size > 0
