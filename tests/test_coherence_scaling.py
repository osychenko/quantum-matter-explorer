"""Patch 10 validation and end-to-end VMC coherence scaling tests."""

import csv
import json
from dataclasses import replace

import numpy as np
import pytest

from qmc.cells import CubicCell
from qmc.coherence_scaling import (
    CoherenceScalingConfig, _summary, commensurate_modes,
    run_coherence_scaling, validate_scaling_config,
)


def small():
    return CoherenceScalingConfig(
        particle_counts=(4, 6), reduced_densities=(0.5,), gammas=(4.0,),
        jastrow_amplitudes=(0.2, 0.7), order_amplitudes=(0.0, 0.08),
        cutoff=1.0, n_walkers=8, training_steps=14,
        validation_steps=18, burn_in=4, sample_every=3,
        proposal_sigma=0.28, n_radii=4, draws_per_configuration=1,
        replicas=2, seed=3434,
    )


def test_config_and_unit_controls():
    Ns, densities, gammas, amps, orders = validate_scaling_config(small())
    assert Ns == (4, 6)
    assert densities == (0.5,)
    assert gammas == (4.0,)
    assert 0 in orders and 0 not in amps
    cell = CubicCell(4.0)
    indices, m, k = commensurate_modes(cell, 1.0, 2 * np.pi)
    assert m == 4
    np.testing.assert_array_equal(indices, [[4, 0, 0], [0, 4, 0], [0, 0, 4]])
    assert k == pytest.approx(2 * np.pi)


def test_nearest_mode_can_differ_between_sizes():
    a = commensurate_modes(CubicCell(2.12), 1.0, 2 * np.pi)
    b = commensurate_modes(CubicCell(3.17), 1.0, 2 * np.pi)
    assert a[1] == 2 and b[1] == 3
    assert abs(a[2] - b[2]) > 0.01


@pytest.mark.parametrize("replace_with", [
    {"particle_counts": (4,)},
    {"particle_counts": (4, 4)},
    {"particle_counts": (1, 8)},
    {"particle_counts": (4.1, 6)},
    {"reduced_densities": (2.0,)},  # L < 2*rc for N=4
    {"reduced_densities": (float("nan"),)},
    {"gammas": (0.0,)},
    {"cutoff": -1.0},
    {"kinetic_coefficient": 0.0},
    {"target_k_cutoff": 0.0},
    {"order_amplitudes": (0.0,)},
    {"order_amplitudes": (0.08,)},
    {"order_amplitudes": (0., -0.1)},
    {"jastrow_amplitudes": (-1.0,)},
    {"training_steps": 4, "burn_in": 4},
    {"validation_steps": 4, "burn_in": 4},
    {"training_steps": 5, "burn_in": 4, "sample_every": 3},
    {"replicas": 1},
    {"n_walkers": 1},
    {"draws_per_configuration": 0},
    {"n_radii": 1},
    {"seed": -4},
    {"proposal_sigma": 0.0},
])
def test_invalid_config(replace_with):
    with pytest.raises((ValueError, TypeError)):
        validate_scaling_config(replace(small(), **replace_with))


def test_replica_sem_requires_at_least_two_independent_runs():
    with pytest.raises(ValueError):
        _summary([1.0])
    assert _summary([0., 2.]) == pytest.approx((1., 1.))


def test_full_small_workflow_reproducible(tmp_path):
    cfg = small()
    training, replica, summary = run_coherence_scaling(cfg, tmp_path / "first")
    assert len(training) == 2 * 2 * 2  # 2 N, 2 a, 2 lambda
    assert len(replica) == 2 * 2 * cfg.replicas
    assert len(summary) == 2 * 2
    assert {r["branch"] for r in summary} == {"liquid", "modulated"}
    assert all(np.isfinite(r["n0_over_n"]) for r in summary)
    assert all(np.isfinite(r["s_over_n"]) for r in summary)
    assert all(np.isfinite(r["n0_stderr"]) for r in summary)
    assert all(r["s_over_n"] >= 0 for r in summary)
    assert all(r["replicas"] == 2 for r in summary)
    assert all(r["order_amplitude"] == 0 for r in summary if r["branch"] == "liquid")
    assert all(r["order_amplitude"] > 0 for r in summary if r["branch"] == "modulated")
    assert len({r["seed"] for r in training + replica}) == len(training) + len(replica)
    for name in ("scaling.png", "coherence_vs_structure.png", "g1_scaling.png"):
        assert (tmp_path / "first" / name).stat().st_size > 1000
    with (tmp_path / "first" / "protocol.json").open(encoding="utf-8") as f:
        protocol = json.load(f)
    assert protocol["config"]["particle_counts"] == [4, 6]
    with (tmp_path / "first" / "g1.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4 * cfg.n_radii
    assert all(float(x["g1"]) == 1.0 for x in rows if x["radius"] == "0.0")
    for name, expected in (("training.csv", len(training)),
                           ("replicas.csv", len(replica)),
                           ("summary.csv", len(summary))):
        with (tmp_path / "first" / name).open(encoding="utf-8") as f:
            assert len(list(csv.DictReader(f))) == expected
    a, b, c = run_coherence_scaling(cfg, tmp_path / "second")
    assert training == a and replica == b and summary == c
