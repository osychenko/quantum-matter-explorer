"""Independent-replica numerical benchmark and protocol tests (Patch 8)."""

import csv
import json
from dataclasses import replace

import numpy as np
import pytest

from qmc.dmc_benchmark import (
    DMCBenchmarkConfig, DMCReplica, fit_bias_plane,
    run_dmc_benchmark, validate_benchmark,
)


def tiny():
    return DMCBenchmarkConfig(
        n_particles=3, reduced_density=0.3, gamma=4., cutoff=1.,
        jastrow_amplitudes=(0.2,), order_amplitudes=(0., 0.04),
        mode_indices=((1, 0, 0),),
        dt_values=(0.1, 0.05), walker_counts=(8, 12),
        schemes=("metropolis",), replicas=2,
        vmc_walkers=8, vmc_steps=16, vmc_burn_in=4,
        vmc_sample_every=2, vmc_block_size=2,
        warmup_sweeps=8, equilibration_tau=0.4,
        production_tau=1.2, block_tau=0.25,
        bootstrap_samples=25, seed=912,
    )


def fake_grid(*, intercept=1.23, dt_slope=2.0, inv_slope=4.0):
    """Create exact known linear surface with two independent replica offsets."""
    data = []
    for dt in (0.1, 0.05, 0.025):
        for n in (16, 32, 64):
            for replica, offset in enumerate((-0.04, 0.04)):
                value = intercept + dt_slope * dt + inv_slope/n + offset
                data.append(DMCReplica(
                    "liquid", "metropolis", dt, n, replica, 100+replica,
                    500+replica, 100, 30, 70,
                    30*dt, 70*dt,
                    value, 0.01, value + 0.07, 0.02, 0.95,
                ))
    return data


def test_extrapolation_recovers_known_surface():
    result = fit_bias_plane(fake_grid(), bootstrap_samples=60, seed=1)
    assert [r.estimator for r in result] == ["mixed", "growth"]
    assert result[0].intercept == pytest.approx(1.23, abs=1e-12)
    assert result[1].intercept == pytest.approx(1.30, abs=1e-12)
    for r in result:
        assert r.dt_slope == pytest.approx(2.0, abs=1e-12)
        assert r.inverse_walker_slope == pytest.approx(4.0, abs=1e-12)
        assert 0 < r.bootstrap_replica_stderr < 0.1
        assert r.n_grid_points == 9


def test_extrapolation_requires_complete_grid():
    p = fake_grid()
    with pytest.raises(ValueError, match="complete two-factor"):
        fit_bias_plane([q for q in p if not (q.dt == 0.05 and q.n_walkers == 16)])
    with pytest.raises(ValueError, match="replica indices"):
        bad = [replace(q, replica=0) for q in p]
        fit_bias_plane(bad)
    with pytest.raises(ValueError, match="one branch"):
        fit_bias_plane(p + [replace(p[0], branch="modulated")])


@pytest.mark.parametrize("change", [
    {"n_particles":1}, {"replicas":1}, {"walker_counts":(16,)},
    {"dt_values":(0.1,)}, {"walker_counts":(1, 20)},
    {"order_amplitudes":(0.,)}, {"order_amplitudes":(0.02,)},
    {"jastrow_amplitudes":(-0.2,)}, {"dt_values":(0.0, 0.1)},
    {"reduced_density":100.0}, {"gamma":-1.}, {"block_tau":2.0},
    {"schemes":("invalid",)}, {"schemes":("metropolis", "metropolis")},
    {"vmc_steps":3}, {"warmup_sweeps":2},
    {"bootstrap_samples":1}, {"mode_indices":((1,0,0),(-1,0,0))},
])
def test_bad_config(change):
    with pytest.raises((ValueError, TypeError)):
        validate_benchmark(replace(tiny(), **change))


def test_minimal_reproducible_end_to_end(tmp_path):
    cfg = tiny()
    train, val, summary, fit = run_dmc_benchmark(cfg, tmp_path / "a")
    assert len(train) == 2
    assert len(val) == 2
    assert len(summary) == 2 * 2 * 2  # branches * dt * populations
    assert len(fit) == 2 * 2  # branches * mixed/growth
    assert all(np.isfinite(r.mixed_energy_per_particle) for r in summary)
    assert all(np.isfinite(r.growth_energy_per_particle) for r in summary)
    assert all(0 <= r.acceptance <= 1 for r in summary)
    assert all(r.n_replicas == 2 for r in summary)
    assert all(np.isfinite(r.intercept) for r in fit)
    output = tmp_path / "a"
    for name in ("training.csv", "vmc_validation.csv", "dmc_replicas.csv",
                 "dmc_summary.csv", "extrapolation.csv", "protocol.json",
                 "bias_liquid_metropolis.png", "bias_modulated_metropolis.png"):
        assert (output / name).stat().st_size > 50
    assert json.loads((output / "protocol.json").read_text())["configuration"]["seed"] == cfg.seed
    with (output / "dmc_replicas.csv").open() as f:
        replicas = list(csv.DictReader(f))
    assert len(replicas) == 16  # 2 branches * 2 dt * 2 populations * 2 replicas
    assert len({r["seed"] for r in replicas}) == len(replicas)
    # Reproducibility checks sample summaries and the fitted intercepts.
    train2, val2, summary2, fit2 = run_dmc_benchmark(cfg, tmp_path / "b")
    assert train == train2
    assert val == val2
    assert summary == summary2
    assert fit == fit2


def test_split_propagator_returns_nan_acceptance(tmp_path):
    cfg = replace(tiny(), schemes=("second_order",))
    _, _, summary, extrap = run_dmc_benchmark(cfg, tmp_path / "split")
    assert all(np.isnan(x.acceptance) for x in summary)
    assert all(np.isfinite(x.mixed_energy_per_particle) for x in summary)
    assert len(extrap) == 4
