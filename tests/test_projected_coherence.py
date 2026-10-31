"""Patch 11: analytic extrapolation and actual small DMC/VMC integration."""

from dataclasses import replace
import csv
import json

import numpy as np
import pytest

from qmc.projected_coherence import (
    ProjectedCoherenceConfig,
    _mean_sem,
    validate_projected_config,
    run_projected_coherence,
)


def minimum():
    # Uniform exact ground state of the free periodic Bose Hamiltonian:
    # Psi_T = constant, all off-diagonal ratios identically 1.
    return ProjectedCoherenceConfig(
        n_particles=3, cell_length=4., cutoff=1.5,
        potential_height=0., jastrow_amplitude=0., order_amplitude=0.,
        branches=("liquid",), dt_values=(0.04,), walker_populations=(12,),
        equilibration_tau_values=(0.08, 0.16), production_tau=0.16,
        vmc_walkers=12, vmc_steps=12, vmc_burn_in=4, vmc_sample_every=2,
        observable_stride=1, n_radii=3, replicas=2, seed=75,
    )


def test_exact_analytic_mixed_extrapolation_cancels_first_order_error():
    # One periodic orbital psi_0(x)=1+a cos(kx); trial=1+b cos(kx).
    # Full-cell zero-k occupation can be evaluated analytically:
    # pure=1/(1+a²/2); mixed=1/(1+ab/2); VMC=1/(1+b²/2).
    a = 0.4
    exact = 1/(1+a*a/2)
    errors = []
    for eps in (0.04, 0.02):
        b = a + eps
        v = 1/(1+b*b/2)
        m = 1/(1+a*b/2)
        errors.append(abs(2*m-v-exact))
    assert errors[1] < 0.28*errors[0]  # quadratic rather than linear


@pytest.mark.parametrize("changes", [
    {"n_particles": 1}, {"cell_length": -1}, {"cutoff": 2.5},
    {"potential_height": -1}, {"jastrow_amplitude": -0.2},
    {"branches": ()}, {"branches": ("liquid", "liquid")},
    {"branches": ("unknown",)}, {"dt_values": (0,)},
    {"dt_values": (0.04, 0.04)}, {"walker_populations": (1,)},
    {"equilibration_tau_values": (-0.1,)}, {"production_tau": 0},
    {"observable_stride": 0}, {"vmc_walkers": 1},
    {"vmc_steps": 0}, {"vmc_burn_in": 12}, {"vmc_sample_every": 0},
    {"n_radii": 1}, {"draws_per_configuration": 0},
    {"replicas": 1}, {"proposal_sigma": 0}, {"seed": -1},
    {"scheme": "bad"}, {"indices": ((1,0,0),(-1,0,0))},
])
def test_bad_configs(changes):
    with pytest.raises((ValueError, TypeError)):
        validate_projected_config(replace(minimum(), **changes))


def test_extrapolated_sem_must_use_replica_level_differences():
    vmc = np.array([0.2, 0.4, 0.7])
    mixed = np.array([0.3, 0.6, 0.8])
    mean, sem = _mean_sem(2*mixed-vmc)
    assert mean == pytest.approx(np.mean(2*mixed-vmc))
    assert sem == pytest.approx(np.std(2*mixed-vmc,ddof=1)/np.sqrt(3))
    with pytest.raises(ValueError):
        _mean_sem([1.])


def test_real_dmc_free_reference_and_output(tmp_path):
    cfg = minimum()
    rows, summaries = run_projected_coherence(cfg, tmp_path)
    assert len(rows) == 4 and len(summaries) == 2
    for r in rows:
        assert r.vmc_n0_over_n == 1
        assert r.mixed_n0_over_n == 1
        assert r.extrapolated_n0_over_n == 1
        assert r.vmc_energy_per_particle == pytest.approx(0)
        assert r.dmc_mixed_energy_per_particle == pytest.approx(0)
        assert r.n_observed_dmc_configurations > 0
        assert r.vmc_seed != r.dmc_seed
    for p in ("replicas.csv", "summary.csv", "g1.csv", "protocol.json",
              "occupation_liquid.png", "g1_liquid.png"):
        assert (tmp_path / p).stat().st_size > 0
    with (tmp_path / "g1.csv").open() as f:
        g = list(csv.DictReader(f))
    assert len(g) == 2*cfg.n_radii
    assert all(float(r["extrapolated_g1"]) == 1 for r in g)
    payload = json.loads((tmp_path/"protocol.json").read_text())
    assert "NOT pure" in payload["disclaimer"]


def test_second_order_dmc_observable_hook(tmp_path):
    rows, summary = run_projected_coherence(
        replace(minimum(), scheme="second_order", equilibration_tau_values=(0.08,)), tmp_path
    )
    assert len(rows) == 2 and len(summary) == 1
    assert all(r.extrapolated_n0_over_n == 1 for r in rows)


def test_reproducible_quick_case(tmp_path):
    cfg = replace(minimum(), equilibration_tau_values=(0.08,))
    a, s = run_projected_coherence(cfg, tmp_path/"one")
    b, t = run_projected_coherence(cfg, tmp_path/"two")
    assert a == b
    assert s == t
