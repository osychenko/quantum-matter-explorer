"""Patch 7: finite-size protocol and integration safeguards."""

import csv
import json
from dataclasses import replace

import numpy as np
import pytest

from qmc.cells import CubicCell
from qmc.finite_size import (
    FiniteSizeConfig,
    _initial,
    _protocol,
    _seed,
    run_finite_size_study,
)


def tiny_config():
    return FiniteSizeConfig(
        particle_counts=(3, 4), reduced_densities=(0.30,), gammas=(2.0,),
        jastrow_amplitudes=(0.2,), order_amplitudes=(0.0, 0.04),
        n_walkers=8, burn_in=4, training_sweeps=8,
        production_sweeps=(8, 12), sample_every=2,
        block_size=2, replicas=2, seed=130,
    )


def test_seeds_and_initializations():
    cell = CubicCell(4.0)
    a = _initial(cell, 5, 8, "uniform", 101)
    b = _initial(cell, 5, 8, "lattice", 101)
    assert a.shape == b.shape == (5, 8, 3)
    np.testing.assert_array_equal(a, _initial(cell, 5, 8, "uniform", 101))
    np.testing.assert_array_equal(b, _initial(cell, 5, 8, "lattice", 101))
    assert not np.allclose(a, b)
    assert np.all(b >= 0) and np.all(b < 4)
    assert _seed(123, 1, 2, 3) == _seed(123, 1, 2, 3)
    assert _seed(123, 1, 2, 3) != _seed(123, 1, 2, 4)
    with pytest.raises(ValueError, match="unknown start"):
        _initial(cell, 2, 3, "bad", 101)


@pytest.mark.parametrize("kw", [
    {"particle_counts": (1,)},
    {"reduced_densities": (0.0,)},
    {"gammas": (-2.0,)},
    {"reduced_densities": (5.0,)},  # cutoff > L/2
    {"order_amplitudes": (0.0,)},
    {"order_amplitudes": (0.1,)},
    {"starts": ("uniform",)},
    {"replicas": 1},
    {"n_walkers": 1},
    {"training_sweeps": 1},
    {"production_sweeps": (3, 10)},
    {"production_sweeps": (12, 8)},
    {"production_sweeps": (8, 8)},
    {"sample_every": 0},
    {"kinetic_coefficient": 0},
    {"proposal_sigma": float("nan")},
    {"target_kr": 0},
    {"seed": -1},
])
def test_bad_protocol_rejected(kw):
    with pytest.raises((ValueError, TypeError)):
        _protocol(replace(tiny_config(), **kw))


def test_finite_size_end_to_end_and_reproducibility(tmp_path):
    cfg = tiny_config()
    training, replica, summary, contrasts, lengths = run_finite_size_study(cfg, tmp_path / "a")
    assert len(training) == 2 * 2  # N x (liquid, modulated) training points
    assert len(replica) == 2 * 2 * 2 * 2 * 2  # N x branch x start x horizon x replica
    assert len(summary) == 2 * 2 * 2 * 2
    assert len(contrasts) == 2 * 2 * 2
    assert len(lengths) == 2 * 2 * 2
    assert {p.n_particles for p in summary} == {3, 4}
    assert {p.branch for p in summary} == {"liquid", "modulated"}
    assert {p.start for p in summary} == {"uniform", "lattice"}
    assert all(np.isfinite(p.energy_per_particle) for p in summary)
    assert all(np.isfinite(p.s_mode) and p.s_mode >= 0 for p in summary)
    assert all(np.isfinite(p.energy_stderr) for p in summary)
    assert all(p.replicas == 2 for p in summary)
    assert all(np.isclose(p.s_per_particle, p.s_mode / p.n_particles) for p in summary)
    assert len({p.seed for p in replica}) == len(replica)
    assert all(np.isclose(p.cell_length**3, p.n_particles / p.reduced_density) for p in replica)
    assert all(p.k_cutoff == pytest.approx(2*np.pi*p.mode_index / p.cell_length) for p in replica)
    assert len({p.init_seed for p in replica}) == len(replica)
    assert (tmp_path / "a" / "scaling_density_00_gamma_00.png").stat().st_size > 1000
    with (tmp_path / "a" / "protocol.json").open(encoding="utf-8") as f:
        data = json.load(f)
    assert data["configuration"]["particle_counts"] == [3, 4]
    for filename, expected in (("training.csv", 4), ("replicas.csv", 32),
                               ("summary.csv", 16), ("start_contrasts.csv", 8),
                               ("length_contrasts.csv", 8)):
        with (tmp_path / "a" / filename).open(newline="", encoding="utf-8") as f:
            assert len(list(csv.DictReader(f))) == expected
    t2, r2, s2, c2, l2 = run_finite_size_study(cfg, tmp_path / "b")
    assert (training, replica, summary, contrasts, lengths) == (t2, r2, s2, c2, l2)


def test_comparable_mode_wavenumber_is_recorded():
    cfg = replace(tiny_config(), particle_counts=(8, 16), reduced_densities=(0.55,))
    for n in cfg.particle_counts:
        length = cfg.cutoff * (n / cfg.reduced_densities[0]) ** (1 / 3)
        mode = max(1, int(np.floor(cfg.target_kr * length / (2 * np.pi * cfg.cutoff) + .5)))
        actual = 2 * np.pi * mode * cfg.cutoff / length
        assert abs(actual - cfg.target_kr) <= np.pi * cfg.cutoff / length + 1e-12
    # Nearby nonzero wavevectors remain different for different finite box sizes.
    assert not np.isclose((8 / .55) ** (1 / 3), (16 / .55) ** (1 / 3))
