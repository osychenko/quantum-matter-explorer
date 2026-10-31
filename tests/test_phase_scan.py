"""Fast independent-validation scan and output-contract tests."""

import csv
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from qmc.phase_scan import PhaseScanConfig, run_phase_scan, _check_config


def minimal():
    return PhaseScanConfig(
        n_particles=3, cell_length=4, cutoff=1.7,
        strengths=(4.,), jastrow_amplitudes=(0.2, 0.7),
        order_amplitudes=(0.0, 0.08), mode_indices=((1, 0, 0), (0, 1, 0)),
        n_walkers=10, n_steps=20, burn_in=6, sample_every=2,
        block_size=2, validation_replicas=2, n_bins=6, seed=2101,
    )


def test_validation_scan_reproducible_and_files(tmp_path):
    cfg = minimal()
    points, valid = run_phase_scan(cfg, tmp_path / "run1")
    assert len(points) == 4
    assert len(valid) == 2
    assert {v.branch for v in valid} == {"liquid", "modulated"}
    assert all(np.isfinite(v.energy_per_particle) for v in valid)
    assert all(np.isfinite(v.s_mode_mean) for v in valid)
    assert all(v.n_replicas == 2 for v in valid)
    assert all(v.density_cutoff3 == pytest.approx(3 * 1.7**3 / 4**3) for v in valid)
    assert all(v.order_amplitude == 0 for v in valid if v.branch == "liquid")
    assert all(v.order_amplitude > 0 for v in valid if v.branch == "modulated")
    assert (tmp_path / "run1" / "scan.png").stat().st_size > 1000
    assert (tmp_path / "run1" / "cell_gamma_00_liquid.png").stat().st_size > 1000
    with (tmp_path / "run1" / "replicas.csv").open() as f:
        assert len(list(csv.DictReader(f))) == 4
    with (tmp_path / "run1" / "gr_gamma_00_modulated.csv").open() as f:
        g = list(csv.DictReader(f))
        assert len(g) == cfg.n_bins
    with (tmp_path / "run1" / "sk_gamma_00_modulated.csv").open() as f:
        k = list(csv.DictReader(f))
        assert len(k) == len(cfg.mode_indices)
    points2, valid2 = run_phase_scan(cfg, tmp_path / "run2")
    assert points == points2
    assert valid == valid2


@pytest.mark.parametrize("modification", [
    {"validation_replicas":1},
    {"order_amplitudes":(0.0,)},
    {"order_amplitudes":(0.1,0.2)},
    {"strengths":(0.,)},
    {"cell_length":-5.0},
    {"cutoff":2.1},
    {"block_size":100},
    {"mode_indices":((1,0,0),(-1,0,0))},
])
def test_bad_scan_config(modification):
    with pytest.raises((ValueError, TypeError)):
        _check_config(replace(minimal(), **modification))
