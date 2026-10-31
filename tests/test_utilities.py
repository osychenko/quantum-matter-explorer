"""Small deterministic checks for statistics and saved scan formats."""

import csv
import numpy as np
import pytest

from qmc import DMCTimestepPoint, Estimate
from qmc.resampling import systematic_resample
from qmc.stats import block_statistics
from qmc.vmc import save_dmc_dt_scan


def test_systematic_resampling_uniform():
    rng = np.random.default_rng(123)
    indices = systematic_resample(np.full(7, 1 / 7), rng)
    np.testing.assert_array_equal(indices, np.arange(7))


def test_systematic_resampling_indices_in_range():
    rng = np.random.default_rng(321)
    indices = systematic_resample(np.array([0.1, 0.1, 0.2, 0.6]), rng)
    assert indices.shape == (4,)
    assert np.all((indices >= 0) & (indices < 4))
    assert np.all(indices[:-1] <= indices[1:])


def test_block_statistics_of_known_series():
    estimate = block_statistics(np.arange(1, 9, dtype=float), block_size=2)
    assert estimate.n_blocks == 4
    assert estimate.block_size == 2
    assert estimate.mean == pytest.approx(4.5)
    assert estimate.stderr == pytest.approx(np.sqrt(5 / 3))


def test_dmc_dt_csv_has_scheme_column(tmp_path):
    point = DMCTimestepPoint(
        scheme="second_order",
        dt=0.02,
        mixed=Estimate(-0.148, 1e-5, 10, 100),
        growth=Estimate(-0.1481, 2e-5, 10, 100),
        production_acceptance=float("nan"),
    )
    output = tmp_path / "scan.csv"
    save_dmc_dt_scan([point], output)
    with output.open(newline="", encoding="utf-8") as stream:
        records = list(csv.DictReader(stream))
    assert len(records) == 1
    assert list(records[0]) == [
        "scheme", "dt", "mixed_energy", "mixed_stderr",
        "growth_energy", "growth_stderr", "acceptance",
    ]
    assert records[0]["scheme"] == "second_order"
    assert float(records[0]["dt"]) == pytest.approx(0.02)
    assert float(records[0]["mixed_energy"]) == pytest.approx(-0.148)
