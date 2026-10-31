"""Plot g(r), directional S(k), and a periodic cell snapshot for soft-core bosons.

From the repository root:
    python3 -m examples.bosons.run_observables

Outputs in outputs/bosons/. This is an illustrative VMC example, not a
thermodynamic-limit phase diagram or a condensate-fraction estimator.
"""

from pathlib import Path

import numpy as np

from qmc import CubicCell, SoftCoreJastrowTrial, SoftCorePairPotential, VMCConfig, run_vmc
from qmc.observables import PairCorrelation, StaticStructureFactor, reciprocal_index_grid
from qmc.stats import block_statistics
from qmc.visualization import (
    plot_cell_snapshot,
    plot_pair_correlation,
    plot_structure_factor,
)


def main():
    n_particles, n_walkers = 8, 128
    cell = CubicCell(4.0)
    potential = SoftCorePairPotential(cell, height=3.0, cutoff=1.8)
    trial = SoftCoreJastrowTrial(cell, amplitude=0.8, cutoff=1.8)

    initial = np.random.default_rng(2026).uniform(
        0.0, cell.length, size=(n_walkers, n_particles, 3)
    )
    pair_correlation = PairCorrelation(cell, n_particles, n_bins=30)
    structure = StaticStructureFactor(cell, n_particles, reciprocal_index_grid(1))
    result = run_vmc(
        potential, trial,
        VMCConfig(
            n_walkers=n_walkers,
            configuration_shape=(n_particles, 3),
            proposal_mode="particle",
            n_steps=150,
            burn_in=40,
            sample_every=5,
            proposal_sigma=0.35,
            adaptive_proposal=True,
            seed=2026,
        ),
        initial_walkers=initial,
        observables=(pair_correlation, structure),
    )
    energy = block_statistics(result.energy_history, block_size=5)
    print(f"Energy per particle: {energy.mean / n_particles:.6f}"
          f" +/- {energy.stderr / n_particles:.6f}")
    print(f"Move acceptance: {result.production_acceptance:.3f}")
    print(f"Accumulated configurations: {pair_correlation.n_configurations}")

    out = Path("outputs/bosons")
    p = [
        plot_cell_snapshot(result.walkers, cell, out / "cell_snapshot.png"),
        plot_pair_correlation(pair_correlation.result(), out / "pair_correlation.png"),
        plot_structure_factor(structure.result(), out / "structure_factor.png"),
    ]
    for file in p:
        print("Saved:", file)


if __name__ == "__main__":
    main()
