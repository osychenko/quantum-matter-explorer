"""Small periodic interacting-boson VMC example (not a phase-transition scan).

Run from the project root:
    python3 -m examples.bosons.run_soft_core
"""

from __future__ import annotations

import numpy as np

from qmc import CubicCell, SoftCoreJastrowTrial, SoftCorePairPotential
from qmc import VMCConfig, run_vmc
from qmc.stats import block_statistics


def main():
    n_particles = 8
    n_walkers = 256
    cell = CubicCell(4.0)
    cutoff = 1.8  # < L/2: pairs have an unambiguous minimum image

    potential = SoftCorePairPotential(cell, height=3.0, cutoff=cutoff)
    trial = SoftCoreJastrowTrial(cell, amplitude=0.8, cutoff=cutoff)

    rng = np.random.default_rng(2026)
    initial = cell.wrap(
        rng.uniform(0.0, cell.length, size=(n_walkers, n_particles, 3))
    )

    result = run_vmc(
        potential,
        trial,
        VMCConfig(
            n_walkers=n_walkers,
            configuration_shape=(n_particles, 3),
            proposal_mode="particle",
            n_steps=300,
            burn_in=60,
            sample_every=2,
            proposal_sigma=0.35,
            adaptive_proposal=True,
            seed=2026,
        ),
        initial_walkers=initial,
    )

    est = block_statistics(result.energy_history, block_size=10)
    print(f"Particles:         {n_particles}")
    print(f"Cell volume:       {cell.volume:g}")
    print(f"Number density:    {n_particles / cell.volume:.4f}")
    print(f"VMC energy/atom:   {est.mean / n_particles:.6f} ± "
          f"{est.stderr / n_particles:.6f}")
    print(f"Sweep acceptance:  {result.production_acceptance:.3f}")
    print("Final walker shape:", result.walkers.shape)


if __name__ == "__main__":
    main()
