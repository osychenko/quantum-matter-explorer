"""Compare VMC, importance-sampled DMC mixed and extrapolated Bose coherence.

    python3 -m examples.bosons.run_projected_coherence --quick
    python3 -m examples.bosons.run_projected_coherence

The extrapolated off-diagonal quantity is NOT a pure ground-state estimator.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

from qmc.projected_coherence import ProjectedCoherenceConfig, run_projected_coherence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="small smoke-test scan")
    parser.add_argument("--output", default="outputs/bosons/projected_coherence")
    parser.add_argument("--scheme", choices=("metropolis", "second_order"), default="metropolis")
    args = parser.parse_args()
    cfg = replace(ProjectedCoherenceConfig(), scheme=args.scheme)
    if args.quick:
        cfg = replace(
            cfg, branches=("liquid",), n_particles=4,
        )
        cfg = replace(
            cfg, dt_values=(0.04,), walker_populations=(24,),
            equilibration_tau_values=(0.24, 0.48), production_tau=0.48,
            vmc_walkers=24, vmc_steps=36, vmc_burn_in=10,
            vmc_sample_every=3, n_radii=5, replicas=2,
        )
    run_projected_coherence(cfg, args.output)


if __name__ == "__main__":
    main()
