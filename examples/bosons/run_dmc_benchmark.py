"""Run held-out VMC and two-factor DMC convergence diagnostics.

From repository root:
    python3 -m examples.bosons.run_dmc_benchmark --quick
    python3 -m examples.bosons.run_dmc_benchmark

The full run can be computationally demanding. Neither a fitted intercept nor
agreement of mixed and growth estimators establishes an exact ground-state E.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

from qmc.dmc_benchmark import DMCBenchmarkConfig, run_dmc_benchmark


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Small functional smoke run, not reliable convergence")
    parser.add_argument("--output", default="outputs/bosons/dmc_benchmark")
    parser.add_argument("--scheme", choices=("metropolis", "second_order"), default="metropolis")
    args = parser.parse_args()
    config = DMCBenchmarkConfig(schemes=(args.scheme,))
    if args.quick:
        config = replace(config, n_particles=4, reduced_density=0.5,
                         jastrow_amplitudes=(0.3,), order_amplitudes=(0., 0.05),
                         dt_values=(0.08, 0.04), walker_counts=(12, 20),
                         replicas=2, vmc_walkers=12,
                         vmc_steps=20, vmc_burn_in=6, vmc_sample_every=2,
                         vmc_block_size=2, warmup_sweeps=8,
                         equilibration_tau=0.4, production_tau=1.2,
                         block_tau=0.3, bootstrap_samples=20)
    run_dmc_benchmark(config, args.output)


if __name__ == "__main__":
    main()
