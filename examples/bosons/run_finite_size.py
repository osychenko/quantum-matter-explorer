"""Reproducible finite-size and chain-length VMC robustness study.

Quick smoke experiment:
    python3 -m examples.bosons.run_finite_size --quick

Longer exploratory experiment:
    python3 -m examples.bosons.run_finite_size

Outputs are finite-N variational diagnostics, NOT a phase diagram.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

from qmc.finite_size import FiniteSizeConfig, run_finite_size_study


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Reduced-size regression/preview run")
    parser.add_argument("--output", default="outputs/bosons/finite_size", help="Output directory")
    args = parser.parse_args(argv)
    config = FiniteSizeConfig()
    if args.quick:
        config = replace(
            config,
            particle_counts=(8, 16), reduced_densities=(0.55,), gammas=(3.0,),
            jastrow_amplitudes=(0.3,), order_amplitudes=(0.0, 0.08),
            n_walkers=12, burn_in=10, training_sweeps=18,
            production_sweeps=(18, 36), sample_every=3,
            block_size=2, replicas=2,
        )
    training, replicas, summary, contrasts, lengths = run_finite_size_study(config, args.output)
    print(f"Wrote {len(training)} training points, {len(replicas)} replica runs, "
          f"{len(summary)} summaries, {len(contrasts)} start contrasts and "
          f"{len(lengths)} duration contrasts to {args.output}")


if __name__ == "__main__":
    main()
