"""Explore short-range order in periodic soft-core bosons with variational VMC.

    python3 -m examples.bosons.run_phase_scan --quick
    python3 -m examples.bosons.run_phase_scan

The scan does not measure superfluidity or determine a thermodynamic transition.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

from qmc.phase_scan import PhaseScanConfig, run_phase_scan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Small fast verification scan")
    parser.add_argument("--output", default="outputs/bosons/phase_scan")
    args = parser.parse_args()
    cfg = PhaseScanConfig()
    if args.quick:
        cfg = replace(cfg, strengths=(3.0, 18.0),
                      jastrow_amplitudes=(0.3, 0.8),
                      order_amplitudes=(0.0, 0.08),
                      n_walkers=24, n_steps=50, burn_in=14,
                      sample_every=3, block_size=3,
                      validation_replicas=2, n_bins=15)
    run_phase_scan(cfg, output_dir=args.output)


if __name__ == "__main__":
    main()
