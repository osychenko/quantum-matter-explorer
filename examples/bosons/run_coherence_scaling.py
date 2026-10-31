"""Finite-size variational coherence and structural-order study.

    python3 -m examples.bosons.run_coherence_scaling --quick
    python3 -m examples.bosons.run_coherence_scaling

Results are trial-state VMC estimates, not pure-DMC condensate fractions or
thermodynamic proof of Bose-Einstein condensation / crystallization.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

from qmc.coherence_scaling import CoherenceScalingConfig, run_coherence_scaling


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Lightweight two-size smoke experiment")
    parser.add_argument("--output", default="outputs/bosons/coherence_scaling")
    args = parser.parse_args()
    cfg = CoherenceScalingConfig()
    if args.quick:
        cfg = replace(
            cfg,
            particle_counts=(4, 8), reduced_densities=(0.5,), gammas=(6.0,),
            jastrow_amplitudes=(0.3, 0.8), order_amplitudes=(0.0, 0.08),
            n_walkers=12, training_steps=24, validation_steps=36,
            burn_in=8, sample_every=4, replicas=2,
            n_radii=6, draws_per_configuration=1,
        )
    run_coherence_scaling(cfg, output_dir=args.output)


if __name__ == "__main__":
    main()
