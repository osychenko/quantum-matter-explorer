# Patch 28.2: convergent-ratio audit and CPU-budget separation

This change leaves the physical interacting primitive action, Monte Carlo kernels,
random-seed allocation and Rao–Blackwellized Green-function ratio unchanged.

## Blocking diagnostic

`run_validation` now uses 8 jackknife blocks for adequately long chains
(and 4 for legacy short smoke tests), then recomputes the **same ratio**
from its stored time-ordered numerator and denominator with 4, 8 and,
when at least 128 measurements exist, 16 blocks. These are different
partitions of **one** chain, not independent error replicates.

`replicas.csv` contains `block_sem_4`, `block_sem_8`, `block_sem_16`,
`block_scales`, and `block_stability` = maximum nonzero SEM / minimum
nonzero SEM across available scales. A missing/degenerate comparison yields
`NaN` and flags the chain as `blocking_unstable`, rather than silently
reporting a perfect 1.0. The original per-chain jackknife and independent
replica SEM retain their different meanings. Finite block-count stability
alone does not prove convergence.

## Statistical versus operational warnings

`summary.csv` now distinguishes `status` and `flags` (statistical/reference
screening) from `cpu_status` and `cpu_flags` (CPU-target screening).
For example a physical `screen_pass_not_proof` can coexist with
`cpu_status=budget_mismatch`. The latter does not imply physical bias; actual
CPU time and ESS/CPU always use measured worker process time.

## Calibration

A longer, serial, independent pilot measures the full audit workload.
The `--calibration-factor` option (default 1.25) divides the planned
number of validation steps by that explicitly recorded correction factor.
It is a **heuristic** based on previous observed 1.2–1.5 target overshoots,
not an exact calibration guarantee. Adjust it only using pilot runs, not
by conditioning on Green-function values. Fixed validation lengths and seeds
are maintained independently of any subsequent CPU timing.

## Spectral reference

`run_equal_cpu_worm` now computes the independent two-boson spectral
reference once at `k_max=1` and `k_max=2`, using the existing numerical
spectral method and its 16-point grid; both values are recorded in
`spectral_reference.csv` and `protocol.json`. The plotted and audited
reference is `K=2`. Both are **finite-basis numerical references**, not
exact continuum values. The CSV row `spectral_reference` identifies K=2.

## Suggested Ryzen 9 5900HX configuration

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python3 -m examples.bosons.run_equal_cpu_worm \
  --links 6 8 --cpu-seconds 60 --replicas 12 --workers 6 \
  --pilot-steps 3000 --calibration-factor 1.25 \
  --output outputs/bosons/equal_cpu_282
```

Use a separate `--output` directory to preserve Patch 28.1 data.
