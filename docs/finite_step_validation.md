# Patch 29 — projected primitive transfer and replica-based analysis

**Scope and limitations.** For N=1→2 in the cubic soft-core box, the new deterministic reference computes the midpoint Green function from a **truncated symmetric two-particle plane-wave transfer matrix**:

\[ T_{K,\tau} = e^{-\tau K_2/2} P_K e^{-\tau V} P_K e^{-\tau K_2/2}. \]

The matrix of `exp(-tau V)` is evaluated using periodic FFT quadrature and Bose-symmetrized two-particle momentum pairs. The `P_K` insertion at *every time slice* means this is **not an exact finite-timestep coordinate-space worm reference**. Only in the converged momentum-basis and quadrature limits should the two primitive constructions agree. `projected_primitive.csv` therefore reports both K=1 and K=2; their difference is a truncation diagnostic, **not a guaranteed bound**. The existing continuous-time spectral code remains independent.

To analyze a completed Patch 28.2 experiment:

```bash
OPENBLAS_NUM_THREADS=1 python3 -m examples.bosons.run_finite_step_validation \
  --summary outputs/bosons/equal_cpu_convergence_282/summary.csv \
  --replicas outputs/bosons/equal_cpu_convergence_282/replicas.csv \
  --output outputs/bosons/finite_step_29
```

The command writes `projected_primitive.csv`, `replica_diagnostics.csv`, `continuum_fits.csv` and `protocol.json`. For just two available grids, no continuum fit is made. For three or more, exploratory inverse-variance weighted linear fits in tau and tau^2 are computed. With four or more grids, fits excluding the coarsest grid are also reported. Standard errors are **formal** and assume independent grid estimates and the specified fit power, which may not hold when previous runs reused seeds.

Only CSV summaries and per-chain uncertainties are available from Patch 28.2: **full measurement traces were not saved**. Consequently Patch 29 cannot retrospectively perform split-Rhat or a true blocking-length sweep. It computes between-replica SEM, rms within-chain eight-block SEM, and median four/eight/sixteen block ratios as a limited cross-check. A future data-collection patch should persist decorrelated block aggregates or capped traces before stronger within-chain diagnoses can be claimed.

For meaningful continuum extrapolation, gather M=4,6,8,12 with separate seeds and nonoverlapping production runs, compare K truncations, inspect fitted intercept and leave-coarse-grid sensitivity, and do not select a preferred power on the same samples without reporting model uncertainty.
