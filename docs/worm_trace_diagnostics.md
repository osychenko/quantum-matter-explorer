# Patch 30 — Cold-chain measurement traces and convergence diagnostics

This patch adds optional *measurement* persistence. It does not change the
Hamiltonian, proposal transition probabilities, swap acceptance ratios, or
Green-function estimator. Per-chain compressed `.npz` files retain the
**midpoint, zero-momentum** Rao–Blackwellized numerator and its **same-chain**
denominator, plus reproducibility metadata (method, seed, imaginary-time grid,
beta, fugacity-derived scale and full-chain Green estimate). Other modes and
time separations are not saved. Trace files use `numpy.load(allow_pickle=False)`.

## Run on an 8-CPU / 15-GiB Ryzen host

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python3 -m examples.bosons.run_equal_cpu_worm \
  --links 8 12 --cpu-seconds 45 --replicas 8 --workers 6 \
  --pilot-steps 2000 --calibration-factor 1.25 --seed-offset 2000000 --save-traces \
  --output outputs/bosons/equal_cpu_trace_30
python3 -m examples.bosons.run_worm_trace_diagnostics \
  --traces outputs/bosons/equal_cpu_trace_30/traces \
  --output outputs/bosons/worm_trace_audit_30
```

The analysis writes `trace_summary.csv`, `blocking_curves.csv`,
`autocorrelation.csv`, `running_green.png`, `blocking_curves.png`,
`autocorrelation.png`, and `protocol.json`. Pilot chains are **not** saved.
Existing old CSV results cannot recover raw time traces: rerunning with
`--save-traces` is required.

## Statistical interpretation

For each measurement at fixed separation and mode, the ratio estimator is

`G = scale * sum(numerator) / sum(denominator)`.

Ratio influence at a *common pooled estimate* is

`I_t = scale*numerator_t - G_pooled*denominator_t`.

Using a common pooled estimate prevents the erroneous disappearance of
between-chain mean differences that would occur if each chain were centered
separately. We calculate rank-free, **split** Rhat for influence, numerator,
and denominator. This is not rank-normalized Rhat and not a convergence proof.
Do not combine trace files from different physical models, fugacities, beta or
M, or duplicate seeds. Comparisons between the baseline and replica exchange
must be made as *separate* targets/sampling groups.

The autocorrelation estimate uses FFT-based lag covariance with an
initial-positive-pair summation and a 5-tau self-consistent window, subject
to a maximum lag. A `window_at_limit` warning indicates an unresolved tail;
small or unreported warnings do not rule out very long autocorrelation.

`blocking_curves.csv` contains jackknife SEM versus **contiguous block
length**, not solely number of blocks. A plateau is desirable but not a
convergence certificate. Running plots display at most three replicas per
group, while all chains contribute to numerical diagnostics.  Results are
sensitive to measurement cadence and available production length.

## Patch 29 integration

The Patch 29 fit CSV retains the existing reference fields. Patch 30 fixes
missing `spectral_reference` in `continuum_fits.csv` by calculating the
independent continuum K=2 reference for the same hardcoded two-boson model.
This remains a finite-basis Galerkin value, not exact continuum physics.
