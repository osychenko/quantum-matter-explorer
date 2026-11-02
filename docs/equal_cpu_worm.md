# Patch 28 — pilot-calibrated equal-CPU worm benchmark

## Purpose and scope

This benchmark compares the existing Gibbs-sector continuum worm (Patch 22)
with interaction replica exchange (Patch 27). The physics is unchanged:
one ideal background boson, interacting N=2 insertion, soft-core repulsion,
fixed inverse temperature. Only the **cold** replica supplies measurements.

## Correctness rules

1. **Independent calibration**: run short disposable pilots with distinct RNG
   seeds to estimate process CPU seconds per `step()`. Each method gets a
   method-specific *fixed number of validation steps*, determined before any
   measurement. CPU timing is not used to decide when to terminate a chain.
2. **Actual CPU times** include the physical worm, the auxiliary worm for
   replica exchange, and the audit calculations. Pilot timing measures just
   transition steps; therefore equality of actual validation CPU budgets is
   approximate. `cpu_ratio` flags deviations larger than 35%.
3. **Independent replicas**: each validation chain starts from a fresh state
   and uses an independent seed. Do not treat within-chain measurements as
   independent replicates.
4. **Ratio uncertainty**: `audit_worm_chain` computes a block jackknife of
   the numerator/denominator ratio and its influence-function ESS. The summary
   also reports between-replica SEM and the largest individual absolute
   influence contribution. Neither statistic guarantees that rare tails were
   sampled adequately.
5. **Reference**: independent truncated two-boson Fourier calculation (`K=1`
   for fast execution), not an exact continuum result. Do not conflate
   basis error, primitive-action timestep error and Monte Carlo error.
6. **No automatic winner**: `screen_pass_not_proof` means only that no simple
   warning threshold fired. `inconclusive` flags low ESS, rare-event
   concentration, unstable blocking, CPU mismatch or reference discrepancy.
   A reliable physical conclusion needs longer chains, larger basis and
   repeated validation.

## Run

```
python3 -m pytest -q tests/test_equal_cpu_worm.py
python3 -m examples.bosons.run_equal_cpu_worm --quick
python3 -m examples.bosons.run_equal_cpu_worm --cpu-seconds 10 --replicas 6 --links 4 6
```

Output: `pilots.csv`, `replicas.csv`, `summary.csv`, `spectral_reference.csv`,
`protocol.json`, `accuracy.png`, `efficiency.png`.

Short quick mode is only a smoke test. Its small CPU budgets may force the
minimum chain length and are not evidence of equal or sufficient CPU effort.
