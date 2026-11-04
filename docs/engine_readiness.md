# Patch 31 — Scientific readiness gate before the interface milestone

This patch is deliberately an **offline analysis and scope decision**, not a new
sampler. It addresses the latest Patch 30 run, where the M=12 exchange result
(2.136 ± 0.087) differed markedly from the finite-basis spectral K=2 value
(1.691); in the same run, the baseline gave 1.688 ± 0.180.

## Run with your existing results

```bash
python3 -m examples.bosons.run_engine_readiness \
  --replicas outputs/bosons/equal_cpu_trace_30/replicas.csv \
  --traces outputs/bosons/equal_cpu_trace_30/traces \
  --output outputs/bosons/engine_readiness_31
```

Results: `preset_readiness.csv`, `method_consistency.csv`,
`chain_influence.csv`, `decision.json`, `readiness_report.md`. The script never performs new sampling.
Omitting `--traces` permits preliminary CSV-only screening, but **no preset**
can be a trace-backed candidate in that mode. Do not combine old and new
benchmarks with different seeds, run lengths, or physical settings without
explicitly handling provenance.

## How to interpret the statuses

- `blocked`: at least one hard numerical screen failed (reference discrepancy,
  low reported ESS, high trace-based split-Rhat, or significant disagreement
  between the two samplers). Do not make a numerical fidelity claim for it.
- `needs_evidence`: no hard failure, but missing traces, inadequate precision,
  unresolved correlation tails or insufficient chain/block length.
- `candidate`: passed these *operational* screens; **not** a mathematical
  convergence certificate or engine approval.

Screens (initial engineering defaults): >=8 independent seeds; replica SEM
<=5% of estimated Green value; absolute discrepancy <=2 replica SEM relative
to the *finite-basis* spectral K=2 reference; reported minimum ratio ESS >=100;
rank-free split-Rhat for numerator, denominator and influence <=1.05; no
warning that the correlation window is unresolved; and block length at least
10 times estimated integrated autocorrelation time for each chain. A strong
cross-method disagreement (>2 combined replica SEM) blocks both methods on the
grid. These thresholds can be reviewed but must not be tuned after inspecting
results to manufacture a pass.

The `blocking_unresolved` test is conservative: the maximal block length
available in Patch 30's limited sweep may be too short even for an adequately
mixed chain. That is a reason for `needs_evidence`, not for assuming failure.

The script does **not** prove stationarity, independence, precise reference
accuracy or correctness for larger N. It doesn't evaluate Yukawa/DMC, many-N
bosonic phase claims, or actual phase transitions. The spectral K=2 reference
has finite basis truncation. `decision.json` intentionally always sets
`ready_for_engine_freeze=false`: Patch 32 must separately verify exact controls,
reproducibility, a stable API, performance and scientifically honest labels.

## Practical decision for the MVP

- Prototype a GUI now, but restrict the **certified numerical** presets to those
  independently evaluated in Patch 32.
- Treat M=12 exchange as a research/diagnostic option until the discrepancy is
  explained. Do not conceal the discrepancy by averaging it with the baseline.
- Allow precomputed, labeled illustrative outputs in the UI even when the
  underlying expensive research mode is not ready.
- Patch 32 is the readiness review and stable experiment/API contract,
  **not** a promise to make all methods numerically mature.

`chain_influence.csv` reports leave-one-chain-out changes in the ensemble mean and,
when traces are provided, first/last-half ratio estimates. Those split-half
estimates are **not independent replicates** and are not by themselves a
stationarity test. They help identify conspicuous temporal drift and chains
worth deeper investigation.
