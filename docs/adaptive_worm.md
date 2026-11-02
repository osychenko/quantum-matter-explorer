# Patch 26: frozen-pilot adaptation for continuum worm benchmarks

The aim is to assess *algorithm selection*, not to adapt an existing chain
while it is collecting physical measurements. Every candidate is one of the
fixed, reversible proposal kernels of Patches 22–25. All kernels sample the
same interacting finite-\(\tau\) canonical particle-addition Green function
for one background boson and one inserted boson in a periodic 3D box.

## What is compared

- **Baseline**: Patch 22 free-reference independent refresh.
- **Moderate guide**: Patch 25 mixture of free and repulsion-aware proposals,
  with evaluated density correction and free component of positive weight.
- **Four-try**: Patch 24 multiple-try independence Metropolis with a complete
  reverse set (not a greedy maximum-weight selection).
- **Strong guide**: a second fixed, stronger mixture; available in full mode.

The physical Hamiltonian, number of links, inverse temperature, fugacity,
reference normalization, and measured observable are identical for each arm
within a model/grid. Varying the proposal **does not** alter the Trotter
propagator or the bosonic exchange physics.

## Pre-registered two-stage protocol

For each interaction and imaginary-time grid:

1. **Pilot**: run each arm with independent seeds and a fixed step budget.
   Estimate the normalized \(G_{k=0}(\beta/2)\) via the *same* Patch 23
   block-jackknife ratio estimator. Record the ratio-influence ESS and process
   CPU time. Use the independent interacting \(K=2\) spectral reference from
   Patch 22, checking its \(K=1\) sensitivity separately.
2. **Screen**: a pilot arm is ineligible if any replica has ratio ESS below
   the predefined threshold or unstable block-scale uncertainty (four-block
   error more than three times the chosen block-jackknife error). A discrepancy
   from the spectral reference is **flagged**, not used to disqualify an arm:
   its size is part of the score.
3. **Select** one arm with the lowest pilot score, breaking ties by name.
   If none is eligible, select baseline with an explicit fallback reason.
4. **Freeze** the selected parameters. Run *every* arm, including the selected
   one and the baseline, on a fresh validation seed namespace. Never update
   the selected arm using validation results.
5. **Report** independent-replica mean and SEM, within-chain block-jackknife
   error, ESS per process CPU second, spectral deviation, and all failed runs.
   A pilot winner is **not** automatically a held-out winner.

The diagnostic pilot score, with all terms in consistent energy/Green units,
is

\[
 S = \overline{t_{\rm CPU}}\left[ (\overline G - G_{K=2})^2
       + {\rm SEM}_{\rm replica}^2 + \overline{\sigma^2_{\rm JK}}
       + (G_{K=2}-G_{K=1})^2\right].
\]

The score is a **heuristic**, not an unbiased estimator of integrated mean
squared error or a general convergence certificate. In particular, the
within-chain jackknife and between-chain SEM terms contain overlapping
uncertainty information and are intentionally conservative. It compares
algorithms at equal attempted steps while penalizing CPU cost. The held-out
assessment is the primary evidence of whether selection generalized.

**Avoid interpreting this selection procedure as physical calibration for an
unknown interacting system**: it deliberately trains against a known spectral
answer. When the reference is not known, a different pre-registered selection
criterion that does not use the unknown answer is needed. Similarly, agreement
with the continuum spectral reference at a single primitive time step does not
rule out cancellation of Monte Carlo bias and Trotter error.

## Running

```bash
python3 -m pytest -q tests/test_adaptive_worm.py
python3 -m examples.bosons.run_adaptive_worm --quick
python3 -m examples.bosons.run_adaptive_worm
python3 -m examples.bosons.run_adaptive_worm \
    --links 4 6 --pilot-steps 2000 --validation-steps 4000 \
    --pilot-replicas 4 --validation-replicas 6
```

The quick mode is an integration test using two pilot replicas and two held-out
replicas. Full mode evaluates two time grids with more independent chains;
neither mode automatically certifies equilibration. The `--include-ideal` flag
adds a free-boson control with its exact canonical reference.

## Outputs

All files are created under `outputs/bosons/adaptive_worm/` unless an explicit
`--output` directory is given:

- `pilot_runs.csv`: each independently seeded pilot chain, or its error.
- `pilot_summary.csv`: per-arm pilot summaries, eligibility, and scores.
- `selected.csv`: the **frozen** arm, reason, and protocol per model/grid.
- `validation_runs.csv`: all held-out arms on fresh seeds.
- `validation_summary.csv`: held-out accuracy, precision, and efficiency.
- `spectral_reference.csv`: independent \(K=1\)/\(K=2\) reference values.
- `protocol.json`: machine-readable candidates, fixed protocol, seed policy,
  Hamiltonian parameters, reference use, and limitations.
- `validation_accuracy.png` and `validation_efficiency.png`.

## Important limitations

This is still the **one-background-boson to two-boson** continuum insertion
benchmark, with finite periodic volume, finite temperature, finite primitive
\(\tau\), truncated spectral reference, and potentially slow-mixing G/Z sector
ratios. The new code introduces **no general N-body spacetime worm**, no
continuous online/adaptive MCMC during production, and no ground-state pure
OBDM estimator. Error bars from very short or metastable chains may be
misleading. Process CPU time depends on hardware, thread settings, and other
runtime load; rerun the pilot on the target machine before inferring a speedup.

The selection result is local to the chosen model and imaginary-time grid:
never assume a winning proposal transfers unchanged to a different box,
density, temperature, interaction, or particle number.
