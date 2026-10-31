# Patch 10 — finite-size variational coherence and spatial order

This patch studies the existing periodic, repulsive soft-core boson model using
**variational Monte Carlo (VMC)**. It does **not** evaluate pure DMC off-diagonal
correlations, demonstrate thermodynamic Bose–Einstein condensation, or establish a
liquid–crystal phase boundary.

## Physical protocol

For each particle count `N`, reduced density `rho* = (N/L³) rc³`, and dimensionless
interaction strength `gamma = V0 rc² / D`, use a cubic simulation cell with

    L = rc (N / rho*)^(1/3) ,  V0 = gamma D / rc².

The cutoff is required to be `rc <= L/2`. Keeping both `rho*` and `gamma`
fixed while increasing `N` defines the finite-size sequence. The VMC trial is

    log Psi = -a sum_(i<j) (1 - rij²/rc²)_+³
              + lambda/(2N) sum_G |sum_j exp(i G.rj)|² .

For each `(N, rho*, gamma)`, independent training chains select `a` and
`lambda` separately for two families: **liquid-like** (`lambda=0`) and
**modulated** (`lambda>0`). The latter favors selected density modes but may
represent **cluster ordering**, not a single-occupancy crystal.

The selected reciprocal vectors are the three axial directions with the integer
mode index nearest a fixed target `|G| rc`. Integer reciprocal vectors are
quantized by the periodic box, so exact `|G| rc` **cannot** be held constant
for every `N`. `summary.csv` explicitly records `mode_index`, actual
`k_cutoff = |G|rc`, and `target_k_cutoff`. Differences in these quantities
must be considered when comparing apparent size trends.

After training, each candidate is measured on fresh independently seeded VMC
runs; the training samples do **not** enter validation statistics. We collect:

- **Finite-cell zero-momentum occupation fraction** `n0_over_n`: the full-cell
  displacement integral of the normalized one-body density matrix, as
  implemented in Patch 9. This is not inferred from the last point of `g1(r)`.
- **Radial `g1(r)`**: sampled for `0 <= r <= L/2` with isotropic orientations.
  `g1.csv` gives both `r/rc` (fixed physical length scale) and `r/L`
  (fraction of the cell), as well as the between-replica standard error.
- **Selected-mode `S(G)/N`**: mean over the preselected axial directions,
  divided by `N`. This is a spatial density-modulation diagnostic, not a
  complete crystal identification or a superfluidity estimator.
- VMC energy per particle and sweep acceptance for quality control.

Each `replica` in `replicas.csv` has its own sampling, initialization and
observable random seeds. Standard errors in `summary.csv` are
`std(replica_estimates, ddof=1)/sqrt(n_replicas)`, not statistics over
correlated walkers or sweeps. A small number of replicas produces an
uncertain error estimate; explore more seeds, equilibration lengths, initial
conditions and increased displacement draws before claiming differences.

## Run

```bash
cd ~/quantum-matter-explorer
python3 -m pytest -q
python3 -m examples.bosons.run_coherence_scaling --quick
python3 -m examples.bosons.run_coherence_scaling
```

The full experiment uses `N=(8,16,32)`, `rho*=0.65`, `gamma=(3,18)` and three
validation replicas, with independently selected trial parameters for each
point. It is intentionally more expensive than `--quick`. Choose a custom
output directory with `--output PATH`.

Outputs are written under `outputs/bosons/coherence_scaling/`:

- `training.csv` — every candidate and its training energy;
- `replicas.csv` — independently held-out VMC estimates and seeds;
- `summary.csv` — means and replica SEMs for energy, `n0/N` and `S(G)/N`;
- `g1.csv` — radial coherence and replica SEMs in both normalized coordinates;
- `protocol.json` — all configuration parameters and estimator caveats;
- `scaling.png` — finite-N trends versus `1/N` (no extrapolated intercept);
- `coherence_vs_structure.png` — paired coherence/order diagnostics;
- `g1_scaling.png` — radial coherence from each finite cell.

## Interpretation: what would be evidence, and what would not?

The ideal noninteracting zero-temperature Bose ground state has `n0/N=1`.
Interactions can deplete this occupation. A nonzero **finite-cell** `n0/N`
is expected even in settings where its thermodynamic limit may vanish.
**Two or three particle counts do not justify a reliable `N->infinity`
extrapolation.** In particular, `n0/N` and `S(G)/N` both have finite-size
and commensurability effects. Comparing their finite-size trends is more
informative than labeling either a categorical phase indicator.

The selected variational families are trial functions, not independent phase
measurements. A lower variational energy in one family is *not* proof that the
exact ground state has that phase; poor optimization or long-lived metastable
states can bias the result. The trial's density-wave term retains bosonic
exchange symmetry and is not a diagnostic of nonzero superfluid response.

Next independent validations would include broader `N` and density scans,
multiple starts, autocorrelation-aware sampling, independent mode families,
and pure (not mixed) DMC or path-integral estimators for off-diagonal order.
