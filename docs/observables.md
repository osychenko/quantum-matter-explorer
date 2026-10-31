# Production observables and visualization (Patch 5)

The numerical estimators accept batched configurations `(n_walkers, n_particles, 3)`
and can be accumulated from **both VMC and DMC production steps**:

```python
from qmc.observables import PairCorrelation, StaticStructureFactor, reciprocal_index_grid

g = PairCorrelation(cell, n_particles=8, n_bins=30)
s = StaticStructureFactor(cell, n_particles=8, indices=reciprocal_index_grid(1))

result = run_vmc(potential, trial, vmc_config, observables=(g, s))
# Or:
# result = run_dmc_importance(
#     potential, trial, dmc_config, observables=(g, s), observable_stride=10
# )

g_data = g.result()
s_data = s.result()
```

**Pass fresh observable objects for each simulation run.** The accumulators retain
all previous batches until you create new instances; they are independent of the
VMC and DMC `Result` dataclasses. The VMC driver observes at `sample_every`
intervals after burn-in. The DMC driver observes every `observable_stride`
steps after equilibration, *after fixed-population resampling*. Hooks do not
consume random numbers or modify walkers.

## Pair-correlation normalization

We count each unordered pair exactly once, and normalize each radial shell by

\[
N_{\mathrm{config}}\,\frac{N(N-1)}{2}\frac{4\pi(r_{i+1}^3-r_i^3)}{3\,V}.
\]

This finite-`N` normalization gives `g(r)=1` in expectation for independent,
uniform particles. Shells must fit entirely inside the Wigner--Seitz cell.
`PairCorrelation` defaults to the cell inradius; for general skew lattices it
uses a conservative lower bound. **Do not extend the histogram beyond this
radius** without geometric shell-volume corrections.

## Structure factor

For nonzero reciprocal modes `k`,

\[
S(\mathbf{k}) = \frac{1}{N}\left\langle\left|\sum_j
 e^{i\mathbf{k}\cdot\mathbf{r}_j}\right|^2\right\rangle.
\]

`StaticStructureFactor` keeps each reciprocal direction separate. The zero
mode is excluded because its raw value is always `N`. For independent uniform
particles, nonzero modes have `S(k)=1` in expectation. The scatter plot uses
`|k|` on its horizontal axis **without averaging across equivalent lengths**.
Direction-resolved peaks are often more informative about crystalline order.

## Physics and statistics caveats

- A VMC estimate samples `|Psi_T|²`. DMC walkers sample a **mixed distribution**
  (approximately `Psi_T * Psi_0`), so diagonal pair/structure observables from
  DMC are *mixed estimators* and need not equal pure ground-state expectations.
- Successive sweeps and resampled DMC walkers are correlated. These classes
  intentionally give **no naive error bars**. Reliable uncertainty estimates
  require independent runs or appropriate time-block/replica analysis.
- The plotted 3D configuration is a single Monte Carlo configuration, not
  a classical trajectory. Bosons are indistinguishable, even if dots are
  drawn for visualization.
- The BCC cell snapshot uses the actual truncated-octahedral Wigner--Seitz
  boundary; the general cell snapshot uses a primitive parallelepiped.
- Neither `g(r)` nor `S(k)` measures superfluidity or condensate fraction.
  OBDM/BEC and winding-number/helicity-response methods require separate work.

Run the lightweight example from the project root:

```bash
python3 -m examples.bosons.run_observables
```

The example writes `outputs/bosons/cell_snapshot.png`, `pair_correlation.png`,
and `structure_factor.png`.
