# Patch 17 — Explicit bosonic permutation cycles and CPU efficiency

Patch 17 adds **two distinct experiments**. They deliberately use different
quantum statistical ensembles and their results must **not** be combined as
though they came from the same distribution:

1. **Closed-worldline canonical PIMC at finite temperature** to sample the
   Bose permutation sum and measure exchange-cycle frequencies.
2. **Open-path, fixed-projection PIGS** to tune Patch 16's staging/bisection
   mixture by effective samples **per process CPU second**, using held-out runs.

Neither experiment implements a worm/off-diagonal sector, a pure interacting
one-body density matrix, condensate fraction, or a superfluid-fraction
estimator. Boson exchange frequency by itself proves none of these.

## Canonical finite-temperature exchange ensemble

For `N` identical bosons, `M` time slices, `tau = beta/M`, and a periodic 3D
orthorhombic cell, the primitive-action partition-function integrand is

\[
\pi(X,P) \propto
\left[\prod_{s=0}^{M-2}\prod_{i=1}^{N}
K_{\tau}(\mathbf r_{si},\mathbf r_{s+1,i})\right]
\left[\prod_{i=1}^{N}
K_{\tau}(\mathbf r_{M-1,i},\mathbf r_{0,P(i)})\right]
\exp\left[-\tau\sum_{s=0}^{M-1}V(R_s)\right].
\]

Here `P` is a **permutation closing the polymer strands**. Summing over all
permutations and integrating the bead coordinates is the canonical Bose trace,
up to an overall `1/N!` independent of the sector. The periodic free kernel
`K_tau` is the full image sum from Patch 14; the potential is the periodic
many-body pair interaction from Patch 4. The diagonal primitive action has
finite-`tau` bias in interacting models.

Reassigning *every bead's labels* in an open PIGS path does not generate a
physical permutation sector: the labels can simply be gauged away. For that
reason we introduce a separate closed-time ensemble. Its `beta` is **inverse
temperature**, whereas in Patches 15–16 the `beta` parameter denotes
**projection time on each side** of the path. These are not equivalent.

### Moves and detailed balance

- **Swap:** transpose two *closure targets* `P(i)` and `P(j)`. Only two
  free-kernel closing links change; the interaction action cancels.
- **Single bead:** wrapped symmetric Gaussian displacement. Update the two
  connected links, including the correctly permuted first/last boundary link.
- **Interior bridge:** replace the beads of a strand between fixed endpoints
  using an exact periodic Brownian bridge. Its free kinetic proposal ratio
  cancels the free part of the target ratio; acceptance depends only on the
  interior potential-action difference, `-tau * delta(sum V)`.
- **Global translation:** translate all particles and beads together; the
  full action ratio is used, so non-translation-invariant test potentials
  remain valid.
- **Free Bose refresh:** choose a complete permutation with exact noninteracting
  sector probability proportional to `prod_cycles Z_1(length*beta)`; sample
  each cycle as a periodic Brownian loop. This independent proposal is the
  normalized *free bosonic* path measure. In the interacting Metropolis ratio
  it cancels all free kinetic factors, leaving only `-tau * delta(sum V)`.
  For zero potential refreshes accept with probability exactly one.

The full-refresh proposal enumerates permutations (`N!`) and is limited to
`2 <= N <= 5`. This is an exact **small-particle diagnostic**, not a scalable
large-N exchange algorithm. The free kinetic image sums and periodic bridge
winding sums have controlled Gaussian tail truncation. The reciprocal
single-particle partition function uses a Poisson-resummed form at small beta
so high temperatures do not require huge reciprocal grids.

### An independent analytic free-boson benchmark

For `N=2` noninteracting bosons,

\[
P_{\mathrm{exchange}}(\beta) =
\frac{Z_1(2\beta)}{Z_1(\beta)^2+Z_1(2\beta)},\qquad
Z_1(\beta)=\sum_{\mathbf k}e^{-\beta D|\mathbf k|^2}.
\]

This probability is *not* generically `1/2`; it depends on beta, the cell
volume and kinetic coefficient. Patch 17 tests the analytic formula,
free-refresh acceptance and the sampled exchange frequency. For interacting
systems there is no comparable simple closed form. The exchange statistic in
the CSV is the fraction of sampled permutations with at least one cycle of
length greater than one; for `N=2` this is the transposition probability.

Short single-chain standard errors are not appropriate for this observable.
The script writes per-replica exchange estimates and uses standard error
across independent replicas; with only 2 quick replicas this is very noisy.
No precise thermodynamic predictions should be drawn from `--quick`.

## CPU-normalized open-path move comparisons

Patch 16 provided an FFT integrated autocorrelation estimator for midpoint
potential samples:

\[
N_{\mathrm{eff}}=\min\!\left(N_{\mathrm{samples}},
\frac{N_{\mathrm{samples}}}{2\tau_{\mathrm{int}}}\right).
\]

The new comparison reports `N_eff / process_cpu_seconds`. Unlike comparing
`N_eff` alone, this includes the extra computational cost of constructing
staging/bisection bridges. It also records the raw per-chain mean, chain
stderr, tau_int, CPU seconds, and per-move acceptance. It times **the same
midpoint-potential measurement workload** for every mixture rather than
timing expensive unrelated estimators inside some runs.

The pilot tests candidate bridge probabilities (including zero) with
independent seeds. The candidate with the best observed pilot *mean* ESS/CPU
is compared with baseline using new seeds in held-out validation runs. The
selection is not a universal optimization: results depend on chain length,
observables, box, hardware, Python/NumPy builds and CPU contention. CPU time
is genuinely measured, not deterministic or comparable across machines. The
positive-pair autocorrelation-window estimator can be unstable for short or
metastable trajectories; a tiny number of replicas is only a smoke test.

## Run

From the repository root:

```bash
python3 -m pytest -q
python3 -m examples.bosons.run_boson_permutations --quick
python3 -m examples.bosons.run_boson_permutations
```

`--output PATH` overrides `outputs/bosons/boson_permutations/`.

The files are:

- `permutation_replicas.csv` and `permutation_summary.csv`: independent
  finite-temperature, `N=2` exchange and midpoint-potential statistics;
- `efficiency_pilot.csv` and `efficiency_validation.csv`: separate
  open-PIGS mixing measurements with measured CPU time;
- `protocol.json`: configurations, seed provenance and ensemble distinctions;
- `exchange_probability.png`: exchange comparison with the analytic *free*
  reference (the line is not an interacting reference);
- `cpu_efficiency.png`: pilot and held-out ESS/CPU values.

## Next scientific milestones

The next major method requires an interacting **off-diagonal (worm/open-line)
sector** with explicit relative sector normalization and a well-defined
one-body density-matrix estimator. Exchange cycles and diagonal finite-T
PIMC statistics are necessary stepping stones but cannot be substituted for
that estimator. Start by validating a worm implementation in a free,
finite-N periodic box with exact momentum occupations; only then add
interactions and perform timestep, thermal/projection, size and sampling
convergence studies.
