# Patch 24 — rare-configuration diagnostics and reversible multiple-try sampling

This **additive** patch targets the same finite-temperature, canonical
particle-addition Green function as Patches 21–23. The supported system has
`N_background` 0 or 1 in an orthorhombic 3D periodic cell, with at most two
bosons inside the insertion interval. Nothing here is a pure ground-state
one-body density matrix, a condensate-fraction estimator, or a general-N worm.

## The reference density and extended target

The augmented Patch 22 extended ensemble is

\[
\pi(x,z)\propto q(x) [\mathbf{1}_{z=Z}+R(x)\mathbf{1}_{z=G}],
\qquad x=(\mathrm{host},\mathrm{cut},\ell,\mathrm{extra}).
\]

Here `q` is the **normalized**, explicitly sampleable free reference law:
closed N=0/1 host, uniformly chosen cut and length, and an independent open
free extra strand. `R` is `log_relative_weight` exponentiated, and includes
sector fugacity `eta`, the symmetrized two-boson free propagator and the
primitive interaction action. The Gibbs sector move and Rao–Blackwellized
Green estimator from Patch 22 are retained unchanged.

## Exact independent multiple-try Metropolis

In sector G, independently propose \(y_1,\ldots,y_K\sim q\). Choose one
\(y_*\) with probability \(R(y_*)/\sum_iR(y_i)\). Independently draw
\(K-1\) further reference states and add the **old** state \(x\) to form
the reverse set. Accept the chosen proposal with

\[
\alpha(x\rightarrow y_*)=\min\left(1,
\frac{\sum_{i=1}^{K}R(y_i)}{R(x)+\sum_{j=1}^{K-1}R(x_j')}\right).
\]

This is the independence multiple-try Metropolis detailed-balance correction.
The target remains \(q(x)R(x)\); no reweighting of final worm measurements or
unknown normalizer is introduced. **Selecting the largest R without the
reverse-set correction would be biased.** K=1 is standard independent
Metropolis. The number of R evaluations per K-try step is \(2K-1\), so an
improvement in acceptance does **not** imply an improvement in CPU efficiency.

`multiple_try_probability=0` recovers the Patch 22 global-refresh kernel;
`multiple_try_probability=1` replaces all open-sector global refreshes with
multiple-try steps. All other moves and sector scheduling remain unchanged.

## Independent stratified importance benchmark

A separate estimator draws each interval length \(\ell\) independently from
its **free q-reference law**. This is possible here because the N=0/1
background has an exactly normalized free closed distribution:

\[
G_k^{(+)}(\ell\Delta\tau)=
\mathbb E_{q(\cdot\mid\ell)}\left[
\frac{R(x)}{\eta}\cos(k\cdot[r_\mathrm{head}-r_\mathrm{tail}])
\right].
\]

This identity gives an independent non-MCMC cross-check of both the
normalization and rare-event behavior. It is *not* automatically precise:
large importance weights may be extremely rare and unobserved even in an
apparently well-behaved independent sample. The benchmark therefore reports
top-1% absolute contribution concentration and uncertainties rather than
asserting convergence.

## Benchmark metrics

- `replicas.csv`: per-chain time-resolved Green values, raw-sector estimates,
  ratio influence-series ESS, and jackknife uncertainty
- `diagnostics.csv`: mid-time ESS per process CPU-second, elapsed process CPU
  time, acceptance, count of MTM proposals, and top-1% contribution share
- `independent_reference.csv`: independent stratified q results per interval,
  with standard errors and contribution concentration
- `summary.csv`: independent-replica Green mean, SEM, split-Rhat, and
  approximate spectral-reference discrepancy screens
- `spectral_reference.csv`: independent two-body Fourier reference at K=1,2
- `green_reference.png`, `rare_efficiency.png`, and `protocol.json`

The single-chain ESS estimate and split-Rhat are **screens**, not guarantees.
Short chains may miss a high-weight state entirely; split-Rhat may be near one
while the estimate is biased. Screening labels remain `inconclusive` or
`reference_discrepancy`, never `converged`. Timestep errors, basis truncation,
finite-temperature behavior, finite volume, and Monte Carlo error are distinct.

## Commands

```bash
python3 -m pytest -q tests/test_rare_worm.py
python3 -m examples.bosons.run_rare_worm --quick
python3 -m examples.bosons.run_rare_worm
python3 -m examples.bosons.run_rare_worm --steps 4000 --replicas 6
```

**Interpretation:** Patch 24 introduces a reversible candidate-selection
technique and an independent control estimator, **not a guaranteed speedup**.
Only longer independent-chain and independent-q experiments demonstrating
agreement with the interacting spectral reference, stable uncertainty versus
run length, and adequate CPU-adjusted efficiency can support a performance
claim. A larger particle count requires an extended physical sampler and a
new reference ensemble; it is not provided by this patch.
