# Patch 23 — Statistical and systematic-error audit of continuum worms

This patch adds **diagnostics**, not a new Hamiltonian or modified sampler.
It reuses Patch 22's `ReliableContinuumSpacetimeWorm` and its independently
computed `TwoBosonAdditionReference`. The physical observable is still the
canonical finite-temperature **particle-addition** Green function

\[
G_k^{(+)}(t)=\frac{1}{Z_1}\operatorname{Tr}_{N=1}
\left[e^{-(\beta-t)H_1}a_k e^{-tH_2}a_k^\dagger\right].
\]

It is **not** a condensate fraction, a pure ground-state off-diagonal density
matrix, or a large-N result. The background has one boson; the inserted
interval has two interacting bosons.

## The actual correlated ratio

For separation `ell` and momentum `k`, the Patch 22 conditional estimate is

\[
\widehat G=\frac{M-1}{\eta}
\frac{\sum_s p_s(G)\,{\bf1}_{\ell_s=\ell}
\cos(\mathbf k\cdot\Delta\mathbf r_s)}{\sum_s[1-p_s(G)]}.
\]

The numerator and denominator are measured on the **same correlated Markov
chain**. The code provides four complementary checks:

- **Contiguous-block jackknife of the complete ratio.** Recompute the ratio
  after leaving each time block out. This accounts for numerator/denominator
  covariance. It is only reliable if time blocks are longer than the relevant
  autocorrelation time; 8-block and 4-block SEMs are both recorded.
- **Ratio-influence effective sample size.** For each observable use
  \(h_s=cA_s-\widehat G B_s\), where \(A_s=p_s(G)1_{\ell_s=\ell}\cos(k\cdot\Delta r)\),
  \(B_s=1-p_s(G)\), \(c=(M-1)/\eta\). Estimate autocorrelation from `h`, not
  from the sector indicator alone. This FFT ESS is diagnostic, not a
  guaranteed confidence interval.
- **Independent-replica SEM.** Standard deviation of independent chain means
  divided by the square root of the number of chains. A few chains can produce
  deceptively small SEM. Do not add this to the block-jackknife SEM in
  quadrature: they use the same observations.
- **Split-Rhat of the pooled ratio-influence traces.** Center every chain
  around a *common* pooled-ratio value; centering each on its own estimate
  would conceal between-chain drift. This is conventional split-Rhat, not
  rank-normalized Rhat. Values near 1 do not prove good mixing.

The raw count-based and conditional/Rao–Blackwellized estimates are both
reported. They share the underlying Markov chain and are not independent
references.

## Separate systematic checks

- **Time-step bias at fixed inverse temperature:** `tau=beta/M`, for even
  `M`. All grids have `t=beta/2` exactly. Only matching physical separations
  and modes are compared. This avoids interpolation artifacts.
- **Run-length stability:** compare independently seeded short and long runs
  at identical `beta`, `tau`, interaction and measurement time.
- **Move-mixture sensitivity:** compare two *predeclared*, fixed configurations:
  `baseline` and `refresh_heavy`. No selection based on validation results.
- **Fourier-basis truncation:** independently calculate the interacting
  two-boson Green function with `K=1` and `K=2`. Their difference is a
  sensitivity diagnostic, **not a bound** on remaining basis/quadrature error.
- **Inverse temperature:** full mode uses two `beta` values. These change the
  *physical finite-temperature observable*. Do not pool different `beta`
  values as though they measured the same thermodynamic state.

`contrasts.csv` contains matched-physics differences between steps or run
lengths, using independent-replica SEMs. It is **not** a zero-time-step
extrapolation; two noisy time steps cannot justify an accurate extrapolation.

## Explicit warnings and their interpretation

`summary.csv` lists `short_protocol`, `low_ess`, `few_open_visits`, `rhat`,
`block_sensitivity`, `reference_tension` and `zero_replica_sem` when triggered.
The numerical cutoffs (ESS below 30, split-Rhat over 1.10, blocking SEM change
above 60%, two-replica-SEM tension with K=2) are **heuristics**. They do not
constitute a statistical test with controlled false-positive rates.

`inconclusive` is the expected status of the quick test.
`passes_screening_only` is **not** a statement of proven convergence.
Consider rare extreme-weight worm states and replica outliers when reviewing
ratios. Inspect traces and run longer independent chains before drawing any
physical conclusions.

## Run

Apply Patch 22 first, then from the project root:

```bash
python3 -m pytest -q
python3 -m examples.bosons.run_worm_audit --quick
python3 -m examples.bosons.run_worm_audit
```

Override the sampling budget without modifying code:

```bash
python3 -m examples.bosons.run_worm_audit \
    --steps 4000 12000 --replicas 6 \
    --links 4 6 8 --betas 2.4 \
    --output outputs/bosons/worm_audit_large
```

The full default comprises two temperatures, two time-step grids, two
interaction heights, two fixed move mixtures, two durations and three
independent replicas: **96 chains**. It is substantially more expensive than
quick mode. The independent spectral reference uses an optional one-thread
BLAS limit when `threadpoolctl` is installed; it remains functional without
that optional dependency.

Under `outputs/bosons/worm_audit/`:

- `replicas.csv` — each chain's normalized Green function, original raw ratio,
  jackknife error, ESS, open visits and process CPU time
- `summary.csv` — independent-replica SEM, ratio split-Rhat, spectral values,
  explicit screening flags
- `contrasts.csv` — independent run-length and fixed-beta time-step differences
- `spectral_reference.csv` — both K=1 and K=2 values for the same observables
- `protocol.json` — exact inputs and interpretation
- `timestep_audit.png`, `length_audit.png`, `ratio_ess.png` — diagnostics

### What is not established

The primitive action still has finite-timestep bias; the Fourier reference is
truncated. The host particle number is restricted to one, no larger bosonic
many-body result is established, and none of these tests prove BEC or
superfluidity. Patch 23 is intended to reveal unreliable estimates before
the sampler is generalized to more particles.
