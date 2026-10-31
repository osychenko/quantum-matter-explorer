# Patch 16 — Periodic staging, bisection, autocorrelation, and an interacting reference

This patch extends the **diagonal** primitive-action continuum PIGS method of
Patch 15. It does not introduce finite-temperature permutation cycles, a worm
sector, a *pure off-diagonal* estimator, or a ground-state condensate fraction.
The spatial coordinate system is a 3D **orthorhombic periodic cell**. One
labelled particle is modified in each bridge move; the trial wavefunction may
be exchange symmetric, but the sampler does not explicitly exchange worldlines.

## Free conditional bridge proposal

Select a particle and segment beads `a,...,b`, with fixed wrapped endpoint
coordinates `x_a`, `x_b` and `m=b-a` time steps of length `τ`. The free path
weight conditioned on these endpoints is

\[
q(X_{a+1:b-1}\mid x_a,x_b) =
\frac{\prod_{j=a}^{b-1} K_\tau(x_j,x_{j+1})}
{K_{m\tau}(x_a,x_b)}.
\]

The **periodic** bridge is not an ordinary Euclidean Brownian bridge from the
minimum-image endpoint difference. First select an integer winding in each
cell direction with weights

\[
\Pr(n_d\mid x_a,x_b)\propto
\exp\left[-\frac{(\delta_d+n_d L_d)^2}{4D m\tau}\right],
\]

where `δ_d` is the minimum-image endpoint difference. Conditional on that
unwrapped endpoint, generate an ordinary free Brownian bridge with covariance
`2 D τ [min(i,j)-i*j/m]` per Cartesian coordinate. The implementation offers:

- **Staging:** generate free Gaussian increments and condition their sum.
- **Bisection:** recursively sample conditional Gaussian midpoints between
  fixed unwrapped endpoints, including non-power-of-two segment lengths.

Winding probabilities use the same eight-Gaussian-standard-deviation image
truncation convention as Patch 14's periodic free propagator. This neglects
~1e-15 tails at typical parameter ranges; it is not algebraically identical
to an infinite image sum at machine precision.

The proposal reversibility identity gives

\[
\log\frac{P(X')q(X\mid X')}{P(X)q(X'\mid X)}
= -\tau\sum_{j=a+1}^{b-1} \big[V(R'_j)-V(R_j)\big].
\]

Thus **all free kinetic links cancel**, and the endpoints also retain the same
trial factors. The segment length, location, particle label, and method are
selected from state-independent distributions. Their proposal factors also
cancel. In the free or constant-potential limit, bridge acceptance is exactly
one (though this alone does not guarantee that the chain mixes globally).

The new `StagedInteractingPath` subclass leaves Patch 15's original
`PrimitiveInteractingPath` unchanged. It combines staging/bisection with the
older single-bead, whole-path refresh, and collective-translation updates.
Default bridge lengths are 2, 4, and 8 links; choose lengths no larger than
`n_links`. Setting `bridge_probability=0` recovers the original move mixture.

## Autocorrelation and sampling precision

For a scalar production series, the integrated autocorrelation time is

\[
\tau_{\mathrm{int}}=\frac12+\sum_{k\ge1}\rho(k),\quad
N_{\mathrm{eff}}\approx\frac{N_{\mathrm{measurements}}}{2\tau_{\mathrm{int}}}.
\]

`autocorrelation_estimate()` uses an FFT autocovariance, an initial positive
pair window, and a self-consistent cutoff. Its reported within-chain standard
error is `sqrt(variance / N_eff)`. The lag index counts **recorded measurements**,
not attempts or imaginary-time slices. A constant series has zero uncertainty
and `N_eff=N`. The window is heuristic, particularly when the chain is short
or has rare transitions between metastable basins. Independent seeded replicas
are therefore also retained; `summary.csv` uses the **between-replica SEM**.

`effective_samples.png` compares ESS **per equal number of Monte Carlo move
attempts**, not ESS per CPU second. Multi-bead moves typically cost more per
attempt. High acceptance or ESS from two quick replicas is not evidence of
an efficiency gain. Monitor both uncertainty and cost before optimizing move
probabilities or segment lengths.

## Independent interacting reference: N=2 in 3D

For two identical bosons with a periodic central soft-core interaction, define
relative coordinate `r=r_2-r_1` and the uniform center-of-mass sector. Then

\[
H_{\mathrm{rel}}=-2D\nabla_r^2+V(r),\quad
V(r)=V_0\max\!\left(0,1-\frac{r^2}{r_c^2}\right)^3.
\]

The reference code `qmc/two_boson_reference.py` constructs a finite
Fourier–Galerkin matrix with integer momenta `-K,...,K` along each axis:

\[
H_{kk'}=2D|k|^2\delta_{kk'}+\widehat V(k-k').
\]

Potential and pair-Jastrow trial Fourier coefficients are evaluated
independently on a uniform 3D grid. A trial projected for imaginary time
`β` on each side yields a pure **diagonal** midpoint reference:

\[
\langle V\rangle_{\beta} =
\frac{\langle\Phi|e^{-\beta H} V e^{-\beta H}|\Phi\rangle}
{\langle\Phi|e^{-2\beta H}|\Phi\rangle}.
\]

The central benchmark compares this finite-`β` reference to the PIGS
midpoint-potential estimator. Increase **both** `K` and the Fourier quadrature
grid before treating the reference as numerically converged; truncation is a
separate systematic error from PIGS primitive-timestep bias, finite `β`, and
MCMC uncertainty. This is a different numerical formulation from the path
sampler, but **not** a closed-form exact continuum answer.

## Run

```bash
python3 -m pytest -q
python3 -m examples.bosons.run_path_staging --quick
python3 -m examples.bosons.run_path_staging
```

The default full example uses two repelling bosons and fixed projection
`β=0.8`, varies `τ` at fixed `n_links * τ = 2β`, and compares the original
Patch 15 move mixture to multi-bead staging/bisection using the same number
of Markov steps and independent seeded replicas. The full run is considerably
more expensive than the quick smoke test.

Output directory: `outputs/bosons/path_staging/`.

- `replicas.csv`: individual chains, means, within-chain autocorrelation
  estimates, acceptance diagnostics, and raw thermodynamic/endpoint means.
- `summary.csv`: between-replica SEM, reference comparison, mean ESS.
- `fourier_reference.csv`: K=1 vs K=2 truncation comparison for projected
  energy and midpoint potential.
- `protocol.json`: all physical and simulation settings, estimator scope.
- `midpoint_convergence.png`: diagonal midpoint potential versus τ, with
  the truncated Fourier reference.
- `effective_samples.png`: estimated independent samples per chain.

This is a **controlled small-system methods benchmark**, not a demonstration
of bulk crystalline order, BEC, superfluidity, or a thermodynamic phase
boundary. The next stages are systematic algorithmic tuning (including CPU
cost), validated larger-N diagonal observables, and only then worldline
permutation/open-sector coherence sampling.
