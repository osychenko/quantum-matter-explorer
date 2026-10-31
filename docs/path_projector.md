# Patch 13 — Stochastic open-path midpoint projector (lattice benchmark)

**Scope.** This method samples open imaginary-time paths for a *small 1D
Bose–Hubbard ring* at fixed particle number. It is an independent stochastic
check of the exact two-sided projector in `qmc/pure_projector.py` (Patch 12).
It is **not** a pure OBDM estimator for the existing 3D continuum soft-core
bosons. A continuum short-time propagator, open ends/worm moves, importance
sampling, timestep and population controls, and extensive benchmarking are
separate future milestones.

## Model and positive path measure

The Hamiltonian and finite occupation basis are provided by `BoseHubbardRing`.
For positive hopping `t`, the Hamiltonian is stoquastic in the occupation
basis. The exact finite-dimensional imaginary-time kernel is

\[
T=e^{-\delta\tau (H-E_0)}.
\]

The `E0` shift is a harmless scalar normalization. Unlike a continuum
Trotter–Suzuki approximation, the **matrix exponential is constructed exactly
using dense eigendecomposition**. For the small basis tested here `T` is
symmetric and entrywise nonnegative. The method needs strictly positive
trial amplitudes \(\phi(x)>0\) in every occupation state. We cap the model
at dimension 120, because storing the dense operator bridges scales as
\(L^2 d^2\) (with `d` the Fock-space dimension) and evaluating them is not
scalable.

For a path of `2m` links and `2m+1` configurations, the probability is

\[
P(x_0,\ldots,x_{2m})\propto
\phi(x_0)\left[\prod_{s=0}^{2m-1}T(x_s,x_{s+1})\right]\phi(x_{2m}).
\]

Both endpoints carry the *trial* factor, not a squared trial factor. The
midpoint observable therefore has the **two-sided pure projected** expectation

\[
\langle O\rangle_\beta=
\frac{\langle\phi|e^{-\beta H} O e^{-\beta H}|\phi\rangle}
{\langle\phi|e^{-2\beta H}|\phi\rangle},
\qquad \beta=m\delta\tau.
\]

It equals the exact finite-time projector result at equilibrium. The residual
projection bias disappears as \(\beta\) grows, assuming nonzero overlap with
an isolated ground state. There is no mixed-estimator or linear-extrapolation
assumption in this measurement.

## Path updates and measurement

The chain alternates two **stationarity-preserving** updates:

- Reptation Metropolis–Hastings: randomly drop either endpoint, extend the
  other by drawing from `T[x,y]/sum_z T[x,z]`, and accept with the exact
  reverse-proposal correction. The code is tested against a direct path-weight
  and proposal-ratio calculation.
- Single-bead Gibbs: sample an interior occupation from
  `T[left,x] T[x,right]` or an endpoint from `phi[x] T[x,neighbor]`.
  This improves mixing without changing the target measure.

For the one-body matrix \(O_{ij}=a_i^\dagger a_j\), we integrate the midpoint
occupation state out analytically, conditional on the two neighboring beads
\(l=x_{m-1},\ r=x_{m+1}\):

\[
\widehat{\rho}_{ij}(l,r)=
\frac{(T\,a_i^\dagger a_j\,T)_{lr}}{(T^2)_{lr}}.
\]

This **conditional bridge estimator** is a variance-reduced off-diagonal
observable: it does not need a worm-sector normalization. It does require the
dense, small-basis matrix kernel. The sampled estimates are symmetrized when
forming the final OBDM to remove finite-Monte-Carlo antisymmetric noise.
The single-particle zero-momentum occupation is

\[
\frac{n_0}{N}=\frac{1}{LN}\sum_{ij}\rho_{ij},
\qquad
g_1(d)=\frac1N\sum_i\rho_{i,i+d\pmod L}.
\]

For any sampled path, the conditional OBDM has trace \(N\) exactly. Off-site
matrix elements and energy fluctuate. Monte Carlo values can lie slightly
outside physical bounds at finite sampling; the program does not clip them.

## Exact validation and convergence caveats

`tests/test_path_projector.py` checks conditional operator insertion against
independently constructed occupation-space operators, exact bridge marginal
summation against `ExactBosonProjector.measure(beta)`, detailed-balance
acceptance ratios, the free-boson limit, and stochastic agreement with the
finite-time exact projector. A short numerical run **cannot** establish that
the chain has mixed or that the ground-state limit has been reached.

The example saves independent-chain **replica** means and standard errors.
Samples within a chain are autocorrelated even after thinning; estimates of
uncertainty with 2–3 chains are illustrative. Longer chains, integrated
autocorrelation-time diagnostics, multiple initializations, changing `dt`
and `half_steps` at fixed `beta`, and more independent replicas are needed for
reliable scientific uncertainty. The kernel has **no Trotter timestep error**
for this lattice benchmark; varying `dt` at fixed `beta` tests the *sampler*
and numerical conditioning, not continuum propagator convergence.

## Run

From the repository root:

```bash
python3 -m pytest -q
python3 -m examples.bosons.run_path_projector --quick
python3 -m examples.bosons.run_path_projector
```

The full example is more expensive. The output directory can be changed with
`--output PATH`. Files under `outputs/bosons/path_projector/`:

- `replicas.csv`: raw seed, mean energy, occupancy, and reptation acceptance;
- `summary.csv`: between-chain SEM and exact finite-beta reference;
- `g1.csv`: one-body coherence by site separation, with SEM and exact target;
- `projection_convergence.png`: exact and sampled energy and occupation;
- `g1_comparison.png`: exact and sampled one-body coherence;
- `protocol.json`: configuration and methodology disclosure.

The earlier 3D soft-core VMC/DMC solvers are unchanged. Extending this method
to continuum bosons would require a separate, physically validated
short-time action and appropriate off-diagonal path-sector sampling.
