# Patch 20 — Spacetime-separated canonical worm: exact lattice benchmark

**Scope.** This patch introduces an actual creation/annihilation *pair of worm
endpoints at independently movable imaginary times and lattice sites*. The
interval between them carries **N+1 bosons**; the remainder of the periodic
imaginary-time circle carries **N bosons**. It measures a **finite-temperature
particle-addition Green function** for a **finite one-dimensional Bose–Hubbard
ring**. It is *not* a continuum 3D soft-core-boson worm or a pure ground-state
projector. The finite Hilbert-space transfer matrices are computed by dense
exact diagonalization (Patch 12), so this is a rigorously normalized small-system
benchmark, not an efficient method at large particle numbers.

## Ensemble and normalization

Use the Patch 12 Bose–Hubbard Hamiltonian, with a fixed particle count `N` in
its canonical partition function. Let `L` be the number of sites, `M` the number
of imaginary-time slices, `dt=beta/M`, and `T_n=exp[-dt (H_n-E_shift)]` with the
**same** constant energy shift in the `N` and `N+1` sectors. This shared shift
cancels between the Z and G ratios. Define

\[
Z_N=\mathrm{Tr}(T_N^M), \quad
F_{ij}(s)=\mathrm{Tr}\left(T_N^{M-s}\,a_i\,T_{N+1}^{s}\,a_j^\dagger\right),
\qquad 1\leq s\leq M-1.
\]

The worm's **tail** creates a particle at `(j,t)`, and its **head** annihilates
that particle at `(i,t+s mod M)`. The open interval contains `N+1` bosons and
the complementary interval contains `N`. The configuration-space weights,
after integrating all internal occupation-number paths, are

\[
\pi_Z=Z_N,\qquad
\pi_G(t,s,i,j)=\frac{\eta}{M(M-1)L^2}\,F_{ij}(s),
\]

where `eta` is the **known** dimensionless sector fugacity, not a fitted
normalizer. Each of `t`, `i`, and `j` is an independently represented label.
The finite-temperature time-ordered **particle-addition** Green function is
`G_ij(s dt)=F_ij(s)/Z_N`. The normalization can be estimated using residence
counts. With translation symmetry, set `d=(i-j) mod L` and pool over all tail
sites and all M tail-time positions:

\[
G(d,s\,dt)=\frac{L(M-1)}{\eta}\;
\frac{N_G(d,s)}{N_Z}.
\]

The example Fourier-transforms `G(d,tau)` to `G(k,tau)`. The finite-sampling
ratio may be noisy or outside a naive expected interval and is never clipped.
`Z_G/(Z_Z+Z_G)` is calculated independently by exact contraction and reported
alongside the observed worm residence fraction.

At `tau -> 0+`, operator commutation gives `G(k,0+) = 1 + n_k`, where `n_k`
is the **thermal** occupation *number*, not its fraction. Finite-tau values
are **not** silently interpreted as `n_k`. The run reports an **independent
exact thermal OBDM** and its k=0 occupation fraction for scientific context.

## Markov updates and exact conditional paths

The collapsed-endpoint Metropolis sampler keeps `(t,s,i,j)` as explicit
variables in G and an aggregate state in Z:

- **Open**: propose `(t,s,i,j)` uniformly. Accept with
  `min(1, eta*F_ij(s)/Z_N)`; the reverse **close** move uses its reciprocal.
  The same `toggle_probability` is used in *both* sectors. This is essential:
  unequal attempt rates without a proposal correction bias the occupation.
- **Advance/recede head**: move head time `+1` or `-1` while holding tail time
  fixed; `s` changes. **Advance/recede tail** moves tail time independently
  while preserving the head time. Moves that would produce `s=0` or `s=M`
  reject, leaving `1 <= s < M`. Reversible acceptance uses `F_new/F_old`.
- **Head/tail spatial relocation**: propose another lattice site for either
  endpoint, with symmetric selection and the same `F_new/F_old` acceptance.

Unlike a conventional bead-by-bead worm implementation, this small-system
method *integrates out* the internal worldline occupations when deciding
endpoint moves. Calling `sample_path()` then draws genuine many-body
occupation-number paths conditional on the current state. In the G sector,
explicit creation/annihilation matrix elements proportional to square roots
of occupation numbers connect an `(N+1)`-particle bridge to an `N`-particle
bridge. Conditional bridge draws use the exact Markov identity

\[
P(x_{l+1}=v\mid x_l=u,x_m=w)
=\frac{T_{uv}(T^{m-l-1})_{vw}}{(T^{m-l})_{uw}}.
\]

Both path segments include their endpoints and their link counts sum to M.
The internal path sampler checks its pair-normalizer against the independently
contracted `F_ij(s)` value. The closed sector returns an `N`-particle ring of
M links with equal initial and final states.

This **is stochastic sampling** of an explicitly normalized spacetime-worm
ensemble, but it relies on exact dense matrices to perform collapsed
transitions, which a production continuum QMC calculation cannot afford.

## Validation

Run:

```bash
python3 -m pytest -q
python3 -m examples.bosons.run_spacetime_worm --quick
python3 -m examples.bosons.run_spacetime_worm
```

The tests include:

- comparison of `F/Z` with a separately constructed spectral expression;
- comparison of the exact free-lattice k=0 occupation with a canonical
  ideal-Bose partition-function recursion using single-particle energies;
- `G(k,0+)=1+n_k` via an independent thermal OBDM;
- reversible time, space, and open/close proposal-ratio checks;
- a long-chain sector-residence check, specifically guarding against
  asymmetric Z/G move-attempt probabilities;
- exact `N` to `N+1` occupation jumps at the worm endpoints;
- reproducibility, bad-configuration rejection, and output contract.

Quick mode is a software and normalization smoke test, not a precision study.
The larger example includes both free and repulsive Bose–Hubbard models at
fixed inverse temperature and two imaginary-time grids. Because the matrix
transfer kernel is exact, changing `M` changes the set of sampled time
separations but introduces **no primitive-action timestep bias**. It does
change worm autocorrelation and the variance of estimates.

Output directory `outputs/bosons/spacetime_worm/`:

- `replicas.csv`: raw independent seeded chains and sector diagnostics;
- `summary.csv`: between-replica means/SEMs and exact sector references;
- `green.csv`: stochastic `G(k=0,tau)`, `G(k=1,tau)` and exact spectral values;
- `protocol.json`: full configuration and distinction between finite-tau
  Green and equal-time occupation;
- `green_vs_time.png`, `sector_normalization.png`: comparison plots.

## What is still needed for interacting 3D continuum bosons

We have **not** extended Patch 19's 3D continuous-space primitive-action
sampler to arbitrary worm endpoint times. That would require genuinely local
creation/annihilation of an open worldline segment, the correct changes to
interaction action when the particle is present for only part of imaginary
time, endpoint and insertion proposal Jacobians, canonical/grand-canonical
normalization choices, and rigorous free/interacting reference tests.
Additional continuum discretization and long-chain convergence studies are
necessary before interpreting BEC or superfluid response. This patch provides
an exact, explicit normalization and detailed-balance target for that work.
