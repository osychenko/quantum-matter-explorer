# Patch 18: normalized finite-temperature boundary-link worm benchmark

This patch introduces a **canonical, finite-temperature, off-diagonal**
worldline estimator built on Patch 17's `BosonPermutationPaths`. It is an
explicit, reversible two-sector (`Z`/`G`) Markov chain. **It is not a pure
zero-temperature projector, nor a full mobile head/tail worm algorithm:** the
open displacement occurs only on one imaginary-time closure link. The system
has a fixed number of bosons, an orthorhombic periodic cell, and the primitive
Trotter action. `n_particles` is restricted to 2–5 by the existing exact
free-cycle proposal enumerator.

## The normalized sector weight

Write a closed configuration as `X=(R_0,...,R_{M-1},P)`, where the permutation
`P` joins final bead `R[M-1,i]` to first bead `R[0,P(i)]`. Its positive
primitive canonical weight is

\[
W_Z(X)=\prod_{j=0}^{M-2}\prod_i K_\tau(R_{j+1,i}-R_{j,i})
  \prod_i K_\tau(R_{0,P(i)}-R_{M-1,i})
  \exp\!\left[-\tau\sum_j V(R_j)\right].
\]

`K_tau` is the **periodic** free-particle heat kernel, not a minimum-image
Gaussian. The `G` sector keeps the same underlying `X` but selects a closure
link `i` and translates its closing endpoint by `s` in the primitive cell:

\[
W_G(X,i,s)\,d^3s=\frac{\eta}{NV}W_Z(X)
\frac{K_\tau(\Delta_i+s)}{K_\tau(\Delta_i)}\,d^3s,
\qquad \Delta_i=R_{0,P(i)}-R_{M-1,i}.
\]

`eta>0` is a **known dimensionless** sector fugacity; the discrete link
selection has the explicit `1/N`, and `d^3s/V` is the uniform displacement
measure. No fitted sector normalization is used.

For the primitive canonical bosonic trace, the shifted-link ratio is the
one-body translation insertion. At finite `M`, it measures the 1BDM of the
specified discretized density operator. The exact continuum-quantum result
requires extrapolating `tau -> 0` at fixed `beta = M*tau`. The **unshifted
potential at the cut** is an explicit unsymmetrized off-diagonal insertion
convention, even though the closed trace uses the standard primitive action.
Consequently the interacting off-diagonal observable may have an **O(tau)
endpoint insertion error** in addition to primitive-action errors; a
second-order convergence claim for this estimator is **not** justified.
The free-particle reference is unaffected by this choice.

Define `g1(s)=Tr[exp(-beta H) T_s]/Tr[exp(-beta H)]` with the averaged
single-particle translation operator `T_s=(1/N)sum_i T_s^(i)`; hence
`g1(0)=1`. The relative sector partition functions satisfy

\[
\frac{Z_G}{Z_Z}=\eta\frac{1}{V}\int_V g_1(s)\,d^3s
=\eta\frac{n_{k=0}}{N}.
\]

The **absolute normalization** is therefore given by sector visits:

\[
\widehat{n_k/N}=
\frac{\sum_{t\in G}\cos(k\cdot s_t)}{\eta N_Z},
\qquad k=2\pi(n_x/L_x,n_y/L_y,n_z/L_z).
\]

This estimator is not constrained to `[0,1]` at finite sample count, and
noisy nonzero Fourier occupations may even become negative. Neither case is
clipped. The `k=0` estimate is precisely `N_G/(eta*N_Z)`.

For radii `0 <= r <= min(L)/2`, shell-volume-normalized G visits estimate
an angular-averaged radial `g1(r)`:

\[
\hat g_1(r_a)=\frac{N_{G,a}}{\eta N_Z}
\frac{V}{(4\pi/3)(r_{a+1}^3-r_a^3)}.
\]

Shell occupancy is small near the origin, so this radial histogram may be
high-variance; `g1(0)=1` is analytically exact but not imposed on histogram
bin averages.

## Reversible moves and their normalization

Within `Z`, the sampler delegates to Patch 17's diagonal bead, bridge,
permutation swap, translation, and exact-free-cycle refresh moves. In `G`,
the underlying path and permutation are held fixed, while the translated
closure displacement can change. When the sector is toggled, these paths
remain unchanged, so detailed balance is straightforward.

**Metropolis open:** choose `i` uniformly, draw `y` from the normalized
periodic kernel `K_tau(y)`, and set `s = y - Delta_i` modulo the cell.
The proposal for `s` is exactly `K_tau(Delta_i+s)`; therefore the *proposal*
kinetic factor cancels, leaving the opening acceptance ratio

\[
A_{Z\to G}=\min\left(1,\frac{\eta}{V K_\tau(\Delta_i)}\right).
\]

**Metropolis close:** the exact reverse factor is `V*K_tau(Delta_i)/eta`.

**Conditional sector Gibbs (default 80% of sector proposals):** integrate
over `s` analytically. At fixed closed path and permutation, the relative
weights for `Z,G_0,...,G_(N-1)` are

\[
(1,\;\eta/[N V K_\tau(\Delta_0)],\ldots,
       \eta/[N V K_\tau(\Delta_{N-1})]).
\]

Choose one of these `N+1` states from the normalized weights. In a `G_i`
state draw a fresh `s` from its conditional periodic kernel. This Gibbs
update has **acceptance one**, including for the interacting action because
only the conditional `Z/G` link factor differs. Heatbath-only displacement
updates also accept with probability one; symmetric local displacement
moves use the exact kernel ratio. These identities are covered by tests.

**Sector-collapsed refresh:** one additional independent free-boson worldline
proposal updates the path/permutation even while the worm sector is open.
Integrating both G displacements and the discrete sector gives the closed-path
marginal weight `W_Z(X) * F(X)` with

\[
F(X)=1+\sum_irac{\eta}{N V K_	au(\Delta_i)}.
\]

An independent free-boson proposal `q_free(X')` cancels all free kinetic
factors, leaving an MH ratio

\[
\exp[-	au(\sum_j V(R'_j)-\sum_jV(R_j))]\;rac{F(X')}{F(X)}.
\]

Immediately drawing a new conditional sector by Gibbs restores the full joint
law. Without these path updates, at small `tau` a rare tiny closure kernel may
trap the worm sector and make the naive visit-ratio estimator unusable. The
collapsed refresh improves connectivity but cannot guarantee efficient mixing.

Because `Z` moves can change paths and permutations and all `G` states can
close, the two-sector chain can explore the full extended target provided
the underlying `Z` sampler is ergodic. This does **not** guarantee that
finite chains equilibrate or are independent.

## Two independent estimator routes

The patch additionally supplies `closed_link_occupations`. It samples
**only Z** paths, analytically integrating over the translated link:

\[
\widehat{n_k/N}_{Z\text{-only}}=
\left\langle\frac{1}{N}\sum_i
\frac{e^{-D\tau|k|^2}\cos(k\cdot\Delta_i)}
{V K_\tau(\Delta_i)}\right\rangle_Z.
\]

This is a Rao–Blackwellized fixed-path estimator for the *same primitive
ensemble*, not a separate analytic solution of the interacting system. Its
variance may be large when closure kernel values are tiny. Agreement between
this route and the worm sector-ratio estimate is a useful independent
**estimator implementation** check, not a proof of vanishing timestep or
mixing bias.

For **free bosons**, the model *does* have an exact, independent, finite-N
canonical momentum occupation formula:

\[
Z_0=1,\quad
Z_N=\frac1N\sum_{l=1}^N Z_1(l\beta)Z_{N-l},\quad
\frac{n_k}{N}=\frac{1}{NZ_N}\sum_{l=1}^N
Z_{N-l}e^{-l\beta D|k|^2}.
\]

`canonical_free_occupations` evaluates this directly using the periodic
single-particle trace from Patch 17. Tests cover exact `N=1,2`, occupation
sum rules and finite-temperature convergence. The full example overlays
the exact free reference and both MC estimators. **The interacting case has
no matching analytic off-diagonal reference.**

## Running the benchmark

```bash
python3 -m pytest -q
python3 -m examples.bosons.run_worm --quick
python3 -m examples.bosons.run_worm
```

Results under `outputs/bosons/worm/`:

- `replicas.csv`: every independent seeded run; open/closed visits, estimates,
  measured sector effective sample sizes, move acceptances.
- `summary.csv`: mean and between-replica SEM for both `n0` estimators, free
  reference where applicable, and basic exchange diagnostics.
- `momenta.csv`: first reciprocal modes from both estimators, with replica
  uncertainty and ideal-gas exact values.
- `g1_radial.csv`: shell-normalized radial histogram and replica errors.
- `occupation.png`, `momentum.png`, `g1_radial.png`: benchmark figures.
- `protocol.json`: physical parameters, seed scheme, and limitations.

The `--quick` configuration is an integration smoke test, not a converged
physics result. Increase independent replicas, lengthen production chains,
check state autocorrelation, examine timestep convergence at fixed `beta`,
and compare different initializations and move mixtures before interpreting
interactions. All uncertainties reported here are across independent replicas
rather than treating correlated path measurements as independent.

## What this does NOT establish

- This is a **finite-temperature** canonical 1BDM and momentum distribution,
  not the pure zero-temperature 1BDM studied in Patches 12–16.
- The boundary-link opening is deliberately limited to the imaginary-time
  closure. A general mobile worm with arbitrary imaginary-time separation,
  insertion/removal of worldline segments, and grand-canonical particle-number
  fluctuations is **not** implemented.
- There is no superfluid-fraction or winding-number estimator here.
- Even when a finite system has significant `n0/N`, demonstrating BEC requires
  careful particle-number/temperature scaling. It does not alone imply
  superfluidity or supersolidity.
- Primitive-action discretization, the asymmetric off-diagonal insertion at
  finite tau, permutation mixing, sector residence-time correlation, and
  possible metastability remain substantial concerns.

The next milestone should test a more mobile worm sector and windings, then
perform controlled finite-size/temperature scans of interacting systems.
