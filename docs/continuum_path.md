# Patch 14 — Free 3D continuum open-path QMC benchmark

This patch establishes a **stochastically sampled continuum imaginary-time path**
with a **known analytic off-diagonal reference**, without altering the VMC/DMC
engines. It is the continuum analogue of the finite-lattice stochastic benchmark
from Patch 13, not yet a pure OBDM solver for interacting soft-core bosons.

## Scope

One positive trial orbital is projected under the **noninteracting free-particle**
Hamiltonian `H = -D ∇²` in a 3D **orthorhombic** periodic cell. The orbital is

`phi(x) = 1 + a cos(2*pi*mode*x_axis / L_axis)`, with `|a| < 1`.

For **N ideal bosons all occupying the same orbital**, the many-body wavefunction
is a symmetric product. The normalized OBDM and zero-mode occupation per boson
are identical to the one-particle calculations performed here. This factorized
reference does not sample generic N-body exchange or permutation cycles.

The open path has `2m+1` beads and `2m` time steps of length `tau`; `beta=m*tau`
is the imaginary projection time **on each side**. Its normalized weight is

\[
 P(x_0,\ldots,x_{2m}) \propto
 \phi(x_0)\left[\prod_{j=0}^{2m-1}K_\tau(x_j,x_{j+1})\right]\phi(x_{2m}).
\]

### Periodic short-time propagator

The exact heat kernel on an orthorhombic torus is a product of wrapped-normal
kernels:

\[
 K_t(x,y) = \prod_{d=1}^{3}\frac{1}{\sqrt{4\pi D t}}
 \sum_{n\in\mathbb{Z}}
 \exp\left[-\frac{(y_d-x_d+nL_d)^2}{4Dt}\right].
\]

The implementation sums the periodic images numerically, including image tails
beyond eight diffusion standard deviations. The sampler draws increments as
`sqrt(2*D*tau)*normal(0,1)` and wraps each step. **Simply using a single
minimum-image Gaussian would not be a normalized periodic propagator.**

The free kernel is already exact for any positive `tau` (apart from negligible
numerical image-tail truncation); varying the number of links at **fixed beta**
checks the sampler and autocorrelation, **not an interacting Trotter error**.

### Two valid Metropolis moves

1. **Whole-path refresh:** propose a uniformly distributed first bead and
   independent wrapped-normal increments. The proposal density equals
   `q(X)=prod(K_tau)/V`. Therefore the Metropolis acceptance ratio reduces to
   `phi(x0_new)*phi(xend_new)/(phi(x0_old)*phi(xend_old))`.
2. **Bidirectional reptation:** discard one end, grow a new bead from the other
   end via `K_tau`, and randomly choose the growth direction. Detailed balance
   again gives exactly the same endpoint-orbital ratio. The sampler mixes both
   move types (default refresh probability 0.25) to reduce path autocorrelation.

The refresh move has positive proposal density everywhere because `|a|<1`,
so the chain is not reliant solely on slow reptation updates.

## A pure, two-sided projected coherence reference

The exact projected orbital at half-time `beta` is

\[
 \phi_\beta(x)=1+a_\beta\cos(kx),\qquad
 a_\beta=a\exp(-Dk^2\beta),\quad k=2\pi\,\mathrm{mode}/L_x.
\]

Consequently the normalized single-particle coherence of the projected
**pure state** is

\[
 g_1(\mathbf s)=\frac{1+\tfrac12 a_\beta^2\cos(ks_x)}
 {1+\tfrac12 a_\beta^2},\qquad
 \overline{g_1}(r)=\frac{1+\tfrac12 a_\beta^2\,\mathrm{sinc}(kr)}
 {1+\tfrac12 a_\beta^2},
\]

where `sinc(z)=sin(z)/z`. The finite-cell `k=0` occupation is

\[
 \frac{n_0}{N}=\frac{1}{1+\tfrac12a_\beta^2}.
\]

The formulas become exactly 1 for `a=0` and tend to 1 as `beta` increases.
This is a **free reference**, not a prediction for interacting condensate
depletion or a thermodynamic phase transition.

### Stochastic OBDM ratios

For this free, translationally invariant propagator, all internal beads can be
integrated out analytically *without changing the distribution sampled*.
The resulting endpoint estimator is

\[
 \widehat g_1(\mathbf s;X) =
 \frac{K_{2\beta}(x_0,x_{2m}+\mathbf s)}
      {K_{2\beta}(x_0,x_{2m})}.
\]

A **midpoint-block** alternative uses
`K_beta(x0,x_m+s)/K_beta(x0,x_m)`; it has the same expected two-sided
projector OBDM but often much larger variance at large displacement.

The full-volume zero-mode integral can also be performed **per sampled path**
using `integral_cell K_t(x,y+s) ds = 1`:

\[
 \widehat{n_0/N}(X)=\frac{1}{V K_{2\beta}(x_0,x_{2m})}.
\]

This is Rao–Blackwellization of a uniform-displacement estimator, not an
additional approximation. Its finite-run value may be outside `[0,1]` from
sampling noise; reported values are never clipped. Large fluctuations remain
possible for small projection times and spatially separated endpoints.

**Crucial:** the blocked endpoint and midpoint ratios above are valid only
because the *free* one-particle propagator is known and the displacement
operator commutes with the free Hamiltonian. They are **not** a valid general
pure interacting OBDM estimator if the kernel is replaced by an interacting
short-time approximation. That would require an off-diagonal open/worm sector
and a properly normalized estimator.

## How to run

From the project root:

```bash
python3 -m pytest -q
python3 -m examples.bosons.run_continuum_path --quick
python3 -m examples.bosons.run_continuum_path
```

Use `--output DIR` to choose an output location. Output files:

- `replicas.csv`: one measurement per independently seeded Markov chain;
- `summary.csv`: means and replica SEMs for `n0/N` and exact values;
- `g1.csv`: radial coherence and replica SEMs, alongside exact curves;
- `protocol.json`: full numerical settings and scope declaration;
- `occupation_convergence.png`: stochastic and exact occupation vs `beta`;
- `g1_comparison.png`: stochastic and exact radial curves.

The per-replica SEM does **not** prove independence of successive path samples.
Run longer chains, vary thinning and refresh/reptation proportions, compare
independently seeded results, and estimate integrated autocorrelation times
before interpreting uncertainties as converged. Also check projection time
separately from autocorrelation.

## Roadmap to a real interacting continuum pure estimator

An interacting 3D periodic Bose system needs, at minimum:

- a positive, validated **many-particle** short-time propagator with a
  potential factor and controlled `tau -> 0` discretization bias;
- worldline and exchange/permutation updates for indistinguishable bosons;
- an off-diagonal endpoint/worm sector or equivalent estimator for one-body
  coherence, with correct relative-sector normalization;
- projection-length, sampling-time, time-step, population/finite-size and
  trial-state convergence studies;
- references against known free results and, where possible, exactly soluble
  or high-quality interacting limits.

Patch 14 covers the **free periodic kernel and a correct open-path sampler**,
not these remaining interacting extensions.
