# Patch 12 — exact pure OBDM reference by symmetric projection

This patch adds a **controlled pure-estimator benchmark**, not a production
continuum projector Monte Carlo implementation. It uses dense exact
Hamiltonian diagonalization on a *one-dimensional finite Bose-Hubbard ring*,
separate from the existing 3D continuum soft-core Hamiltonian. No physical
result can be transferred between the two models without separate analysis.

## Why a separate reference model?

The Patch 9 and 10 off-diagonal estimator evaluates trial-wavefunction ratios
on VMC configurations. Patch 11 evaluates the same ratios on mixed-distribution
DMC walkers and constructs `2*mixed - VMC`. For an off-diagonal observable,
this is not a pure estimator; ordinary descendant weighting of diagonal walker
observables does **not** fix off-diagonal insertion.

A genuine open-path/reptation QMC method introduces additional path degrees of
freedom, endpoint weights, off-diagonal sectors, moves, and normalization. Those
steps demand exact small-system validation. Patch 12 supplies such a reference
for a finite Hilbert space, with *no Monte Carlo sampling or statistical bias*.

## Model

The one-dimensional periodic Bose-Hubbard Hamiltonian is

\[
H=-t\sum_{i=0}^{L-1}(a^\dagger_{i+1}a_i+a^\dagger_i a_{i+1})
+\frac{U}{2}\sum_i n_i(n_i-1)+V\sum_i n_i n_{i+1},
\]

with periodic site indices, `L >= 3`, fixed particle number `N >= 1`,
nonnegative `t`, `U`, `V`, and a fixed-total-N occupation basis.

The Hilbert-space dimension is `comb(N+L-1, L-1)`. Dense matrices are limited
to 600 basis vectors by default. This cannot be scaled like the VMC/DMC engine:
the goal is exact verification rather than many-body throughput.

## Symmetric imaginary-time projection

For a seed `|Phi>` with nonzero overlap with a unique ground state, define

\[
|\Psi_\beta\rangle=\frac{e^{-\beta(H-E_0)}|\Phi\rangle}
{\|e^{-\beta(H-E_0)}|\Phi\rangle\|}.
\]

The **pure**, equal-imaginary-time one-body matrix at finite projection time is

\[
\rho_{ij}(\beta)=
\frac{\langle\Phi|e^{-\beta H}a_i^\dagger a_j e^{-\beta H}|\Phi\rangle}
{\langle\Phi|e^{-2\beta H}|\Phi\rangle}.
\]

At `beta=np.inf` and nonzero overlap, this converges to the exact ground-state
expectation `⟨Psi_0|a_i† a_j|Psi_0⟩`. **`beta` is the projection time on
_each_ side** of the OBDM insertion (total two-sided length `2*beta`).

The program computes this using exact eigenpairs of the full small Hamiltonian,
not a primitive propagator, diffusion walkers, or Monte Carlo importance
sampling. It reports the spectral gap; unique-ground-state assumptions must be
revisited when the gap is zero or the initial seed is orthogonal.

## Physical observables and normalization

`obdm[i,j]` is `⟨a_i† a_j⟩`, Hermitian and positive semidefinite, with
`trace(obdm) = N`. The distance-resolved `g1(d)` is the site-averaged

\[
g_1(d)=\frac{1}{N}\sum_{i=0}^{L-1}\rho_{i,i+d\,\mathrm{mod}\,L},
\quad g_1(0)=1.
\]

The zero-momentum orbital has normalized uniform amplitudes `1/sqrt(L)`:

\[
\frac{n_{k=0}}{N}=\frac{1}{LN}\sum_{i,j}\rho_{ij}.
\]

Also report the largest eigenvalue of `obdm` divided by `N` (the largest
natural-orbital occupation). For an inhomogeneous finite-beta seed it need not
equal the uniform-mode occupation. Values are bounded by [0,1] aside from
floating-point roundoff. These definitions are discrete-lattice quantities,
not the continuum `g1(r)` and volume integral in Patches 9–11.

### Exact regression limits

- `U=V=0, t>0`: the coherent orbital product is exact, `g1(d)=1`,
  `n_k0/N=1`, and `E0=-2tN` on a ring.
- `t=0, U>0, N=L, V=0`: the unique unit-filled state has
  `obdm = identity`, `g1(d>0)=0`, and `n_k0/N=1/L`.
- At nonzero hopping and positive ground overlap, very different seeds
  approach the same pure matrix as beta increases.

## Running

```bash
python3 -m pytest -q
python3 -m examples.bosons.run_pure_projector --quick
python3 -m examples.bosons.run_pure_projector
```

Optional switches: `--onsite-u 4`, `--nearest-v 0`, `--output PATH`.
The normal example uses six sites and five bosons (252 Hilbert states). The
quick example uses four sites and three bosons (20 states).

Outputs: `projection.csv` (energy and occupation vs beta), `obdm.csv`
(`g1(d)`), `natural_occupations.csv` (OBDM eigenvalues), `protocol.json`,
`projection_convergence.png`, and `obdm_comparison.png`.

## Next necessary implementation (not included)

A true **3D continuum** pure off-diagonal QMC estimator needs a validated
open-path/reptation or worm-sector formalism, path-action discretization,
transition moves with detailed balance, endpoint normalization, timestep and
projection-length extrapolation, and proper population/time correlation
analysis. Cross-check it first against this finite-basis reference (or
small continuum cases with independent exact solutions) before publishing any
pure condensate fraction. Neither this lattice benchmark nor the previous
mixed/extrapolated DMC estimates establishes thermodynamic BEC or superfluidity.
