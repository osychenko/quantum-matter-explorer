# Patch 15 — Interacting continuum open paths (diagonal benchmarks)

This patch is the first **interacting 3D periodic** imaginary-time path sampler.
It is not yet a bosonic open-worldline/worm implementation and does **not**
produce a pure off-diagonal one-body density matrix or condensate fraction.

## Target and approximation

For a positive, permutation-symmetric boundary trial `Phi`, the target is

\[
\pi(X) \propto \Phi(R_0)\Phi(R_M)
  \prod_{s=0}^{M-1}K_0(R_s,R_{s+1};\tau)
  \exp\!\left[-\tau\left(\tfrac12 V(R_0)
    + \sum_{s=1}^{M-1}V(R_s) + \tfrac12 V(R_M)\right)\right].
\]

`K0` is the **exact normalized periodic free heat kernel** from Patch 14,
including images. This is the symmetric primitive (Strang) approximation
`exp(-τV/2) exp(-τT) exp(-τV/2)` per link. It has time-step bias of
order `O(τ²)` for sufficiently smooth potentials at fixed projection time,
but the coefficient need not be small. Keep `β=Mτ/2` fixed when comparing τ.
The present default is a finite-range soft-core repulsion from Patch 4.

The path uses **labelled particles** and no explicit permutation cycles.
For the fixed-N zero-temperature ground state, a strictly positive symmetric
boundary condition has overlap with the positive bosonic ground state, but
finite β, large systems and metastability can prevent convergence. **Do not
interpret this as a finite-temperature bosonic partition function:** exchange
cycles and permutation sampling are indispensable there. Off-diagonal
observables additionally require an open-worldline sector and normalization.

## Moves and detailed balance

Three Metropolis–Hastings updates preserve the primitive-action target:

1. A Gaussian *single-particle, single-bead* move; its proposal is symmetric
   on the periodic torus. Only one potential slice, adjacent free links,
   and a boundary amplitude (for an endpoint) enter the ratio.
2. A collective translation of all beads of one labelled particle. Its free
   link factors cancel; only boundary and potential terms change.
3. An independent full-path refresh from the normalized **free** path measure.
   All kinetic factors cancel against the proposal density; the acceptance
   depends on the potential action and two boundary trial amplitudes.

The whole-path refresh ensures irreducibility at finite projection length,
but its acceptance can become exponentially poor with large N or β.
These elementary updates are a foundation, not a high-efficiency production
PIGS/reptation method (which needs staging/bisection and improved mixing).

## Three distinct diagonal estimators

- `midpoint_potential`: `V(R_M/2)` samples the two-sided projected diagonal
  observable of the **primitive-action** path measure.
- `endpoint_local_energy`: the mean endpoint
  `V(R) - D*(∇² log Phi + |∇ log Phi|²)`. For the exact propagator, its
  path average equals the projected energy; with primitive propagation it is
  a finite-τ diagnostic, **not** an unbiased ground-state energy.
- `thermodynamic_energy`: `-∂ ln Z_M / ∂(Mτ)` at **fixed M** for the discrete
  action, using the *periodic-image weighted* free-kernel derivative.
  This estimator has large variance. Comparing its result with endpoint energy
  and scanning τ is a useful consistency check, not proof of convergence.

We report independent-replica standard errors; steps within a Markov chain
are autocorrelated. Two replicas are sufficient to test the workflow but **not
for robust uncertainty quantification**. Check longer runs, blocking,
initialization sensitivity, autocorrelation, larger populations/β, and a
fixed-β τ→0 scan before asserting a ground-state energy.

## Independent analytical references

- **Uniform free case**: zero energy and zero potential; free full-path
  refresh and translation moves accept with probability one.
- **Constant potential**: midpoint and endpoint-local estimates equal the
  constant exactly; the thermodynamic estimator fluctuates around it.
- **Periodic cosine potential (N=1)**: a truncated Fourier Hamiltonian
  independently computes the exact finite-β projected energy and midpoint V.
  This is a nontrivial reference for diagonal imaginary-time projection.
- **Soft-core interacting bosons**: exercises true pair interactions and
  the existing Jastrow trial. Results are exploratory until a dedicated
  high-precision reference and convergence study have been performed.

## Commands

From the repository root:

```bash
python3 -m pytest -q tests/test_interacting_path.py
python3 -m examples.bosons.run_interacting_path --quick
python3 -m examples.bosons.run_interacting_path
```

Outputs under `outputs/bosons/interacting_path/`:
`replicas.csv`, `summary.csv`, `protocol.json`,
`timestep_convergence.png`, `thermodynamic_energy.png`.

## Next numerical milestone

Patch 16 can introduce multi-bead staging/bisection moves, autocorrelation
measurements and an independent interacting-reference calculation (e.g., a
finite-basis two-particle periodic benchmark). After diagonal projected
observables are well controlled, a properly normalized off-diagonal worm/open-
worldline sector can be designed and validated. **Do not reuse the free
particle's blocked OBDM estimator unchanged for interacting paths.**
