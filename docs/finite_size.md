# Patch 7 — finite-size and Monte Carlo sensitivity study

This module is a reproducible **VMC diagnostic** of finite-size, initialization,
and sampling-length effects in the Patch 6 soft-core boson model. It does **not**
find a thermodynamic phase boundary, quantify BEC, or establish a crystal.

## Held-fixed physics and system size

For each choice of particle number `N`, reduced density `rho_rc3`, and
interaction coupling `gamma`, the box length and repulsive potential height are

\[
L=r_c (N/(\rho r_c^3))^{1/3},\qquad
V_0=\gamma D/r_c^2.
\]

Thus `N` varies at **fixed dimensionless density and coupling**. The interaction
cutoff must satisfy `r_c <= L/2`; invalid cells are rejected. All boxes are cubic.

The density-wave trial uses three positive axial reciprocal vectors, selected
near the target *physical* reduced magnitude `target_kr = |G| r_c`:

\[
m=\max(1,\operatorname{round}[(k_{\rm target}r_c)L/(2\pi r_c)]),
\quad G=(2\pi m/L)\hat{e}_{x,y,z}.
\]

Each output row records the selected integer `mode_index` and **actual**
`k_cutoff = |G|r_c`. Finite-box reciprocal grids are discrete; their selected
modes are *not identical* in physical wave number. Results may contain
commensurability artifacts, and comparisons at constant target `k*r_c` are
not the same as comparing exactly identical modes. The density-wave trial
selects particular orientations, which further constrains interpretation.

## Protocol

For every `(N, density, gamma)` point:

1. **Training:** sample each trial-grid candidate from fresh independent
   uniformly initialized VMC chains. Select the lowest *training* energy in
   the `lambda=0` and `lambda>0` families separately. Reoptimize at every N.
2. **Held-out validation:** run each selected trial with newly seeded chains.
   For each chain length, run independent replicas from two initialization
   families: **uniform positions** and **jittered lattice-like positions**.
3. Record each run's energy/particle, accepted-move fraction, preselected-mode
   `S(G)`, and `S(G)/N`. Report means and standard errors across independent
   replica means **within each start family**.
4. Compare starts and independent short/long chains. No automatic significance
   test or phase assignment is made.

The initial lattice uses randomly chosen sites from a cubic grid with a random
translation, a random particle ordering, and Gaussian jitter. It is an
**initialization diagnostic**, not a crystal trial wavefunction. Differences
between starts after equilibration can indicate insufficient mixing or
metastability, and should prompt longer and stronger sampling checks.

The two trial families are **nested**: `lambda=0` belongs to the larger
`lambda>=0` family. A lower estimated energy for one family over the other is
not itself a phase-transition signal; finite-grid optimization and statistical
noise can reverse apparent energy order. The density-wave trial may favor
clusters rather than a conventional single-occupancy crystal.

## Uncertainty and convergence limits

The reported standard errors are `std(independent_replica_means, ddof=1) /
sqrt(n_replicas)`. The correlated walkers within a run and successive VMC
sweeps are **never** treated as independent replicas. Training block standard
errors are used only during parameter selection. With just two validation
replicas (quick mode) the SEM is highly uncertain; run many longer chains for
scientific claims. The short and long protocols use **independent seeds**, not
nested prefixes of a single chain.

A true finite-size extrapolation needs more and substantially larger particle
numbers, careful mode matching and optimized trials, correlation-time checks,
possible alternate crystalline morphologies, and sampling/initialization
convergence. `S(G)/N` helps distinguish O(N) Bragg-like growth from O(1)
liquid-like S but is only a diagnostic at the few sizes sampled. The current
bounded repulsion can support clustering, including cluster crystals.

This is **variational** VMC. DMC mixed-estimator biases, time-step error,
finite-population bias, and pure-structure estimators are outside this patch.
There is no defensible nonzero-order-parameter extrapolation or transition
location from the included quick/default grids.

## Run

From the repository root:

```bash
cd ~/quantum-matter-explorer
python3 -m pytest -q
python3 -m examples.bosons.run_finite_size --quick
python3 -m examples.bosons.run_finite_size --output outputs/bosons/finite_size_full
```

The **default/full run is substantially more expensive** than quick mode.
The file `examples/bosons/run_finite_size.py` exposes the `--quick` flag and
output directory; more extensive protocols can be configured by importing
`FiniteSizeConfig` and using `dataclasses.replace`.

Output files:

- `protocol.json` — exact configuration, seeds, estimator and caveat;
- `training.csv` — all candidate grid evaluations and training energies;
- `replicas.csv` — every held-out run, full physical settings, start and seed;
- `summary.csv` — per-start, per-length means with between-replica SEM;
- `start_contrasts.csv` — *lattice minus uniform* mean differences and
  quadrature SEM, not a hypothesis test;
- `length_contrasts.csv` — *longest minus shortest* production-run mean
  differences and quadrature SEM (independent runs, not paired trajectories);
- `scaling_density_XX_gamma_YY.png` — energy/particle and `S(G)/N` versus
  `1/N`, using the **longest** sampled production length, with both starts.

To interpret apparent ordering, inspect `replicas.csv` and start contrasts
before using the plots. Larger or more uniform peaks do not by themselves
establish spontaneous crystalline order.
