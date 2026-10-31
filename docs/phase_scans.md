# Patch 6 — finite-size VMC scans of density modulation

This is an **exploratory** scan of finite-range repulsive soft-core bosons, not
an established quantum phase boundary or a calculation of superfluidity/BEC.

## Physical control parameters

The periodic many-boson model is

\[
H=-D\sum_i\nabla_i^2+\sum_{i<j}V_0 f(r_{ij}/r_c),\quad
f(x)=(1-x^2)^3\; (0\leq x<1),\quad f(x)=0\; (x\geq1).
\]

The scan parameter `gamma = V0 * cutoff**2 / D` is dimensionless. At fixed
particle count `N`, box volume `V`, and cutoff `r_c`, the reduced density is
`rho_rc3 = (N/V)*r_c**3`. The current scan varies gamma at **fixed density**:
changing gamma changes the repulsion height, not particle spacing. Varying
`D` would require re-tuning all variational parameters; the scan automatically
recomputes `V0` from the selected `D` and gamma.

A stable thermodynamic phase characterization would require scans in both
coupling and density as well as finite-size and time-step analyses.

## Exchange-symmetric order-biased trial

The baseline is the smooth pair Jastrow trial from Patch 4. The enlarged
variational family is

\[
\ln \Psi(R) = \ln\Psi_J(R) + \frac{\lambda}{2N}\sum_{G\in K}
\left|\rho_G(R)\right|^2,\quad
\rho_G(R)=\sum_j e^{iG\cdot r_j}.
\]

`K` contains one representative of each `+/-G` reciprocal-lattice pair;
passing both would duplicate the same density-wave term and is rejected.
The default `(2,0,0)`, `(0,2,0)`, `(0,0,2)` modes describe a trial compatible
with two repetitions across each cubic axis. They **bias** the system toward
chosen commensurate density modulations; they do not pin particles to sites
or assign distinguishable labels. Global translations and particle exchanges
leave the trial invariant.

`lambda=0` reduces exactly to the liquid-like pair Jastrow baseline.
The `lambda>0` family can generate a modulated density, **including clusters**;
it is not automatically an ordinary single-occupancy crystal. By selecting a
fixed set of G modes, the trial also favors particular orientations.

The analytical `grad_log_psi`, `laplacian_log_psi`, and incremental
`log_ratio_particle_move` methods are tested against finite differences and
full wavefunction evaluations. Both VMC and DMC can evaluate the trial;
Patch 6 runs variational VMC only.

## Training and independent validation

For each gamma, the scan searches a grid of Jastrow amplitudes and lambda
values using independent training seeds. It chooses the lowest estimated
training energy separately among `lambda=0` (liquid baseline) and
`lambda>0` (modulated candidate). **These are variational trial families,
not established thermodynamic phases.** Grid minima from short chains may
have selection bias, so fresh seeds are used for validation.

For each selected trial, two or more independent validation runs measure

- energy per particle and uncertainty across independent *replica means*;
- the mean directional structure factor at the **preselected** G vectors;
- radial pair correlation g(r), move acceptance, and representative cell plots.

The reported SEM is `std(replica_means, ddof=1)/sqrt(n_replicas)`; a small
number of replicas yields an unstable error estimate. Individual walkers
and successive sweeps are **not** treated as independent measurements.
Within-run block-stderr values guide parameter selection but are not used as
proof of significance. If runs remain trapped in metastable regions, changing
only random seeds may still underestimate uncertainty. Convergence and
independent initial states must be examined before interpreting results.

The `S_mode` summary averages the *fixed selected G directions*; a high
value is a **density-modulation indicator**, not a unique crystal diagnostic.
Do not identify a phase transition from a crossing of these noisy finite-N
variational energies or from a single peak. More complete structure-factor
maps, distinct initializations, and N scaling are needed.

## Run

From `~/quantum-matter-explorer`:

```bash
python3 -m pytest -q
python3 -m examples.bosons.run_phase_scan --quick
python3 -m examples.bosons.run_phase_scan
```

The default scan is more expensive; use `--quick` for a short end-to-end test.
Use `--output PATH` for a custom results directory.

Results under `outputs/bosons/phase_scan/`:

- `candidates.csv`: all training grid points and block-based estimates;
- `validation.csv`: held-out mean energy and S at selected modes, with
  between-replica standard errors;
- `replicas.csv`: each validation seed and raw per-replica measurements;
- `scan.png`: energy/particle and mean selected-mode S versus gamma;
- `gr_gamma_XX_*.csv`, `sk_gamma_XX_*.csv`: replicated structural estimates;
- `cell_gamma_XX_*.png`: example Monte Carlo snapshots (not trajectories).

### Next scientific steps

Systematically vary N, the density, trial reciprocal mode sets, and starting
configurations; extend the variational comparison to optimized DMC with
finite-population/time-step checks. After that, implement off-diagonal
one-body density matrix estimators and study BEC, which cannot be inferred
from g(r) or S(k) alone.
