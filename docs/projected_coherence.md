# Patch 11 — mixed and extrapolated DMC-assisted coherence

This patch **does not implement a pure DMC one-body density matrix**. It uses
Patch 9's trial-wavefunction displacement ratio and the already supported VMC
and DMC observable hooks to compare three different quantities. An independent
forward-walking estimator of a *diagonal* observable would not suffice to make
this off-diagonal estimator pure: the ratio still contains the trial wavefunction.
A genuinely pure ground-state OBDM requires a different estimator/algorithm,
e.g. an appropriately validated open-path or reptation method.

## Derivation and caveats

For a strictly positive real bosonic trial wavefunction, let `O` be the
Hermitian one-body translation operator averaged over all particles and either
isotropic displacement directions (radial `g1`) or the full cell (zero-momentum
occupation). Its local ratio is

    O_L(R) = (1/N) sum_i [Psi_T(R with particle i translated) / Psi_T(R)] .

The existing `OneBodyDensityMatrix` and `VariationalCondensateOccupation`
observers perform stochastic sampling over particle indices and displacements.
When attached to VMC, their expectation is

    V = <Psi_T | O | Psi_T> / <Psi_T | Psi_T>.

When attached **after resampling** to the importance-sampled DMC production
walkers, the same local ratio yields the mixed estimate

    M = <Psi_0 | O | Psi_T> / <Psi_0 | Psi_T>,

in the ideal converged DMC limit. Fixed population, finite projector time,
finite time step and correlated walker genealogies add errors. Mixed estimates
are **not** ground-state pure estimates for off-diagonal operators.

For real Hermitian `O`, if `Psi_T` is close to the nodeless `Psi_0`, the familiar

    X = 2*M - V

cancels the **first-order trial-wavefunction error**, leaving a generally
unquantified second-order trial bias. This perturbative argument does not
remove DMC projection/time-step/population bias; a strongly inaccurate trial
can make X unphysical. No clipping is applied to negative estimates or values
above one. An observed estimate exceeding [0, 1] is an indication to examine
statistical/systematic uncertainty rather than a physical occupation fraction.

The scalar occupation is the *full-periodic-cell* displacement integral
`n0/N = (1/V) integral_cell g1(s) d^3s`, not the last point of radial `g1(r)`.
All coherence values are estimates at the finite particle number N.

## Experimental protocol

The reference Hamiltonian is the smooth finite-range soft-core repulsive Bose
Hamiltonian from Patch 4. Each branch uses **fixed** Jastrow and density-wave
trial parameters; this benchmark does *not* reoptimize them. The liquid branch
sets `lambda=0` while the modulated branch uses the configured positive lambda.
These trial states need not correspond to distinct thermodynamic phases.

For each branch and independent replica:

1. Generate a uniformly initialized VMC chain with a unique seed. Save the
   variational OBDM, zero-momentum occupation, energy, and directional S(G)/N.
2. For each `(dt, n_walkers, equilibration_tau)` point, generate a **fresh** DMC
   population using a different seed and the same trial and Hamiltonian.
3. Turn off adaptive timestep, discard `ceil(tau_eq/dt)` projection steps, and
   sample after fixed-population resampling for `ceil(tau_prod/dt)` steps.
4. Form the **paired replica** extrapolation `2*mixed - VMC`. Report means and
   `std(replica estimates, ddof=1)/sqrt(n_replicas)` across independent runs.
5. Record the *actual* projection times (because `ceil` discretizes them), DMC
   acceptance and energy as diagnostics, and compare multiple populations and
   timesteps without claiming an automatic zero-bias extrapolation.

Note: the same VMC replica baseline is reused across all DMC grid points,
introducing **cross-grid covariance**. Error bars for an individual point are
computed from replica-wise extrapolations, so that covariance within each
paired extrapolation is respected; error bars across grid points cannot be
interpreted as independent. At a small number of replicas, all error estimates
are themselves uncertain. DMC walker's genealogies and sampling times are
correlated; counting each walker as an independent replica would be incorrect.

The selected `G` vectors are preselected diagnostics, not proof of crystallinity
or superfluidity. The estimated S(G)/N is diagonal and mixed in DMC; the OBDM
is off-diagonal and needs the above trial-ratio qualification.

## Run

```bash
cd ~/quantum-matter-explorer
python3 -m pytest -q
python3 -m examples.bosons.run_projected_coherence --quick
python3 -m examples.bosons.run_projected_coherence
# Optional alternative (experimental, not verified second-order DMC bias):
python3 -m examples.bosons.run_projected_coherence --quick --scheme second_order
```

Full settings include both trial families, two timesteps, two walker populations,
two equilibration durations, and three independent replicas. The quick example
uses a smaller system and one trial family.

Results under `outputs/bosons/projected_coherence/`:

- `replicas.csv`: individual VMC/mixed/extrapolated occupations, S diagnostics,
  DMC acceptance/energy, and exact independent seeds.
- `summary.csv`: condition-wise means and independent-replica standard errors.
- `g1.csv`: radial VMC, mixed and extrapolated curves with replica error bars.
- `occupation_{branch}.png`: projection-time sensitivity at dt/population grid.
- `g1_{branch}.png`: representative coherent curves (not a pure-estimator limit).
- `protocol.json`: every numerical setting and explicit estimator caveats.

## Interpreting results

A stable X after changing projection time, population and timestep is reassuring
but **cannot alone establish a pure ground-state condensate fraction**.
Check sensitivity to the trial family, independent seeds, production length,
and larger particle counts. The next major algorithmic milestone is a properly
validated pure off-diagonal estimator, beyond the current VMC/DMC sampling hooks.
