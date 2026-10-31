# Patch 8 — Two-factor DMC convergence diagnostics

The DMC benchmark studies **algorithmic bias**, not a phase boundary, superfluid
fraction, or condensate fraction. It holds the finite periodic Hamiltonian
fixed, selects each importance-sampling trial from independent VMC training
runs, and repeats held-out VMC validation and DMC production with independent
random seeds. The VMC result is a variational reference, **not** a known exact
value for the ground-state energy.

## Run

```bash
cd ~/quantum-matter-explorer
python3 -m pytest -q
python3 -m examples.bosons.run_dmc_benchmark --quick
python3 -m examples.bosons.run_dmc_benchmark
# Experimental split/RK2 kernel, convergence order unverified:
python3 -m examples.bosons.run_dmc_benchmark --scheme second_order --output outputs/bosons/dmc_split
```

Full mode is relatively expensive: by default it evaluates two trial families,
three time steps, three walker populations, and three independent replicas (54
DMC production runs). **Quick mode is just a wiring smoke test**; its energy
values and extrapolated intercepts cannot be interpreted quantitatively.

## Held-fixed physical system

We use `SoftCorePairPotential` and `DensityWaveJastrowTrial` from earlier patches
with the same simulation cell, cutoff and kinetic coefficient across all DMC
runs. With `N` particles, reduced density `rho_rc3`, cutoff `rc`, and kinetic
coefficient `D`, the cell length and pair repulsion height are

\[
L=r_c \left(\frac{N}{\rho r_c^3}\right)^{1/3},\qquad
V_0=\frac{\gamma D}{r_c^2}.
\]

The trial families are `lambda=0` (liquid-like pair Jastrow) and `lambda>0`
(exchange-symmetric density-wave-biased), **not independently established
phases**. VMC training selects the best Jastrow amplitude and density-wave
amplitude *separately* in each family. These parameters are then frozen for
all DMC timestep/population settings in that family. Fresh VMC validation
replicas measure the energy of those fixed trials without reusing training
chains. Training minima may have selection bias; the held-out runs mitigate
but do not eliminate uncertainty.

## Convergence experiment

For each scheme, walker population `W`, timestep `dt`, and replica:

1. Produce an independently seeded VMC warm-up distribution approximating
   \(|\Psi_T|^2\), used only for DMC initial walker coordinates.
2. Run DMC at **fixed** timestep and **fixed** population, no timestep
   adaptation. With `eq_steps=ceil(equilibration_tau/dt)` and
   `prod_steps=ceil(production_tau/dt)`, imaginary times are approximately
   constant across different timesteps.
3. Block the production history using
   `block_steps=ceil(block_tau/dt)`. Record *mixed local-energy* and
   *reference/growth-energy* means and within-run block errors separately.
4. Summarize across the independently seeded **replica means**; the plotted
   SEMs are the standard deviation of replica means divided by their square
   root count. Neither successive sweeps nor resampled DMC walkers are treated
   as independent observations.

The split/RK2 propagator has no Metropolis acceptance test, so its acceptance
field is `NaN`, meaning **not applicable**. Its nominal integrator construction
is not proof of second-order energy convergence. Both methods are benchmarked
with the same descriptive fitting model rather than asserting different powers.

## Descriptive extrapolation (not certified)

At each fixed trial and propagation scheme, fit the grid means using

\[
E(dt,W)/N = E_0/N + a\,dt + b/W.
\]

This is a **convenient diagnostic model**. Its intercept is *not* a defensible
zero-time-step, infinite-population estimate unless the range, functional form,
population dependence and remaining systematic errors are established. In
particular:

- Real DMC timestep bias need not be linear over the chosen range.
- Walker bias need not be proportional to `1/W` for the actual resampling
  algorithm or wavefunction.
- Cross-terms such as `dt/W` are omitted.
- Population control can induce correlations and nontrivial finite-population
  bias; independent seeds do not remove systematic errors.
- DMC equilibration may be inadequate or a state may remain metastable.
- Error bars obtained by resampling independent replica means at each grid
  cell reflect **replica variability conditional on the model**, not uncertainty
  about the model or converged energy.

Compare mixed and growth energies and repeat the study with longer imaginary
times and expanded grids before using any extrapolated value scientifically.
Neither DMC nor these estimators measure superfluidity or the one-body density
matrix. For a bosonic trial with no nodes, DMC can project toward a ground
state, but only when its approximation and equilibration biases are controlled.

## Output files

Under `outputs/bosons/dmc_benchmark/` (or `--output`):

- `training.csv`: all independent VMC selection candidates.
- `vmc_validation.csv`: held-out VMC reference energy per trial family.
- `dmc_replicas.csv`: all DMC run seeds, actual imaginary times, per-replica
  mixed/growth energies, blocked uncertainties, and acceptance diagnostics.
- `dmc_summary.csv`: replica means and SEM for each `(branch, scheme, dt, W)`.
- `extrapolation.csv`: descriptive fit coefficients and replica-only bootstrap
  uncertainty, with model name explicitly recorded.
- `bias_{branch}_{scheme}.png`: timestep and inverse-population plots,
  including the independent VMC reference.
- `protocol.json`: exact inputs, physical cell and potential parameters,
  Python/NumPy versions, seed policy, and interpretation warning.

The two fitted trial families are compared as **importance sampling choices**,
not proof of competing thermodynamic phases. Do not interpret a fit intercept
below or above the VMC estimate as a definitive result on its own.
