# Quantum Matter Explorer

An experimental, NumPy-based quantum Monte Carlo (QMC) engine intended to
become the numerical foundation for an educational quantum-matter simulator.
The numerical VMC/DMC engine now accepts arbitrary configuration shapes.
Many-body bosonic trial states, periodic boundaries and educational visualizations
are future milestones.

## Current capabilities

- Metropolis variational Monte Carlo (VMC) with trial-parameter scans.
- Importance-sampled, fixed-population diffusion Monte Carlo (DMC).
- Metropolized drift-diffusion and an experimental split/RK2 propagator.
- Block-based energy uncertainty estimates and timestep-convergence scans.
- Headless Matplotlib diagnostics (PNG) and CSV exports.
- A cusp-preserving trial wavefunction for the attractive 3D Yukawa problem.

The split/RK2 propagator is **not yet certified as second-order DMC**: its
time-step convergence must be validated independently before relying on a
quadratic-bias claim.

## Install

From the repository root, with Python 3.10 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e '.[dev]'
python3 -m pytest -q
```

The `dev` extra adds pytest. The runtime requirements are NumPy and Matplotlib.

## Run the Yukawa experiment

```bash
python3 -m examples.yukawa.run_yukawa
```

This is a **full experiment**, not a quick smoke test: it performs coarse and
fine VMC optimization, independent validation, a two-propagator fixed-imaginary-
time DMC scan, and a final DMC production run. It can be computationally expensive.
Results are written to `outputs/yukawa/` (ignored by Git).

The example uses atomic units (`hbar = m = 1`) and the attractive potential

`V(r) = -g exp(-mu r) / r`.

The Yukawa trial preserves the radial cusp while VMC optimizes its two
parameters, `alpha` and `beta`. The harmonic-oscillator regression tests provide
an independent analytic reference with an exact Gaussian trial.

## Layout

| Path | Purpose |
| --- | --- |
| `qmc/vmc.py` | VMC sampling, scans, and current CSV writers |
| `qmc/dmc.py` | DMC driver and fixed-imaginary-time timestep scans |
| `qmc/propagators.py` | DMC drift-diffusion moves |
| `qmc/evaluation.py` | Potential, local energy, and trial evaluation |
| `qmc/trials.py` | Single-particle trial wavefunctions |
| `qmc/potentials.py` | Single-particle test potentials |
| `qmc/resampling.py` | Fixed-population systematic resampling |
| `qmc/stats.py` | Block statistics |
| `qmc/plotting.py` | PNG plots without a GUI |
| `examples/yukawa/run_yukawa.py` | End-to-end Yukawa experiment |
| `tests/` | Fast scientific and data-format regression tests |

## Configuration shapes and kinetic coefficient (v0.2)

A walker always represents a **complete quantum configuration**. Set
`configuration_shape=(3,)` for the existing one-body examples, or e.g.
`configuration_shape=(16, 3)` for 16 particles in three dimensions. Trial
functions and potential functions must implement the corresponding batched
input/output conventions:

- `walkers`: `(n_walkers, *configuration_shape)`
- `trial.log_psi(walkers)`: `(n_walkers,)`
- `trial.grad_log_psi(walkers)`: same shape as `walkers`
- `trial.laplacian_log_psi(walkers)`: `(n_walkers,)`
- `potential(walkers)`: `(n_walkers,)`

`VMCConfig` and `DMCConfig` both accept `kinetic_coefficient=D`, where the
kinetic operator is `-D * Laplacian` and `D = hbar**2 / (2*m)` in ordinary
units. The default `D=0.5` preserves the original behavior. DMC uses drift
`2*D*grad(log(Psi_T))` and Gaussian increments of standard deviation
`sqrt(2*D*dt)`. Existing trial-specific `local_energy()` shortcuts are only
used for the original `D=0.5`; otherwise generic log-derivatives are used.

**Not yet implemented:** periodic cells, pair interactions, Jastrow trial
functions, particle-by-particle VMC proposals, and BEC/structure observables.
The generic VMC proposal still moves all configuration coordinates at once;
its efficiency for many particles will be improved in the next milestone.

The current Yukawa cusp trial is tuned for `D=0.5`. With singular Coulomb-like
pair or external potentials, the cusp condition itself depends on `D`; changing
mass without adapting that trial can create a divergent local energy at short
distances. The independent-oscillator tests cover general `D` safely.

## Development roadmap

1. **v0.1 cleanup**: packaging, documentation, CSV correctness, baseline tests.
2. **v0.2 many-body-ready core**: generic walker configurations and explicit kinetic
   coefficient, with existing single-particle examples preserved. Cell geometry
   and periodic boundary conditions are the next independent milestone.
3. **Many-body examples**: repulsive bosons, Jastrow trials, PBC, pair correlation
   `g(r)` and static structure factor `S(k)`.
4. **Quantum phases and coherence**: crystal/liquid energy branches, OBDM/BEC
   estimators, and finite-size/time-step analysis.
5. **Education layer**: interactive visualizations, learning missions, saved
   experiments, and guided diagnostics.

## Licensing

No license has been selected yet. A public GitHub repository does **not** by
itself grant permission to redistribute or modify its code; choose and add a
license deliberately before inviting external reuse or contributions.
