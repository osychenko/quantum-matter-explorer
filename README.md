# Quantum Matter Explorer

An experimental, NumPy-based quantum Monte Carlo (QMC) engine intended to
become the numerical foundation for an educational quantum-matter simulator.
The numerical VMC/DMC engine now accepts arbitrary configuration shapes.
Periodic simulation-cell geometry is now implemented independently of VMC/DMC.
Many-body bosonic trial states and educational visualizations are future milestones.

## Current capabilities

- Metropolis variational Monte Carlo (VMC) with trial-parameter scans.
- Importance-sampled, fixed-population diffusion Monte Carlo (DMC).
- Metropolized drift-diffusion and an experimental split/RK2 propagator.
- Block-based energy uncertainty estimates and timestep-convergence scans.
- Headless Matplotlib diagnostics (PNG) and CSV exports.
- A cusp-preserving trial wavefunction for the attractive 3D Yukawa problem.
- Cubic, orthorhombic, general Bravais, and BCC/truncated-octahedral PBC geometry.

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
| `qmc/cells.py` | 3D periodic geometry, minimum-image distances, reciprocal vectors |
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

## Periodic simulation cells (Patch 3)

Cell geometry lives in `qmc/cells.py`; the VMC/DMC propagators do not need
periodic-specific code. Pair potentials, Jastrow factors, and observables
should all use the **same** cell to calculate pair displacements:

```python
import numpy as np
from qmc.cells import CubicCell, BCCCell

box = CubicCell(10.0)
r_i = np.array([9.0, 1.0, 1.0])
r_j = np.array([1.0, 2.0, 1.0])
dr = box.displacement(r_i, r_j)  # [2, 1, 0], not [-8, 1, 0]
volume = box.volume                # 1000

# BCC *translation lattice*. The Wigner-Seitz (Voronoi) cell is a
# truncated octahedron; `length` is the conventional cubic lattice constant.
to = BCCCell(10.0)
assert np.isclose(to.volume, 500.0)
inside_truncated_octahedron = to.wrap(r_i, centered=True)
```

Cell classes: `PeriodicCell(vectors)` for any nonsingular 3x3 lattice,
`OrthorhombicCell((Lx, Ly, Lz))`, `CubicCell(L)`, and `BCCCell(L)`.
The primitive lattice vectors are **rows** of `cell.vectors`; positions
have Cartesian coordinates in the last array axis. Operations broadcast to
arrays with shapes such as `(walkers, particles, 3)`.

- `cell.wrap(r)` maps positions into the primitive parallelepiped.
- `cell.wrap(r, centered=True)` maps to the Wigner-Seitz cell.
- `cell.minimum_image(dr)` gives the shortest periodic image of `dr`.
- `cell.displacement(r_i, r_j)` gives the shortest displacement from `i` to `j`.
- `cell.volume` is the primitive simulation-cell volume.
- `cell.reciprocal_vectors` provides reciprocal primitive vectors as rows;
  `cell.wavevectors(indices)` returns requested allowed k-vectors.

For the BCC cell, **periodicity is BCC** but the particles are not required to
have BCC crystalline order. Its centered Wigner-Seitz region has the truncated-
octahedral shape; default wrapping uses the equivalent primitive parallelepiped.
The BCC minimum image is evaluated exactly by comparing integer and half-integer
cubic translations. For general skew cells, the exact nearest translation can
require more than componentwise rounding in fractional coordinates; the generic
method therefore performs a bounded nearest-image search (slower than the
specialized cubic/BCC methods).

**Important:** this patch supplies geometry, not a physical periodic Hamiltonian.
Periodic pair interactions, their cutoff/long-range-image treatment, and periodic
many-body trial wavefunctions belong in the next physics milestone. In
particular, merely wrapping DMC positions does not implement a periodic
Hamiltonian, and minimum-image truncation alone is not sufficient for
long-range Coulomb interactions.

**Patch 4 — periodic soft-core bosons:** `qmc/bosons.py` provides a smooth,
finite-range repulsive pair potential and a particle-exchange-symmetric Jastrow
trial wavefunction with analytic gradient and Laplacian. Both use the periodic
minimum-image convention. `VMCConfig(proposal_mode="particle")` proposes moves
one particle at a time; a step is one complete sweep over all particles.
The default `proposal_mode="all"` preserves the original single-particle and
many-dimensional test workflows. For the new Jastrow trial, the particle
proposal uses O(N) incremental logarithmic wavefunction ratios, rather than
recomputing the O(N²) full pair product.

**Run the small interacting-boson example:**

```bash
python3 -m examples.bosons.run_soft_core
```

It reports VMC energy per particle and sampling acceptance for eight bosons
in a periodic cubic cell. This is an illustrative finite-range *soft-core*
repulsion, not an unregularized Yukawa interaction. The compactly supported
pair factor has two continuous derivatives at its cutoff, and both the
interaction and Jastrow must use `cutoff <= cell inradius` (for a cubic box,
`cutoff <= L/2`). For highly skew general cells, the cutoff validation uses a
conservative singular-value bound. Pair sums use one minimum image per pair:
this is not an Ewald or periodic-image-summed treatment of long-range forces.

**Not yet implemented:** production many-body DMC methodology/benchmarks,
BEC/OBDM estimators, and crystalline trial states. A smooth bounded repulsion alone does not guarantee a simple
liquid-to-ordinary-crystal transition as parameters change.

The current Yukawa cusp trial is tuned for `D=0.5`. With singular Coulomb-like
pair or external potentials, the cusp condition itself depends on `D`; changing
mass without adapting that trial can create a divergent local energy at short
distances. The independent-oscillator tests cover general `D` safely.

## Development roadmap

1. **v0.1 cleanup**: packaging, documentation, CSV correctness, baseline tests.
2. **v0.2 many-body-ready core**: generic walker configurations and explicit kinetic
   coefficient, with existing single-particle examples preserved. Periodic
   geometry (Patch 3) is now available independently of the QMC algorithms.
3. **Many-body examples**: Patch 4 supplies a periodic interacting-boson
   baseline; Patch 5 adds pair correlation `g(r)` and static structure factor
   `S(k)` as production-sampled observables.
4. **Quantum phases and coherence**: crystal/liquid energy branches, OBDM/BEC
   estimators, and finite-size/time-step analysis.
5. **Education layer**: interactive visualizations, learning missions, saved
   experiments, and guided diagnostics.

## Patch 5 — structural observables and cell visualization

The `qmc.observables` module supplies normalized `PairCorrelation` and
reciprocal-lattice `StaticStructureFactor` accumulators for VMC/DMC production
walkers. The optional `observables=(g, s)` keyword argument makes it possible
to collect time-series samples during either solver rather than only analyzing
the final walkers. For DMC, `observable_stride=10` may reduce overhead.

Run the plot-producing example:

```bash
python3 -m examples.bosons.run_observables
```

It writes a 3D cell snapshot, `g(r)` and directional `S(k)` scatter plots to
`outputs/bosons/`. BCC cell snapshots show a truncated-octahedral boundary.
Read [docs/observables.md](docs/observables.md) for finite-size normalization,
statistical caveats, and important differences between VMC and DMC estimators.

## Patch 6 — variational density-wave exploration

An exchange-symmetric density-wave-enhanced Jastrow trial and an independent-
validation parameter scan now provide a first comparison of a liquid-like
baseline (`lambda=0`) and a density-modulated candidate (`lambda>0`). The
control parameter `gamma = V0*rc**2/D` varies repulsion relative to kinetic
fluctuations at fixed reduced density `rho*rc**3`.

```bash
python3 -m examples.bosons.run_phase_scan --quick
# Larger scan:
python3 -m examples.bosons.run_phase_scan
```

Results are saved to `outputs/bosons/phase_scan/`, including grid training
energies, independently seeded validation replicas, structure factors at
preselected reciprocal modes, pair correlations, cell snapshots, and a scan
figure. See [docs/phase_scans.md](docs/phase_scans.md) for the ansatz, physics,
and statistical limitations.

**Caution:** selected density-wave peaks or finite-size variational-energy
crossings do not establish a liquid–solid phase transition. Soft-core
interactions may favor cluster ordering; distinguish ordinary crystals from
cluster crystals by additional correlations and finite-size studies.

## Licensing

No license has been selected yet. A public GitHub repository does **not** by
itself grant permission to redistribute or modify its code; choose and add a
license deliberately before inviting external reuse or contributions.
