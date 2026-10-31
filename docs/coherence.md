# Patch 9: variational one-body coherence and zero-momentum occupation

This patch provides **trial-state** off-diagonal observables for positive real,
periodic bosonic wavefunctions. It is **not** a pure ground-state DMC estimator,
and finite-box occupations should not be interpreted as proof of a
thermodynamic Bose–Einstein condensate.

## Definitions

With normalized positive wavefunction `Psi_T(R)`, where `R` contains the
positions of `N` indistinguishable bosons and samples are distributed as
`|Psi_T(R)|^2`, the translationally averaged normalized one-body density
matrix can be written as a ratio expectation:

\[
g_1(\mathbf s) = \frac{\rho_1(\mathbf s)}{\rho}
 = \left\langle \frac{1}{N}\sum_{i=1}^N
\frac{\Psi_T(r_1,\ldots,r_i+\mathbf s,\ldots,r_N)}{\Psi_T(R)}
\right\rangle_{|\Psi_T|^2}.
\]

All displacements and particle positions are periodic. `OneBodyDensityMatrix`
selects random particles and isotropically oriented vectors of specified
lengths, and reports their angular average `g1(r)`. The zero-displacement
identity `g1(0)=1` is exact. The radial grid is restricted to the cell's
Wigner–Seitz inradius (or a conservative bound for a general lattice), so
`r` is an unambiguous separation inside the simulation cell. **Anisotropic
modulated states can have direction-dependent coherence**, which an angular
average hides.

The finite-cell occupation fraction of the zero-momentum orbital is instead

\[
\frac{n_{\mathbf k=0}}{N} = \frac{1}{V}\int_{\rm cell}d^3s\;g_1(\mathbf s).
\]

`VariationalCondensateOccupation` estimates this volume integral using
*uniform fractional displacement vectors*, mapped through the primitive cell
vectors. This works for cubic, orthorhombic, BCC, and general Bravais cells.
The estimator is **not** `g1(r_max)`; equality to an asymptotic plateau is a
thermodynamic-limit statement with important finite-size restrictions.

For an ideal zero-temperature gas, `Psi_T=constant` and both `g1(r)=1` and
`n_{k=0}/N=1` exactly. For the analytic one-particle periodic orbital
`f(x)=1+a cos(2*pi*x/L)`, the zero-mode fraction is
`1/(1+a*a/2)`; this provides a nontrivial numerical integration check.

## API

```python
from qmc.coherence import OneBodyDensityMatrix, VariationalCondensateOccupation
import numpy as np

radii = np.linspace(0, L/2, 13)
g1 = OneBodyDensityMatrix(trial, cell, n_particles, radii,
                          draws_per_configuration=2, seed=13)
n0 = VariationalCondensateOccupation(trial, cell, n_particles,
                                     draws_per_configuration=2, seed=14)
result = run_vmc(potential, trial, vmc_config, initial_walkers=initial,
                 observables=(g1, n0))
print(g1.result().g1)
print(n0.result().n0_over_n)
```

Each observer retains running accumulations. **Create new observers for each
independent simulation replica.** The optional fast
`trial.log_ratio_particle_move()` is used when available; otherwise the
trial's `log_psi()` is reevaluated. Neither move mutates the original walker
configurations or the trial. Drawing one random particle per walker per
measurement is an unbiased way to estimate the particle average without
paying for all N particles at each radius.

## Error bars and interpretation

- Observers provide no standard errors internally. Walker and successive-sweep
  samples are correlated. The example combines *independent VMC run means*
  and uses `std(replica_values,ddof=1)/sqrt(n_replicas)` for the displayed
  standard errors. Two or three replicas only support smoke tests, **not**
  confident statistical statements. Production work requires more replicas,
  autocorrelation/blocked uncertainty studies, convergence tests, and
  multiple sizes and densities.
- The ratio estimator can have large variance, especially at strong
  correlations. Uncertainty estimates may be unstable. Raw finite-sample
  estimates of `n0/N` can lie outside `[0,1]` due to statistical noise and
  are **not clipped**. The physical occupation fraction lies in `[0,1]`.
- All reported values belong to the *variational trial state* `Psi_T`. Do not
  pass these observers into the DMC mixed-distribution hook and interpret them
  as unbiased `Psi_0` off-diagonal expectations. Pure DMC off-diagonal
  estimates require a separate method (e.g. worm/open-path approaches or
  carefully justified forward-walking/reptation variants).
- The density-wave trial uses selected reciprocal vectors and may generate
  clustering or orientational anisotropy. A suppressed zero-momentum fraction
  or directional structure peak alone does not uniquely identify a crystal,
  superfluid, or phase transition. Superfluid response is **not measured**.
- `n_{k=0}/N` is a specific *finite-size momentum occupation*. Establishing a
  thermodynamic condensate fraction additionally needs careful N-scaling.

## Example

From `~/quantum-matter-explorer`:

```bash
python3 -m pytest -q
python3 -m examples.bosons.run_coherence --quick
python3 -m examples.bosons.run_coherence
```

The three illustrative cases are: (1) an ideal-gas analytic reference with
zero pair repulsion, (2) a correlated soft-core Jastrow trial at repulsion
height 3, and (3) a density-wave-biased trial for the same interacting
Hamiltonian. **The ideal-gas case has a different Hamiltonian** and should not
be used in a variational energy comparison with the other two cases. The
other trial parameters are illustrative, not reoptimized in this patch.

The example writes `replicas.csv`, `summary.csv`, `obdm.csv`, `protocol.json`,
`obdm.png`, and `occupation.png` under `outputs/bosons/coherence/` (or your
`--output` directory). Use the raw replica records to inspect convergence.

## Follow-up scientific work

First: optimize trial parameters and repeat the coherence experiment for
several N/densities with more independent chains; compare angular directions
as well as radial averages in ordered states. Later: develop a ground-state
unbiased off-diagonal method and distinguish condensation from superfluid
response. Neither is inferred by this patch.
