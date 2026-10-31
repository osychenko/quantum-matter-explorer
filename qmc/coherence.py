"""Variational off-diagonal one-body coherence for periodic bosonic trial states.

For real, strictly positive trial wavefunctions sampled under |Psi_T(R)|**2,

    g1(s) = < (1/N) sum_i Psi_T(R with r_i -> r_i+s) / Psi_T(R) > .

The radial estimator averages orientations on a sphere of radius r.  The
zero-momentum occupation fraction instead integrates g1(s) over a uniform
vector s in the primitive periodic cell:

    n0/N = (1/V) integral_cell g1(s) d^3s .

Both are *variational trial-state* estimates.  They are not DMC pure estimators,
and g1's finite-r plateau need not be n0/N at finite cell size.

The accumulators do not provide uncertainty estimates. Use independent VMC
runs and compute replica-level uncertainty. Particle and walker samples from
one run are correlated and are not independent error-bar observations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cells import PeriodicCell
from .observables import safe_radial_limit


def _walkers(values, n_particles: int) -> np.ndarray:
    R = np.asarray(values, dtype=np.float64)
    if (R.ndim != 3 or R.shape[0] < 1 or R.shape[1:] != (n_particles, 3)
            or not np.all(np.isfinite(R))):
        raise ValueError(f"positions must be finite (walkers, {n_particles}, 3) array")
    return R


def one_body_ratio(trial, positions, particle_indices, candidate_positions):
    """Evaluate Psi(R with selected particle moved) / Psi(R), one move/walker.

    Parameters
    ----------
    positions : (W,N,3) array
    particle_indices : (W,) integer array
    candidate_positions : (W,3) array

    Does not mutate input. Uses the trial's incremental log-ratio when present,
    otherwise computes the trial log wavefunction before and after the move.
    The returned ratio is for positive real trial functions, not complex phases.
    """
    R = np.asarray(positions, dtype=np.float64)
    if R.ndim != 3 or R.shape[0] == 0 or R.shape[-1] != 3:
        raise ValueError("positions must have shape (W, N, 3)")
    R = _walkers(R, R.shape[1])
    W, N, _ = R.shape
    indices = np.asarray(particle_indices)
    if (indices.shape != (W,) or not np.issubdtype(indices.dtype, np.integer)
            or np.any(indices < 0) or np.any(indices >= N)):
        raise ValueError("particle_indices must be (W,) integers in [0,N)")
    new = np.asarray(candidate_positions, dtype=np.float64)
    if new.shape != (W, 3) or not np.all(np.isfinite(new)):
        raise ValueError("candidate_positions must have shape (W,3) and be finite")
    logratios = np.empty(W, dtype=np.float64)
    if hasattr(trial, "log_ratio_particle_move"):
        # The existing fast trial API accepts a common particle index per batch.
        # Partition walkers by chosen index to obtain one random particle/walker.
        for i in np.unique(indices):
            selected = indices == i
            computed = np.asarray(
                trial.log_ratio_particle_move(R[selected], int(i), new[selected]),
                dtype=np.float64,
            )
            if computed.shape != (int(selected.sum()),):
                raise ValueError("log_ratio_particle_move must return shape (W_selected,)")
            logratios[selected] = computed
    else:
        if not hasattr(trial, "log_psi"):
            raise TypeError("trial must implement log_psi (and optionally log_ratio_particle_move)")
        old_log = np.asarray(trial.log_psi(R), dtype=np.float64)
        proposed = R.copy()
        proposed[np.arange(W), indices] = new
        new_log = np.asarray(trial.log_psi(proposed), dtype=np.float64)
        if old_log.shape != (W,) or new_log.shape != (W,):
            raise ValueError("trial.log_psi must return one scalar per walker")
        logratios = new_log - old_log
    if not np.all(np.isfinite(logratios)):
        raise FloatingPointError("nonfinite off-diagonal trial log ratio")
    with np.errstate(over="ignore", invalid="ignore"):
        ratio = np.exp(logratios)
    if not np.all(np.isfinite(ratio)):
        raise FloatingPointError("off-diagonal trial ratio overflowed")
    return ratio


def _check_base(trial, cell, n_particles, draws, seed):
    if not isinstance(cell, PeriodicCell):
        raise TypeError("cell must be a PeriodicCell")
    if not isinstance(n_particles, (int, np.integer)) or n_particles < 1:
        raise ValueError("n_particles must be a positive integer")
    if not isinstance(draws, (int, np.integer)) or draws < 1:
        raise ValueError("draws_per_configuration must be a positive integer")
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if hasattr(trial, "cell"):
        if (not isinstance(trial.cell, PeriodicCell)
                or not np.allclose(trial.cell.vectors, cell.vectors, rtol=1e-12, atol=0)):
            raise ValueError("observer cell must agree with the trial's periodic cell")
    if not hasattr(trial, "log_ratio_particle_move") and not hasattr(trial, "log_psi"):
        raise TypeError("trial must implement log_psi or log_ratio_particle_move")


@dataclass(frozen=True)
class OneBodyDensityMatrixResult:
    radius: np.ndarray
    g1: np.ndarray
    n_configurations: int
    n_displacement_samples_per_radius: int


class OneBodyDensityMatrix:
    """Radial, angular-averaged g1(r)=rho1(r)/rho for a VMC trial state.

    One random particle per walker and random isotropic displacement direction
    is selected in each draw. The radius list must include r=0; g1(0)=1 is
    exact and not sampled. Radii must fit the Wigner-Seitz inradius to avoid
    ambiguous radial displacement lengths in the periodic cell.
    """

    def __init__(self, trial, cell: PeriodicCell, n_particles: int, radii,
                 *, draws_per_configuration: int = 1, seed: int = 2026):
        _check_base(trial, cell, n_particles, draws_per_configuration, seed)
        values = np.asarray(radii, dtype=np.float64)
        limit = safe_radial_limit(cell)
        if (values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values))
                or values[0] != 0.0 or np.any(np.diff(values) <= 0)
                or values[-1] > limit * (1.0 + 1e-12)):
            raise ValueError(f"radii must start at zero, increase strictly and end <= {limit:g}")
        self.trial = trial
        self.cell = cell
        self.n_particles = int(n_particles)
        self.radii = values.copy()
        self.draws_per_configuration = int(draws_per_configuration)
        self.rng = np.random.default_rng(int(seed))
        self._totals = np.zeros(len(values), dtype=np.float64)
        self.n_configurations = 0
        self.n_displacement_samples_per_radius = 0

    def accumulate(self, positions):
        R = _walkers(positions, self.n_particles)
        W = len(R)
        new_totals = np.zeros_like(self._totals)
        new_totals[0] = W * self.draws_per_configuration
        for _ in range(self.draws_per_configuration):
            for j, r in enumerate(self.radii[1:], start=1):
                selected = self.rng.integers(self.n_particles, size=W)
                directions = self.rng.normal(size=(W, 3))
                norms = np.linalg.norm(directions, axis=1)
                # A precisely zero normal triplet has negligible probability,
                # but avoid NaN even for an adversarial random generator.
                while np.any(norms == 0):
                    directions[norms == 0] = self.rng.normal(size=(np.count_nonzero(norms == 0), 3))
                    norms = np.linalg.norm(directions, axis=1)
                directions /= norms[:, None]
                original = R[np.arange(W), selected]
                candidate = self.cell.wrap(original + float(r) * directions)
                new_totals[j] += float(np.sum(one_body_ratio(self.trial, R, selected, candidate)))
        # Commit a complete batch only after all ratios have succeeded.
        self._totals += new_totals
        self.n_configurations += W
        self.n_displacement_samples_per_radius += W * self.draws_per_configuration

    def result(self) -> OneBodyDensityMatrixResult:
        if self.n_configurations == 0:
            raise ValueError("no configurations accumulated")
        result = self._totals / self.n_displacement_samples_per_radius
        result[0] = 1.0
        return OneBodyDensityMatrixResult(
            radius=self.radii.copy(), g1=result,
            n_configurations=self.n_configurations,
            n_displacement_samples_per_radius=self.n_displacement_samples_per_radius,
        )


@dataclass(frozen=True)
class CondensateOccupationResult:
    n0_over_n: float
    n_configurations: int
    n_displacement_samples: int


class VariationalCondensateOccupation:
    """Monte Carlo integral of g1(s)/V over the full periodic cell.

    Random fractional displacements are uniform in the *primitive* cell, for
    every orthorhombic, BCC and general Bravais cell. This returns the finite-
    volume k=0 mode occupancy fraction for the |Psi_T|^2 variational state.
    Values can exceed [0,1] due to sampling noise: no clipping is performed.
    """

    def __init__(self, trial, cell: PeriodicCell, n_particles: int,
                 *, draws_per_configuration: int = 1, seed: int = 2027):
        _check_base(trial, cell, n_particles, draws_per_configuration, seed)
        self.trial = trial
        self.cell = cell
        self.n_particles = int(n_particles)
        self.draws_per_configuration = int(draws_per_configuration)
        self.rng = np.random.default_rng(int(seed))
        self._total = 0.0
        self.n_configurations = 0
        self.n_displacement_samples = 0

    def accumulate(self, positions):
        R = _walkers(positions, self.n_particles)
        W = len(R)
        batch_total = 0.0
        for _ in range(self.draws_per_configuration):
            selected = self.rng.integers(self.n_particles, size=W)
            # Cell.to_cartesian transforms uniform fractional positions to
            # uniform Cartesian volume for any fixed Bravais lattice.
            displacement = self.cell.to_cartesian(self.rng.random((W, 3)))
            original = R[np.arange(W), selected]
            candidate = self.cell.wrap(original + displacement)
            batch_total += float(np.sum(one_body_ratio(self.trial, R, selected, candidate)))
        self._total += batch_total
        self.n_displacement_samples += W * self.draws_per_configuration
        self.n_configurations += W

    def result(self) -> CondensateOccupationResult:
        if self.n_displacement_samples == 0:
            raise ValueError("no configurations accumulated")
        return CondensateOccupationResult(
            n0_over_n=float(self._total / self.n_displacement_samples),
            n_configurations=self.n_configurations,
            n_displacement_samples=self.n_displacement_samples,
        )
