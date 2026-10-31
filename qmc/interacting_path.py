"""Interacting, distinguishably labelled continuum open paths on a 3D torus.

This is a primitive-action PIGS *diagonal-estimator* benchmark:

    P(X) ∝ Phi(X_0) Phi(X_M) prod_s K0(X_s, X_{s+1};tau)
           exp[-tau*(V(X_0)/2 + sum_{s=1}^{M-1}V(X_s) + V(X_M)/2)].

The free kernel is the EXACT periodic heat kernel (all images) from Patch 14;
the potential action is symmetric but approximate, with O(tau**2) global
operator bias for smooth potentials. The sampler does not symmetrize finite-T
path permutations; its positive symmetric boundary trial targets the bosonic
*ground state* only after adequate projection. There is NO pure off-diagonal
estimator in this module. The kinetic thermodynamic estimator can be noisy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cells import OrthorhombicCell
from .continuum_path import PeriodicFreePropagator


def _positive(value, name):
    try:
        x = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be positive and finite") from exc
    if not np.isfinite(x) or x <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return x


def _integer(value, name, minimum):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _value_potential(potential, bead):
    values = np.asarray(potential(bead[None, :, :]), dtype=float)
    if values.shape != (1,) or not np.all(np.isfinite(values)):
        raise ValueError("potential must return one finite scalar per configuration")
    return float(values[0])


def _log_trial(trial, bead):
    values = np.asarray(trial.log_psi(bead[None, :, :]), dtype=float)
    if values.shape != (1,) or not np.all(np.isfinite(values)):
        raise ValueError("trial.log_psi must return one finite scalar per configuration")
    return float(values[0])


@dataclass(frozen=True)
class InteractingPathConfig:
    time_step: float = 0.12
    n_links: int = 12
    n_steps: int = 2400
    burn_in: int = 600
    thin: int = 8
    move_sigma: float = 0.42
    refresh_probability: float = 0.12
    translate_probability: float = 0.12
    seed: int = 2026

    def validate(self):
        _positive(self.time_step, "time_step")
        _positive(self.move_sigma, "move_sigma")
        for name, minimum in (("n_links", 2), ("n_steps", 2), ("burn_in", 0), ("thin", 1)):
            _integer(getattr(self, name), name, minimum)
        if self.n_links % 2:
            raise ValueError("n_links must be even to define the midpoint")
        if self.burn_in >= self.n_steps:
            raise ValueError("burn_in must be less than n_steps")
        if (self.n_steps - self.burn_in + self.thin - 1) // self.thin < 2:
            raise ValueError("at least two production measurements required")
        for name in ("refresh_probability", "translate_probability"):
            p = float(getattr(self, name))
            if not np.isfinite(p) or p < 0 or p > 1:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.refresh_probability + self.translate_probability > 1:
            raise ValueError("refresh_probability + translate_probability must be <= 1")
        _integer(self.seed, "seed", 0)
        return self

    @property
    def beta(self):
        return self.n_links * self.time_step / 2.0


@dataclass(frozen=True)
class InteractingPathResult:
    beta: float
    midpoint_potential: np.ndarray
    endpoint_local_energy: np.ndarray
    thermodynamic_energy: np.ndarray
    acceptances: dict[str, float]
    attempted: dict[str, int]
    final_path: np.ndarray


class PrimitiveInteractingPath:
    """Metropolis open-path sampler using the symmetric primitive action.

    All beads are *wrapped Cartesian coordinates* in an orthorhombic torus.
    A single-bead displacement is a symmetric wrapped-Gaussian proposal.
    Whole-path refreshes are independent free-path proposals. Translating
    all beads of one *labelled particle* has symmetric proposal density and
    exactly cancels the free kinetic links. All moves preserve detailed balance.

    Both boundary trial amplitudes must be real and strictly positive.
    Sampling a named particle does not implement a finite-temperature Bose
    exchange cycle; the trial can nevertheless be permutation symmetric.
    """

    def __init__(self, cell: OrthorhombicCell, potential, trial, n_particles: int,
                 config: InteractingPathConfig = InteractingPathConfig(),
                 *, kinetic_coefficient: float = 0.5):
        if not isinstance(cell, OrthorhombicCell):
            raise TypeError("primitive path requires an OrthorhombicCell")
        self.cell = cell
        self.potential = potential
        self.trial = trial
        self.n_particles = _integer(n_particles, "n_particles", 1)
        self.config = config.validate()
        self.kernel = PeriodicFreePropagator(cell, kinetic_coefficient)
        for owner in (potential, trial):
            if getattr(owner, "cell", None) is not None and not np.allclose(owner.cell.vectors, cell.vectors, rtol=1e-12, atol=1e-12):
                raise ValueError("potential and trial must use the same periodic cell")
        if not hasattr(trial, "log_psi"):
            raise TypeError("trial must implement log_psi")
        self.rng = np.random.default_rng(self.config.seed)
        self.path = self._free_path()
        self._counts = {key: [0, 0] for key in ("bead", "translate", "refresh")}
        self._check_log_weight(self.path)

    def _free_path(self):
        n = self.n_particles
        start = self.cell.to_cartesian(self.rng.random((n, 3)))
        increments = np.sqrt(2 * self.kernel.kinetic_coefficient * self.config.time_step) * self.rng.normal(
            size=(self.config.n_links, n, 3)
        )
        return self.cell.wrap(start[None, :, :] + np.concatenate(
            (np.zeros((1, n, 3)), np.cumsum(increments, axis=0)), axis=0
        ))

    def _potentials(self, path):
        values = np.asarray(self.potential(path), dtype=float)
        if values.shape != (self.config.n_links + 1,) or not np.all(np.isfinite(values)):
            raise ValueError("potential must return one finite scalar per path bead")
        return values

    def _action_potential(self, path):
        v = self._potentials(path)
        return self.config.time_step * (0.5 * (v[0] + v[-1]) + np.sum(v[1:-1]))

    def _boundary(self, path):
        return _log_trial(self.trial, path[0]) + _log_trial(self.trial, path[-1])

    def log_weight(self, path):
        x = np.asarray(path, dtype=float)
        if (x.shape != (self.config.n_links + 1, self.n_particles, 3)
                or not np.all(np.isfinite(x))):
            raise ValueError("path must have finite (n_links+1, n_particles, 3) shape")
        k = self.kernel.log_kernel(x[1:] - x[:-1], self.config.time_step)
        return float(np.sum(k) + self._boundary(x) - self._action_potential(x))

    def _check_log_weight(self, path):
        weight = self.log_weight(path)
        if not np.isfinite(weight):
            raise FloatingPointError("nonfinite path weight")
        return weight

    def _accept(self, kind, proposal, log_ratio):
        self._counts[kind][1] += 1
        if not np.isfinite(log_ratio):
            raise FloatingPointError("nonfinite Metropolis log ratio")
        if np.log(self.rng.random()) < min(0.0, log_ratio):
            self.path = proposal
            self._counts[kind][0] += 1
            return True
        return False

    def move_bead(self, bead: int | None = None, particle: int | None = None):
        cfg, rng = self.config, self.rng
        j = int(rng.integers(cfg.n_links + 1)) if bead is None else int(bead)
        i = int(rng.integers(self.n_particles)) if particle is None else int(particle)
        if not 0 <= j <= cfg.n_links or not 0 <= i < self.n_particles:
            raise IndexError("bead or particle index out of range")
        x = self.path
        candidate = self.cell.wrap(x[j, i] + cfg.move_sigma * rng.normal(size=3))
        proposed = x.copy()
        proposed[j, i] = candidate
        # Only the two adjacent K0 links, one potential bead, and (for
        # endpoints) the corresponding trial amplitude can change.
        delta = -cfg.time_step * (0.5 if j in (0, cfg.n_links) else 1.0) * (
            _value_potential(self.potential, proposed[j]) - _value_potential(self.potential, x[j])
        )
        if j in (0, cfg.n_links):
            delta += _log_trial(self.trial, proposed[j]) - _log_trial(self.trial, x[j])
        for left, right in ((j - 1, j), (j, j + 1)):
            if left < 0 or right > cfg.n_links:
                continue
            delta += float(self.kernel.log_kernel(proposed[right, i] - proposed[left, i], cfg.time_step)
                           - self.kernel.log_kernel(x[right, i] - x[left, i], cfg.time_step))
        return self._accept("bead", proposed, delta)

    def move_translate(self, particle: int | None = None):
        rng = self.rng
        i = int(rng.integers(self.n_particles)) if particle is None else int(particle)
        if not 0 <= i < self.n_particles:
            raise IndexError("particle index out of range")
        proposed = self.path.copy()
        displacement = self.config.move_sigma * rng.normal(size=3)
        proposed[:, i, :] = self.cell.wrap(proposed[:, i, :] + displacement)
        # Wrapped translations leave every free link unchanged.
        delta = (self._boundary(proposed) - self._boundary(self.path)
                 - self._action_potential(proposed) + self._action_potential(self.path))
        return self._accept("translate", proposed, float(delta))

    def move_refresh(self):
        proposed = self._free_path()
        # The independent proposal is exactly the *normalized free-path*
        # reference distribution; its kinetic factors cancel in MH.
        delta = (self._boundary(proposed) - self._boundary(self.path)
                 - self._action_potential(proposed) + self._action_potential(self.path))
        return self._accept("refresh", proposed, float(delta))

    def step(self):
        u = self.rng.random()
        if u < self.config.refresh_probability:
            return self.move_refresh()
        if u < self.config.refresh_probability + self.config.translate_probability:
            return self.move_translate()
        return self.move_bead()

    def local_energy_at_endpoints(self):
        """Trial local H at path ends: an energy diagnostic, not a pure OBDM.

        For an exact projector, averaging this under the two-ended path weight
        equals the projected energy by H-commutation. With primitive G_tau it
        has finite-tau bias and must be timestep extrapolated.
        """
        if not (hasattr(self.trial, "grad_log_psi") and hasattr(self.trial, "laplacian_log_psi")):
            return float("nan")
        end = self.path[[0, -1], :, :]
        grad = np.asarray(self.trial.grad_log_psi(end), dtype=float)
        lap = np.asarray(self.trial.laplacian_log_psi(end), dtype=float)
        if grad.shape != end.shape or lap.shape != (2,):
            raise ValueError("trial derivatives have incompatible shapes")
        values = np.asarray(self.potential(end), dtype=float)
        if values.shape != (2,):
            raise ValueError("potential has incompatible endpoint shape")
        energy = values - self.kernel.kinetic_coefficient * (
            lap + np.sum(grad**2, axis=(1, 2))
        )
        return float(np.mean(energy))

    def thermodynamic_energy(self):
        """Unbiased derivative-of-Z energy for the *discretized* path action.

        The periodic image weighted squared displacement must be used, not
        simply the squared minimum image. Can have substantial variance.
        """
        tau = self.config.time_step
        dr = self.path[1:] - self.path[:-1]  # (M,N,3)
        sigma2 = 2.0 * self.kernel.kinetic_coefficient * tau
        squared_sum = np.zeros(dr.shape[:-1], dtype=float)
        for axis, L in enumerate(self.cell.lengths):
            delta = dr[..., axis] - L * np.rint(dr[..., axis] / L)
            radius = max(1, int(np.ceil(0.5 + 8 * np.sqrt(sigma2) / L)))
            dimg = delta[..., None] + L * np.arange(-radius, radius + 1)
            exp = -dimg**2 / (2 * sigma2)
            exp -= np.max(exp, axis=-1, keepdims=True)
            prob = np.exp(exp)
            squared_sum += np.sum(dimg**2 * prob, axis=-1) / np.sum(prob, axis=-1)
        M, N = self.config.n_links, self.n_particles
        kinetic = (3.0 * N / (2 * tau)
                   - float(np.sum(squared_sum)) / (4 * self.kernel.kinetic_coefficient * tau**2 * M))
        return kinetic + self._action_potential(self.path) / (M * tau)

    def run(self) -> InteractingPathResult:
        pot, e_end, e_th = [], [], []
        for step in range(self.config.n_steps):
            self.step()
            if step >= self.config.burn_in and (step - self.config.burn_in) % self.config.thin == 0:
                pot.append(_value_potential(self.potential, self.path[self.config.n_links // 2]))
                e_end.append(self.local_energy_at_endpoints())
                e_th.append(self.thermodynamic_energy())
        return InteractingPathResult(
            beta=self.config.beta,
            midpoint_potential=np.asarray(pot),
            endpoint_local_energy=np.asarray(e_end),
            thermodynamic_energy=np.asarray(e_th),
            acceptances={k: (v[0] / v[1] if v[1] else float("nan")) for k, v in self._counts.items()},
            attempted={k: v[1] for k, v in self._counts.items()},
            final_path=self.path.copy(),
        )
