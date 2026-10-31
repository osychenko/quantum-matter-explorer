"""Sign-free stochastic open-path projector for a small Bose-Hubbard ring.

Paths x_0,...,x_(2m) have weight
    phi[x_0] prod_t T[x_t,x_(t+1)] phi[x_(2m)],
where T=exp(-dt*(H-E_ref)) is a nonnegative, symmetric *exact finite-basis*
transfer matrix. Reptation Metropolis moves and single-bead Gibbs updates
sample this positive path weight. No ground-state vector is used in sampling.

The pure midpoint OBDM uses a conditional bridge (Rao-Blackwell) estimator:
    (T O_ij T)[x_(m-1),x_(m+1)] / (T^2)[x_(m-1),x_(m+1)].
It is unbiased under the stationary path measure for the symmetrically
projected finite-time matrix element <phi|T^m O T^m|phi>/<phi|T^(2m)|phi>.

This is a dense, small-system 1D LATTICE benchmark. The kernel is constructed
by exact diagonalization, NOT via a continuum short-time approximation.
This is not yet a production continuum open-path/worm Monte Carlo method.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np

from .pure_projector import BoseHubbardRing, ExactBosonProjector


@dataclass(frozen=True)
class PathConfig:
    dt: float = 0.25
    half_steps: int = 5
    burn_in: int = 800
    n_samples: int = 1000
    sample_stride: int = 4
    gibbs_updates: int = 3
    seed: int = 2026

    def __post_init__(self):
        if not np.isfinite(self.dt) or self.dt <= 0:
            raise ValueError("dt must be finite and positive")
        for name, minimum in (("half_steps", 1), ("burn_in", 0),
                              ("n_samples", 1), ("sample_stride", 1),
                              ("gibbs_updates", 0), ("seed", 0)):
            val = getattr(self, name)
            if isinstance(val, (bool, np.bool_)) or not isinstance(val, (int, np.integer)) or val < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")

    @property
    def beta(self):
        """Imaginary time on EACH side of the midpoint operator."""
        return self.half_steps * self.dt


@dataclass(frozen=True)
class PathResult:
    beta: float
    obdm: np.ndarray
    g1: np.ndarray
    n0_fraction: float
    energy: float
    reptation_acceptance: float
    n_samples: int
    samples: np.ndarray  # sampled midpoint-bridge configurations (left,right)


class OpenPathProjector:
    """Finite-basis reptation + Gibbs path sampler with pure midpoint operator.

    ``phi`` is a strictly positive normalized trial vector. The strict
    condition gives irreducible positive path support when hopping > 0.
    Results from successive paths are autocorrelated; use independent seeds
    and/or block analysis before assigning uncertainties.
    """

    def __init__(self, model: BoseHubbardRing, config: PathConfig,
                 phi=None):
        if not isinstance(model, BoseHubbardRing):
            raise TypeError("model must be a BoseHubbardRing")
        if not isinstance(config, PathConfig):
            raise TypeError("config must be a PathConfig")
        if model.dimension > 120:
            raise ValueError("dense open-path benchmark requires Hilbert dimension <= 120")
        if model.hopping <= 0:
            raise ValueError("reptation sampler needs hopping > 0 for irreducibility")
        self.model = model
        self.config = config
        self.exact = ExactBosonProjector(model)
        dim = model.dimension
        phi = self.exact.trial("coherent") if phi is None else np.asarray(phi, dtype=float)
        if phi.shape != (dim,) or not np.all(np.isfinite(phi)) or np.any(phi <= 0):
            raise ValueError("phi must have strictly positive finite occupation amplitudes")
        self.phi = phi / np.linalg.norm(phi)
        self.log_phi = np.log(self.phi)
        # Shift spectrum by E0 strictly to prevent avoidable numerical overflow;
        # this constant cancels in path normalization and in bridge estimators.
        energies, eigenvectors = self.exact.energies, self.exact.eigenvectors
        self.T = (eigenvectors * np.exp(-config.dt * (energies - energies[0]))) @ eigenvectors.T
        self.T = (self.T + self.T.T) / 2.0
        if np.min(self.T) < -1e-12:
            raise ArithmeticError("transfer matrix is not nonnegative")
        self.T = np.maximum(self.T, 0.0)
        self.rowsums = self.T.sum(axis=1)
        if not np.all(self.rowsums > 0):
            raise ArithmeticError("transfer matrix contains disconnected rows")
        self.Q = self.T / self.rowsums[:, None]
        self.rng = np.random.default_rng(config.seed)
        # Initialize a path entirely in one Fock configuration; subsequent
        # Gibbs/reptation updates mix it into the exact path measure.
        initial = int(self.rng.choice(dim, p=self.phi**2))
        self.path = np.full(2 * config.half_steps + 1, initial, dtype=np.int64)
        self.attempted = 0
        self.accepted = 0
        self._bridge_denominator = self.T @ self.T
        self._operator_bridges = self._build_operator_bridges()
        self._energy_bridge = self.T @ self.exact.hamiltonian @ self.T

    def _build_operator_bridges(self):
        """T a_i^dagger a_j T for all ordered site pairs.

        This small dense bridge contraction integrates out the two states at
        the insertion. It avoids sampling a sign-changing offdiagonal worm
        sector and is a validated small-basis conditional estimator.
        """
        L = self.model.n_sites
        basis = self.exact.basis
        result = np.empty((L, L, self.model.dimension, self.model.dimension))
        for i in range(L):
            for j in range(L):
                op = np.zeros_like(self.T)
                for b, occupation in enumerate(basis):
                    if i == j:
                        op[b, b] = occupation[i]
                    elif occupation[j] > 0:
                        new = occupation.copy()
                        new[j] -= 1
                        new[i] += 1
                        a = self.exact._index[tuple(new)]
                        op[a, b] = sqrt(float(occupation[j] * (occupation[i] + 1)))
                result[i, j] = self.T @ op @ self.T
        return result

    def _choose(self, weights):
        probs = np.asarray(weights, dtype=float)
        total = float(probs.sum())
        if not np.isfinite(total) or total <= 0:
            raise FloatingPointError("invalid path heat-bath distribution")
        return int(self.rng.choice(len(probs), p=probs / total))

    def reptate(self):
        """Metropolis-Hastings shift: remove tail, extend opposite head."""
        old = self.path
        self.attempted += 1
        if self.rng.random() < 0.5:  # remove left endpoint, grow right
            x0, x1, xk = map(int, (old[0], old[1], old[-1]))
            z = self._choose(self.Q[xk])
            new = np.r_[old[1:], z]
            # W(new)/W(old) times reverse/forward proposal probabilities;
            # T is symmetric, so the link factors cancel exactly.
            log_alpha = (self.log_phi[x1] + self.log_phi[z]
                         - self.log_phi[x0] - self.log_phi[xk]
                         + np.log(self.rowsums[xk]) - np.log(self.rowsums[x1]))
        else:  # grow left, remove right
            x0, xkm1, xk = map(int, (old[0], old[-2], old[-1]))
            z = self._choose(self.Q[x0])
            new = np.r_[z, old[:-1]]
            log_alpha = (self.log_phi[z] + self.log_phi[xkm1]
                         - self.log_phi[x0] - self.log_phi[xk]
                         + np.log(self.rowsums[x0]) - np.log(self.rowsums[xkm1]))
        if np.log(self.rng.random()) < min(0.0, log_alpha):
            self.path = new
            self.accepted += 1
            return True
        return False

    def gibbs_update(self, position=None):
        """Exact single-site conditional update, including path endpoints."""
        p = int(self.rng.integers(len(self.path))) if position is None else int(position)
        if not 0 <= p < len(self.path):
            raise IndexError("bead position out of range")
        if p == 0:
            weights = self.phi * self.T[:, self.path[1]]
        elif p == len(self.path) - 1:
            weights = self.T[self.path[-2], :] * self.phi
        else:
            weights = self.T[self.path[p - 1], :] * self.T[:, self.path[p + 1]]
        self.path[p] = self._choose(weights)

    def step(self):
        self.reptate()
        for _ in range(self.config.gibbs_updates):
            self.gibbs_update()

    def midpoint_obdm(self):
        """Conditional pure OBDM for the current sampled path midpoint."""
        m = self.config.half_steps
        l, r = int(self.path[m - 1]), int(self.path[m + 1])
        denom = float(self._bridge_denominator[l, r])
        if denom <= 0:
            raise FloatingPointError("impossible midpoint bridge")
        return self._operator_bridges[:, :, l, r] / denom

    def midpoint_energy(self):
        m = self.config.half_steps
        l, r = int(self.path[m - 1]), int(self.path[m + 1])
        return float(self._energy_bridge[l, r] / self._bridge_denominator[l, r])

    def run(self) -> PathResult:
        for _ in range(self.config.burn_in):
            self.step()
        rho_sum = np.zeros((self.model.n_sites, self.model.n_sites))
        energy_sum = 0.0
        bridges = np.empty((self.config.n_samples, 2), dtype=np.int64)
        for n in range(self.config.n_samples):
            for _ in range(self.config.sample_stride):
                self.step()
            m = self.config.half_steps
            bridges[n] = (self.path[m - 1], self.path[m + 1])
            rho_sum += self.midpoint_obdm()
            energy_sum += self.midpoint_energy()
        rho = rho_sum / self.config.n_samples
        rho = (rho + rho.T) / 2.0  # expected symmetric; finite-MC noise
        n_sites, N = self.model.n_sites, self.model.n_bosons
        g1 = np.asarray([
            sum(rho[i, (i + delta) % n_sites] for i in range(n_sites)) / N
            for delta in range(n_sites)
        ])
        return PathResult(
            beta=self.config.beta, obdm=rho, g1=g1,
            n0_fraction=float(rho.sum() / (n_sites * N)),
            energy=energy_sum / self.config.n_samples,
            reptation_acceptance=self.accepted / self.attempted,
            n_samples=self.config.n_samples,
            samples=bridges,
        )
