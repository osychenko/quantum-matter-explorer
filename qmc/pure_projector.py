"""Exact, sign-problem-free finite-basis *benchmark* for pure bosonic OBDM.

This is a one-dimensional, finite-site Bose-Hubbard ring, NOT the project's
three-dimensional continuum soft-core model and NOT a Monte Carlo algorithm.
It provides rigorous small-Hilbert-space ground-state reference data against
which a future open-path/reptation QMC OBDM estimator can be tested.

Symmetric imaginary-time projection is evaluated exactly:
    |psi_beta> = exp[-beta (H - E0)] |trial> / norm
    rho_ij(beta) = <psi_beta| a_i^dagger a_j |psi_beta>
The beta->infinity result is a *pure ground-state expectation* if the trial
has nonzero ground-state overlap and the ground state is nondegenerate.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import comb, factorial, prod, sqrt

import numpy as np


def _occupations(n_sites: int, n_bosons: int) -> np.ndarray:
    """All weak compositions of N into L sites, in deterministic order."""
    # Stars and bars: the separators choose positions among N+L-1 slots.
    states = []
    for bars in combinations(range(n_bosons + n_sites - 1), n_sites - 1):
        points = (-1,) + bars + (n_bosons + n_sites - 1,)
        states.append(tuple(points[i + 1] - points[i] - 1 for i in range(n_sites)))
    return np.asarray(states, dtype=np.int64)


@dataclass(frozen=True)
class BoseHubbardRing:
    """Periodic 1D Bose-Hubbard model with N identical, spinless bosons.

    H = -t sum_i (a^dagger_(i+1) a_i + h.c.)
        + U/2 sum_i n_i (n_i-1) + V sum_i n_i n_(i+1).

    Units are lattice spacing = 1 and hbar = 1. Require at least three sites
    so each nearest-neighbor undirected ring bond is counted once.
    """

    n_sites: int = 5
    n_bosons: int = 4
    hopping: float = 1.0
    onsite_u: float = 2.0
    nearest_v: float = 0.0
    max_dimension: int = 600

    def __post_init__(self):
        for key, limit in (("n_sites", 3), ("n_bosons", 1), ("max_dimension", 1)):
            value = getattr(self, key)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < limit:
                raise ValueError(f"{key} must be an integer >= {limit}")
        for key in ("hopping", "onsite_u", "nearest_v"):
            value = getattr(self, key)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if self.dimension > self.max_dimension:
            raise ValueError(
                f"Hilbert-space dimension {self.dimension} exceeds max_dimension="
                f"{self.max_dimension}; this is a small-system dense benchmark"
            )

    @property
    def dimension(self) -> int:
        return comb(self.n_bosons + self.n_sites - 1, self.n_sites - 1)


@dataclass(frozen=True)
class PureOBDMResult:
    beta: float
    energy: float
    obdm: np.ndarray
    g1: np.ndarray
    n0_fraction: float
    occupations: np.ndarray
    condensate_eigenvalue_fraction: float
    gap: float


class ExactBosonProjector:
    """Dense diagonalization and symmetric imaginary-time ground-state projection.

    ``beta`` is the projection time applied to EACH side of the OBDM, i.e.
    the full path length for a symmetric estimator would be 2*beta.
    ``beta=np.inf`` selects the exact lowest eigenvector directly.
    """

    def __init__(self, model: BoseHubbardRing):
        if not isinstance(model, BoseHubbardRing):
            raise TypeError("model must be a BoseHubbardRing")
        self.model = model
        self.basis = _occupations(model.n_sites, model.n_bosons)
        self._index = {tuple(state): i for i, state in enumerate(self.basis)}
        self.hamiltonian = self._build_hamiltonian()
        self.energies, self.eigenvectors = np.linalg.eigh(self.hamiltonian)
        self.gap = float(self.energies[1] - self.energies[0]) if len(self.energies) > 1 else float("inf")

    def _build_hamiltonian(self) -> np.ndarray:
        model, basis = self.model, self.basis
        H = np.zeros((len(basis), len(basis)), dtype=np.float64)
        L = model.n_sites
        for col, occ in enumerate(basis):
            H[col, col] += (
                0.5 * model.onsite_u * float(np.sum(occ * (occ - 1)))
                + model.nearest_v * float(np.sum(occ * np.roll(occ, -1)))
            )
            for site in range(L):
                neighbor = (site + 1) % L
                for origin, destination in ((site, neighbor), (neighbor, site)):
                    if occ[origin] == 0:
                        continue
                    shifted = occ.copy()
                    shifted[origin] -= 1
                    shifted[destination] += 1
                    row = self._index[tuple(shifted)]
                    H[row, col] -= model.hopping * sqrt(
                        float(occ[origin] * (occ[destination] + 1))
                    )
        if not np.allclose(H, H.T, atol=1e-12, rtol=0):
            raise RuntimeError("Bose-Hubbard Hamiltonian must be Hermitian")
        return H

    def trial(self, kind: str = "coherent", *, penalty: float = 1.0) -> np.ndarray:
        """Normalized positive seed: coherent, Gutzwiller, uniform, or localized.

        'localized' places every boson on site zero; it has nonzero overlap
        with the unique positive ground state for t>0, but may converge slowly.
        """
        if kind not in ("coherent", "gutzwiller", "uniform", "localized"):
            raise ValueError("unknown trial kind")
        if not np.isfinite(penalty) or penalty < 0:
            raise ValueError("penalty must be finite and nonnegative")
        if kind == "localized":
            seed = np.zeros(len(self.basis))
            seed[self._index[(self.model.n_bosons,) + (0,) * (self.model.n_sites - 1)]] = 1.0
            return seed
        if kind == "uniform":
            return np.full(len(self.basis), 1 / sqrt(len(self.basis)))
        N, L = self.model.n_bosons, self.model.n_sites
        seed = np.asarray([
            sqrt(factorial(N) / (L**N * prod(factorial(int(x)) for x in occ)))
            for occ in self.basis
        ], dtype=float)
        if kind == "gutzwiller":
            seed *= np.exp(-0.5 * penalty * np.sum(self.basis * (self.basis - 1), axis=1))
        return seed / np.linalg.norm(seed)

    def projected_state(self, beta: float, trial: np.ndarray | None = None) -> np.ndarray:
        if np.isnan(beta) or beta < 0:
            raise ValueError("beta must be nonnegative or infinity")
        initial = self.trial("coherent") if trial is None else np.asarray(trial, dtype=float)
        if initial.shape != (len(self.basis),) or not np.all(np.isfinite(initial)):
            raise ValueError("trial must be a finite 1D vector in the occupation basis")
        norm = np.linalg.norm(initial)
        if norm <= 0:
            raise ValueError("trial vector must be nonzero")
        coefficients = self.eigenvectors.T @ (initial / norm)
        if np.isinf(beta):
            if abs(coefficients[0]) < 1e-12:
                raise ValueError("trial has zero overlap with the ground state")
            if self.gap < 1e-10:
                raise ValueError("ground state is degenerate; infinite-time state is not unique")
            return self.eigenvectors[:, 0].copy()
        amplitudes = self.eigenvectors @ (
            np.exp(-float(beta) * (self.energies - self.energies[0])) * coefficients
        )
        amp_norm = np.linalg.norm(amplitudes)
        if amp_norm <= 1e-14:
            raise ValueError("projection has negligible ground-state overlap or norm")
        return amplitudes / amp_norm

    def obdm(self, state: np.ndarray) -> np.ndarray:
        """rho[i,j] = <a_i^dagger a_j> with trace equal to total N."""
        wave = np.asarray(state, dtype=np.complex128)
        dim, L = len(self.basis), self.model.n_sites
        if wave.shape != (dim,) or not np.all(np.isfinite(wave)):
            raise ValueError("state must be a finite occupation-basis vector")
        norm2 = float(np.vdot(wave, wave).real)
        if norm2 <= 0:
            raise ValueError("state must be nonzero")
        wave = wave / sqrt(norm2)
        matrix = np.zeros((L, L), dtype=np.complex128)
        for col, occ in enumerate(self.basis):
            for j in range(L):
                count_j = int(occ[j])
                if count_j == 0:
                    continue
                matrix[j, j] += np.conj(wave[col]) * wave[col] * count_j
                for i in range(L):
                    if i == j:
                        continue
                    shifted = occ.copy()
                    shifted[j] -= 1
                    shifted[i] += 1
                    row = self._index[tuple(shifted)]
                    matrix[i, j] += (
                        np.conj(wave[row]) * wave[col]
                        * sqrt(count_j * (int(occ[i]) + 1))
                    )
        return np.real_if_close(matrix, tol=1000)

    def measure(self, beta: float, trial: np.ndarray | None = None) -> PureOBDMResult:
        state = self.projected_state(beta, trial)
        L, N = self.model.n_sites, self.model.n_bosons
        rho = np.asarray(self.obdm(state))
        g1 = np.asarray([
            float(np.real(sum(rho[i, (i + shift) % L] for i in range(L)))) / N
            for shift in range(L)
        ])
        # Occupation of the translation-invariant k=0 single-particle orbital.
        occupation = float(np.real(np.sum(rho))) / (L * N)
        eigs = np.linalg.eigvalsh(rho)
        energy = float(np.real(np.vdot(state, self.hamiltonian @ state)))
        return PureOBDMResult(
            beta=float(beta), energy=energy, obdm=rho, g1=g1,
            n0_fraction=occupation, occupations=np.asarray(eigs[::-1].real),
            condensate_eigenvalue_fraction=float(eigs[-1].real) / N,
            gap=self.gap,
        )
