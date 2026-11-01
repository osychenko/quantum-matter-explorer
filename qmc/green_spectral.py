"""Independent truncated plane-wave spectral G_k^+(t) for canonical N=1 -> 2.

The *one-particle* thermal background is ideal. The N+1=2 sector has the
periodic soft-core pair interaction and exact bosonic exchange. Single-particle
modes are truncated to [-K,K]^3; these are **Galerkin reference values**, not
exact continuum energies. Convergence in K and real-space FFT quadrature is
an independent numerical systematic error.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from math import sqrt

import numpy as np


@dataclass(frozen=True)
class SpectralGreenResult:
    times: np.ndarray
    modes: np.ndarray
    green: np.ndarray
    k_max: int
    n_grid: int
    n_one_body_modes: int
    partition_one_body: float


class TwoBosonAdditionReference:
    """Canonical finite-temperature G_k^+(t) for an ideal one-boson host.

    The interacting two-boson eigenstates are resolved in a symmetric Fock
    basis with conserved *total* momentum. This is independent of the
    continuum worm discretization and its primitive Trotter action.
    """

    def __init__(self, *, cell_length=2.8, kinetic_coefficient=0.5,
                 height=3.0, cutoff=1.2, k_max=1, n_grid=16):
        for name, val in (("cell_length", cell_length),
                          ("kinetic_coefficient", kinetic_coefficient),
                          ("cutoff", cutoff)):
            try:
                valid = np.isfinite(float(val)) and float(val) > 0
            except (ValueError, TypeError, OverflowError):
                valid = False
            if not valid:
                raise ValueError(f"{name} must be positive and finite")
        if (not np.isfinite(height)) or height < 0:
            raise ValueError("height must be finite and nonnegative")
        if cutoff > cell_length / 2:
            raise ValueError("cutoff cannot exceed half the box length")
        if (isinstance(k_max, (bool, np.bool_)) or not isinstance(k_max, (int, np.integer))
                or not 1 <= k_max <= 2):
            raise ValueError("k_max must be an integer between 1 and 2")
        if (isinstance(n_grid, (bool, np.bool_)) or not isinstance(n_grid, (int, np.integer))
                or n_grid < 4*k_max+1):
            raise ValueError("n_grid must be an integer >= 4*k_max+1")
        self.length, self.D, self.height, self.cutoff = (
            float(cell_length), float(kinetic_coefficient), float(height), float(cutoff))
        self.k_max, self.n_grid = int(k_max), int(n_grid)
        self.modes = np.asarray(list(product(range(-k_max, k_max+1), repeat=3)), dtype=int)
        self.mode_lookup = {tuple(m): i for i, m in enumerate(self.modes)}
        self.eps = self.D*(2*np.pi/self.length)**2*np.sum(self.modes**2, axis=1)
        self._build_potential()
        self._blocks = {}

    def _build_potential(self):
        n, L = self.n_grid, self.length
        coord = np.arange(n)*L/n
        x, y, z = np.meshgrid(coord, coord, coord, indexing="ij")
        rr = sum((axis - L*np.rint(axis/L))**2 for axis in (x, y, z))
        v = self.height*np.maximum(1.0-rr/self.cutoff**2, 0)**3
        self.vhat = np.fft.fftn(v)/n**3
        # For the even real pair potential the Fourier coefficients are real.
        # Tiny imaginary roundoff is retained to preserve Hermitian structure.

    @staticmethod
    def _ordered_components(a, b):
        if a == b:
            return ((a, b, 1.0),)
        c = 1/sqrt(2)
        return ((a, b, c), (b, a, c))

    def _block(self, total):
        key = tuple(int(t) for t in total)
        if key in self._blocks:
            return self._blocks[key]
        pairs = [(i, j) for i in range(len(self.modes))
                 for j in range(i, len(self.modes))
                 if tuple(self.modes[i]+self.modes[j]) == key]
        if not pairs:
            raise ValueError("total momentum has no two-particle basis states")
        idx = {pair: i for i, pair in enumerate(pairs)}
        first = np.asarray([self.modes[p] for p, q in pairs], dtype=int)
        second = np.asarray([self.modes[q] for p, q in pairs], dtype=int)
        repeated = np.asarray([p == q for p, q in pairs], dtype=float)
        # For normalized symmetric pair states |p,q>_S, with total
        # momentum conserved, V_ab = [Vhat(p_a-p_b) + Vhat(p_a-q_b)]
        # / sqrt[(1+delta_pq)_a (1+delta_pq)_b]. This vectorized form
        # avoids Python loops over the small dense momentum blocks.
        d1 = first[:, None, :] - first[None, :, :]
        d2 = first[:, None, :] - second[None, :, :]
        v1 = self.vhat[tuple(np.mod(d1[:, :, axis], self.n_grid) for axis in range(3))]
        v2 = self.vhat[tuple(np.mod(d2[:, :, axis], self.n_grid) for axis in range(3))]
        normalizer = np.sqrt((1+repeated)[:, None]*(1+repeated)[None, :])
        H = (v1+v2)/normalizer
        H[np.diag_indices_from(H)] += np.asarray(
            [self.eps[p]+self.eps[q] for p, q in pairs])
        if not np.allclose(H, H.conj().T, atol=1e-11, rtol=0):
            raise RuntimeError("spectral two-particle Hamiltonian is not Hermitian")
        energies, vectors = np.linalg.eigh(H)
        result = (pairs, idx, energies, vectors)
        self._blocks[key] = result
        return result

    def measure(self, beta, times, indices=((0, 0, 0), (1, 0, 0))) -> SpectralGreenResult:
        beta = float(beta)
        t = np.asarray(times, dtype=float)
        k = np.asarray(indices)
        if (not np.isfinite(beta) or beta <= 0 or t.ndim != 1 or len(t) == 0
                or not np.all(np.isfinite(t)) or np.any(t < 0) or np.any(t >= beta)):
            raise ValueError("require beta>0 and finite 0<=times<beta")
        if (k.ndim != 2 or k.shape[1] != 3 or not np.issubdtype(k.dtype, np.number)
                or not np.all(np.isfinite(k)) or not np.all(k==np.rint(k))):
            raise ValueError("indices must be finite integer triples")
        k = k.astype(int)
        if any(tuple(mode) not in self.mode_lookup for mode in k):
            raise ValueError("all measured modes must lie in the spectral basis")
        Z1 = float(np.sum(np.exp(-beta*self.eps)))
        result = np.zeros((len(t), len(k)), dtype=float)
        for ik, mode in enumerate(k):
            im = self.mode_lookup[tuple(mode)]
            for p in range(len(self.modes)):
                total = self.modes[p]+mode
                pairs, idx, energy, vectors = self._block(total)
                a, b = sorted((p, im))
                ix = idx[(a, b)]
                strength = 2. if p == im else 1.
                overlap = strength*np.abs(vectors[ix])**2
                added = np.exp(-t[:, None]*energy[None, :])@overlap
                result[:, ik] += np.exp(-(beta-t)*self.eps[p])*added
        result /= Z1
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("nonfinite spectral G")
        return SpectralGreenResult(t.copy(), k.copy(), result, self.k_max,
                                   self.n_grid, len(self.modes), Z1)
