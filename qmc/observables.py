"""Production-sampled, diagonal observables of periodic many-particle systems.

Both estimators consume *batches of configurations* with shape (W, N, 3).
They do not depend on the VMC/DMC algorithm, and accumulate across calls.
DMC production walkers sample a mixed distribution, so resulting diagonal
observables are mixed estimates, not generally unbiased pure estimates.

Neither class estimates statistical error bars: walker/sweep correlations make
independence-based Poisson or standard-error formulas inappropriate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bosons import pair_displacements
from .cells import BCCCell, OrthorhombicCell, PeriodicCell


def _check_particles(n_particles: int) -> int:
    if not isinstance(n_particles, (int, np.integer)) or n_particles < 1:
        raise ValueError("n_particles must be a positive integer")
    return int(n_particles)


def _positions(positions, n_particles: int) -> np.ndarray:
    values = np.asarray(positions, dtype=np.float64)
    if (values.ndim != 3 or values.shape[0] == 0 or
            values.shape[1:] != (n_particles, 3) or
            not np.all(np.isfinite(values))):
        raise ValueError(f"positions must be a finite nonempty (W, {n_particles}, 3) array")
    return values


def safe_radial_limit(cell: PeriodicCell) -> float:
    """Radius of a sphere known to fit in the cell's Wigner--Seitz region.

    Exact for orthorhombic and BCC cells; a conservative singular-value bound
    for a general oblique cell. The spherical-shell g(r) normalization is only
    valid for complete shells, i.e. radii not exceeding this limit.
    """
    if not isinstance(cell, PeriodicCell):
        raise TypeError("cell must be a PeriodicCell")
    if isinstance(cell, BCCCell):
        return np.sqrt(3.0) * cell.length / 4.0
    if isinstance(cell, OrthorhombicCell):
        return 0.5 * min(cell.lengths)
    return 0.5 * float(np.linalg.svd(cell.vectors, compute_uv=False)[-1])


@dataclass(frozen=True)
class PairCorrelationResult:
    radius: np.ndarray
    g_r: np.ndarray
    edges: np.ndarray
    pair_counts: np.ndarray
    n_configurations: int
    n_particles: int


class PairCorrelation:
    """Histogram unique particle pairs and normalize to ideal-gas g(r)=1.

    Normalization per configuration uses N(N-1)/2 distinct pairs, not N²/2;
    this gives exactly the ideal-gas expectation for finite N and independent,
    uniformly distributed particles. The bin volume is a full 3D shell.
    """

    def __init__(
        self, cell: PeriodicCell, n_particles: int, *,
        n_bins: int = 40, r_max: float | None = None,
    ):
        self.cell = cell
        self.n_particles = _check_particles(n_particles)
        if self.n_particles < 2:
            raise ValueError("pair correlation requires at least two particles")
        if not isinstance(n_bins, (int, np.integer)) or n_bins < 1:
            raise ValueError("n_bins must be a positive integer")
        limit = safe_radial_limit(cell)
        radius = limit if r_max is None else float(r_max)
        if not np.isfinite(radius) or radius <= 0 or radius > limit * (1 + 1e-12):
            raise ValueError(f"r_max must be positive and <= safe radial limit {limit:g}")
        self.edges = np.linspace(0.0, radius, int(n_bins) + 1)
        self._counts = np.zeros(int(n_bins), dtype=np.int64)
        self.n_configurations = 0

    def accumulate(self, positions) -> None:
        R = _positions(positions, self.n_particles)
        _, _, dr = pair_displacements(R, self.cell)
        distances = np.linalg.norm(dr, axis=-1)
        counts = np.histogram(distances, bins=self.edges)[0]
        self._counts += counts
        self.n_configurations += len(R)

    def result(self) -> PairCorrelationResult:
        if self.n_configurations == 0:
            raise ValueError("no configurations accumulated")
        shell_volumes = (4.0 * np.pi / 3.0) * (
            self.edges[1:] ** 3 - self.edges[:-1] ** 3
        )
        n_pairs = self.n_particles * (self.n_particles - 1) / 2.0
        expected = self.n_configurations * n_pairs * shell_volumes / self.cell.volume
        return PairCorrelationResult(
            radius=(self.edges[1:] + self.edges[:-1]) / 2.0,
            g_r=self._counts / expected,
            edges=self.edges.copy(),
            pair_counts=self._counts.copy(),
            n_configurations=self.n_configurations,
            n_particles=self.n_particles,
        )


def reciprocal_index_grid(max_index: int = 1) -> np.ndarray:
    """Nonzero integer reciprocal indices in [-max_index,+max_index]^3."""
    if not isinstance(max_index, (int, np.integer)) or max_index < 1:
        raise ValueError("max_index must be a positive integer")
    a = np.arange(-int(max_index), int(max_index) + 1)
    idx = np.stack(np.meshgrid(a, a, a, indexing="ij"), axis=-1).reshape(-1, 3)
    return idx[np.any(idx != 0, axis=1)]


@dataclass(frozen=True)
class StructureFactorResult:
    indices: np.ndarray
    wavevectors: np.ndarray
    k_magnitude: np.ndarray
    s_k: np.ndarray
    n_configurations: int
    n_particles: int


class StaticStructureFactor:
    r"""S(k) = <|sum_j exp(i k.r_j)|²>/N, for nonzero reciprocal k.

    Keeps directional k modes separate: crystalline symmetry-breaking can be
    hidden by radial averaging. The k=0 mode is excluded (S(0)=N here).
    """

    def __init__(self, cell: PeriodicCell, n_particles: int, indices):
        self.cell = cell
        self.n_particles = _check_particles(n_particles)
        idx = np.asarray(indices)
        if idx.ndim != 2 or idx.shape[1] != 3 or len(idx) == 0:
            raise ValueError("indices must have shape (n_modes, 3)")
        if not np.issubdtype(idx.dtype, np.number):
            raise ValueError("reciprocal indices must be integers")
        if not np.all(np.isfinite(idx)) or not np.all(idx == np.rint(idx)):
            raise ValueError("reciprocal indices must be finite integers")
        if np.any(np.all(idx == 0, axis=1)):
            raise ValueError("k=0 is excluded: S(0)=N trivially")
        idx = np.asarray(idx, dtype=np.int64)
        if len(np.unique(idx, axis=0)) != len(idx):
            raise ValueError("reciprocal indices must be unique")
        self.indices = idx.copy()
        self.wavevectors = self.cell.wavevectors(idx)
        self._total = np.zeros(len(idx), dtype=np.float64)
        self.n_configurations = 0

    def accumulate(self, positions) -> None:
        R = _positions(positions, self.n_particles)
        # Translation-invariant at allowed k: each particle may be wrapped or
        # shifted by arbitrary integer cell vectors without changing S(k).
        phases = np.einsum("wnd,kd->wnk", R, self.wavevectors, optimize=True)
        density_modes = np.exp(1j * phases).sum(axis=1)
        self._total += (np.abs(density_modes) ** 2).sum(axis=0) / self.n_particles
        self.n_configurations += len(R)

    def result(self) -> StructureFactorResult:
        if self.n_configurations == 0:
            raise ValueError("no configurations accumulated")
        return StructureFactorResult(
            indices=self.indices.copy(),
            wavevectors=self.wavevectors.copy(),
            k_magnitude=np.linalg.norm(self.wavevectors, axis=1),
            s_k=self._total / self.n_configurations,
            n_configurations=self.n_configurations,
            n_particles=self.n_particles,
        )
