"""Smooth finite-range periodic pair repulsion and symmetric bosonic Jastrow trial.

For R.shape == (n_walkers, n_particles, 3), all interactions use the
minimum-image convention of the supplied Bravais cell. The potential and the
trial Jastrow factor have compact radial support bounded by the cell inradius:
no interactions couple a particle to more than one periodic image.

This deliberately models a finite-range *soft-core* repulsion, NOT the
unregularized singular Yukawa potential. The polynomial cutoff is C2 so both
log-Psi gradients and Laplacians are continuous across the cutoff.
"""

from __future__ import annotations

import numpy as np

from .cells import BCCCell, OrthorhombicCell, PeriodicCell


def _check_cutoff(cell: PeriodicCell, cutoff: float) -> float:
    if not isinstance(cell, PeriodicCell):
        raise TypeError("cell must be a PeriodicCell")
    cutoff = float(cutoff)
    if not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("cutoff must be finite and positive")

    if isinstance(cell, BCCCell):
        inradius = np.sqrt(3.0) * cell.length / 4.0
    elif isinstance(cell, OrthorhombicCell):
        inradius = 0.5 * min(cell.lengths)
    else:
        # For any nonzero integer translation n, ||n @ A|| >= s_min(A).
        # Thus s_min(A)/2 is a rigorous (possibly conservative) inradius bound.
        inradius = 0.5 * np.linalg.svd(cell.vectors, compute_uv=False)[-1]
    if cutoff > inradius * (1.0 + 1e-12):
        raise ValueError(
            f"cutoff={cutoff:g} exceeds safe minimum-image radius "
            f"{inradius:g}; reduce cutoff or enlarge the cell"
        )
    return cutoff


def _batched_positions(R) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    if R.ndim != 3 or R.shape[-1] != 3 or R.shape[0] == 0 or R.shape[1] == 0:
        raise ValueError("positions must have shape (n_walkers, n_particles, 3)")
    if not np.all(np.isfinite(R)):
        raise ValueError("positions contain non-finite coordinates")
    return R


def pair_displacements(R, cell: PeriodicCell):
    """Return (i, j, delta) with delta.shape=(walkers, pairs, 3), i<j."""
    R = _batched_positions(R)
    i, j = np.triu_indices(R.shape[1], k=1)
    delta = cell.minimum_image(R[:, i, :] - R[:, j, :])
    return i, j, delta


def _shape_factor(r2: np.ndarray, cutoff: float):
    """f(r)=(1-r**2/rc**2)**3 for r<rc, zero otherwise."""
    t = np.minimum(r2 / cutoff**2, 1.0)
    q = 1.0 - t
    return q**3, q, t


class SoftCorePairPotential:
    """Repulsive bounded pair potential V(r) = height*(1-r²/rc²)^3_+.

    The model is finite-range, isotropic and C2 at the cutoff, with a maximum
    pair energy of `height` at coincidence. The Hamiltonian is the pair sum,
    counting each unordered particle pair exactly once.
    """

    def __init__(self, cell: PeriodicCell, *, height: float, cutoff: float):
        self.cell = cell
        self.cutoff = _check_cutoff(cell, cutoff)
        self.height = float(height)
        if not np.isfinite(self.height) or self.height < 0:
            raise ValueError("height must be finite and non-negative")

    def __call__(self, R):
        _, _, delta = pair_displacements(R, self.cell)
        r2 = np.sum(delta * delta, axis=-1)
        f, _, _ = _shape_factor(r2, self.cutoff)
        return self.height * np.sum(f, axis=1)


class SoftCoreJastrowTrial:
    r"""Symmetric bosonic Jastrow wavefunction Psi=exp[-a sum_(i<j) f(rij)].

    f(r)=(1-r²/rc²)^3_+; its derivatives are analytic and continuous at rc.
    This is invariant under particle permutations and simultaneous translations
    by Bravais-lattice vectors. Unlike a Nosanow trial, it does not pin atoms
    to lattice sites, making it suitable for a liquid baseline.
    """

    def __init__(self, cell: PeriodicCell, *, amplitude: float, cutoff: float):
        self.cell = cell
        self.cutoff = _check_cutoff(cell, cutoff)
        self.amplitude = float(amplitude)
        if not np.isfinite(self.amplitude) or self.amplitude < 0:
            raise ValueError("amplitude must be finite and non-negative")

    def log_psi(self, R):
        _, _, delta = pair_displacements(R, self.cell)
        r2 = np.sum(delta * delta, axis=-1)
        f, _, _ = _shape_factor(r2, self.cutoff)
        return -self.amplitude * np.sum(f, axis=1)

    def grad_log_psi(self, R):
        R = _batched_positions(R)
        i, j, delta = pair_displacements(R, self.cell)
        r2 = np.sum(delta * delta, axis=-1)
        _, q, _ = _shape_factor(r2, self.cutoff)
        # ∇_(ri) logPsi_pair = 6a/rc² * q² * (ri-rj)
        g = (6.0 * self.amplitude / self.cutoff**2) * q[..., None]**2 * delta
        result = np.zeros_like(R)
        # np.add.at handles repeated particle indices correctly.
        walkers = np.arange(len(R))[:, None]
        np.add.at(result, (walkers, i[None, :]), g)
        np.add.at(result, (walkers, j[None, :]), -g)
        return result

    def laplacian_log_psi(self, R):
        _, _, delta = pair_displacements(R, self.cell)
        r2 = np.sum(delta * delta, axis=-1)
        _, q, t = _shape_factor(r2, self.cutoff)
        # 3D Laplacian in delta, doubled for its two particle coordinates:
        # 2 * (6a/rc²) * q * (3 - 7t).
        return (12.0 * self.amplitude / self.cutoff**2) * np.sum(
            q * (3.0 - 7.0 * t), axis=1
        )

    def log_ratio_particle_move(self, R, particle_index: int, new_positions):
        """Fast per-walker log(Psi(new)/Psi(old)) for changing one particle.

        Only the N-1 pairs touching the moved particle are reevaluated.
        This is O(W*N), rather than recomputing O(W*N²) total pairs.
        """
        R = _batched_positions(R)
        index = int(particle_index)
        if index < 0 or index >= R.shape[1]:
            raise IndexError("particle_index is out of bounds")
        new = np.asarray(new_positions, dtype=np.float64)
        if new.shape != (len(R), 3) or not np.all(np.isfinite(new)):
            raise ValueError("new_positions must have shape (n_walkers, 3)")
        others = np.arange(R.shape[1]) != index
        old_dr = self.cell.minimum_image(R[:, index, None, :] - R[:, others, :])
        new_dr = self.cell.minimum_image(new[:, None, :] - R[:, others, :])
        old_f, _, _ = _shape_factor(np.sum(old_dr**2, axis=-1), self.cutoff)
        new_f, _, _ = _shape_factor(np.sum(new_dr**2, axis=-1), self.cutoff)
        return -self.amplitude * np.sum(new_f - old_f, axis=1)
