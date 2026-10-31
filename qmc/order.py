"""Exchange-symmetric, translationally invariant density-modulated bosonic trial.

    log Psi(R) = log Psi_J(R) + lambda/(2N) * sum_{k in K} |rho_k|^2
    rho_k(R) = sum_i exp(i * k dot r_i)

K contains a single representative from each +/- reciprocal pair. This does
not label particles or pin the center of mass. A density-wave term can favor
clusters or multiple density patterns; it is NOT itself proof of crystallinity.
"""

from __future__ import annotations

import numpy as np

from .bosons import SoftCoreJastrowTrial, _batched_positions
from .cells import PeriodicCell


def canonical_modes(indices) -> np.ndarray:
    """Validate and canonicalize nonzero reciprocal modes, removing +/- duplicates.

    Supplying both k and -k would double count one |rho_k|^2 term. We reject
    duplicates explicitly rather than changing the effective lambda silently.
    """
    idx = np.asarray(indices)
    if idx.ndim != 2 or idx.shape[1] != 3 or len(idx) == 0:
        raise ValueError("indices must have shape (n_modes, 3), n_modes >= 1")
    if not np.issubdtype(idx.dtype, np.number):
        raise ValueError("reciprocal mode indices must be numeric integers")
    if not np.all(np.isfinite(idx)) or not np.all(idx == np.rint(idx)):
        raise ValueError("reciprocal mode indices must be finite integers")
    idx = idx.astype(np.int64)
    if np.any(np.all(idx == 0, axis=1)):
        raise ValueError("zero reciprocal mode is not allowed")
    if len({tuple(int(x) for x in row) for row in idx}) != len(idx):
        raise ValueError("duplicate reciprocal modes are not allowed")
    seen = {tuple(int(x) for x in row) for row in idx}
    if any(tuple(-int(x) for x in row) in seen for row in idx):
        raise ValueError("choose only one of each +/- reciprocal mode pair")
    return idx


class DensityWaveJastrowTrial:
    """Pair-Jastrow trial multiplied by a bosonic collective density-wave term.

    Positive ``order_amplitude`` raises probability at large |rho_k|.
    ``indices`` are reciprocal lattice indices, e.g. [(2,0,0),(0,2,0),(0,0,2)]
    for a two-period modulation in each direction of a cubic box.

    The trial is invariant under arbitrary particle permutation and uniform
    translation, including periodic boundary translations of single particles.
    No particular physical crystalline phase is asserted by this ansatz.
    """

    def __init__(
        self, cell: PeriodicCell, *, jastrow_amplitude: float,
        order_amplitude: float, cutoff: float, indices,
    ):
        if not isinstance(cell, PeriodicCell):
            raise TypeError("cell must be a PeriodicCell")
        strength = float(order_amplitude)
        if not np.isfinite(strength) or strength < 0:
            raise ValueError("order_amplitude must be finite and nonnegative")
        self.cell = cell
        self.order_amplitude = strength
        self.indices = canonical_modes(indices)
        self.wavevectors = cell.wavevectors(self.indices)
        self.pair_trial = SoftCoreJastrowTrial(
            cell, amplitude=jastrow_amplitude, cutoff=cutoff,
        )
        self.cutoff = self.pair_trial.cutoff
        self.jastrow_amplitude = self.pair_trial.amplitude

    def _phase_data(self, positions):
        R = _batched_positions(positions)
        angles = np.einsum("wnd,kd->wnk", R, self.wavevectors, optimize=True)
        phase = np.exp(1j * angles)
        return R, phase, phase.sum(axis=1)

    def log_psi(self, positions):
        R, _, rho = self._phase_data(positions)
        n = R.shape[1]
        return (
            self.pair_trial.log_psi(R)
            + (self.order_amplitude / (2.0 * n)) * np.sum(np.abs(rho) ** 2, axis=1)
        )

    def grad_log_psi(self, positions):
        R, phase, rho = self._phase_data(positions)
        n = R.shape[1]
        # derivative |rho|^2 = -2 k Im(exp(i k.r_i) conj(rho))
        z = phase * np.conj(rho)[:, None, :]
        collective_grad = -(self.order_amplitude / n) * np.einsum(
            "wnk,kd->wnd", z.imag, self.wavevectors, optimize=True,
        )
        return self.pair_trial.grad_log_psi(R) + collective_grad

    def laplacian_log_psi(self, positions):
        R, _, rho = self._phase_data(positions)
        n = R.shape[1]
        k_squared = np.sum(self.wavevectors ** 2, axis=1)
        # Full configuration Laplacian |rho|^2 = 2 k^2 (N - |rho|^2).
        collective_lap = (self.order_amplitude / n) * np.sum(
            k_squared[None, :] * (n - np.abs(rho) ** 2), axis=1
        )
        return self.pair_trial.laplacian_log_psi(R) + collective_lap

    def log_ratio_particle_move(self, positions, particle_index: int, new_positions):
        R, phases, rho = self._phase_data(positions)
        n = R.shape[1]
        idx = int(particle_index)
        if idx < 0 or idx >= n:
            raise IndexError("particle_index is out of range")
        new = np.asarray(new_positions, dtype=float)
        if new.shape != (len(R), 3) or not np.all(np.isfinite(new)):
            raise ValueError("new_positions must be finite with shape (n_walkers, 3)")
        updated = np.exp(1j * (new @ self.wavevectors.T))
        rho_new = rho + updated - phases[:, idx, :]
        delta_collective = (self.order_amplitude / (2.0 * n)) * np.sum(
            np.abs(rho_new) ** 2 - np.abs(rho) ** 2, axis=1
        )
        return self.pair_trial.log_ratio_particle_move(R, idx, new) + delta_collective
