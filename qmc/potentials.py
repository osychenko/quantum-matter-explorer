from __future__ import annotations

import numpy as np

Array = np.ndarray


def _radius(r: Array, r_eps: float = 1e-12) -> Array:
    radius = np.linalg.norm(r, axis=1)
    return np.maximum(radius, r_eps)


def yukawa_potential(
    r: Array,
    *,
    g: float = 1.0,
    mu: float = 0.5,
    r_eps: float = 1e-12,
) -> Array:
    """Attractive Yukawa potential V(r) = -g exp(-mu r) / r, in atomic units."""
    radius = _radius(r, r_eps)
    return -g * np.exp(-mu * radius) / radius


def harmonic_oscillator(r: Array, *, omega: float = 1.0) -> Array:
    """3D isotropic harmonic oscillator V = 1/2 omega^2 r^2."""
    return 0.5 * omega**2 * np.sum(r * r, axis=1)


def anisotropic_harmonic_oscillator(
    r: Array,
    *,
    omega_x: float = 1.0,
    omega_y: float = 1.5,
    omega_z: float = 2.0,
) -> Array:
    x, y, z = r[:, 0], r[:, 1], r[:, 2]
    return 0.5 * (
        omega_x**2 * x**2
        + omega_y**2 * y**2
        + omega_z**2 * z**2
    )


def double_well_x(
    r: Array,
    *,
    a: float = 1.0,
    b: float = 1.0,
    omega_y: float = 1.0,
    omega_z: float = 1.0,
) -> Array:
    x, y, z = r[:, 0], r[:, 1], r[:, 2]
    return (
        a * (x**2 - b**2) ** 2
        + 0.5 * omega_y**2 * y**2
        + 0.5 * omega_z**2 * z**2
    )


def gaussian_well(r: Array, *, depth: float = 5.0, sigma: float = 1.0) -> Array:
    r2 = np.sum(r * r, axis=1)
    return -depth * np.exp(-r2 / (2.0 * sigma**2))
