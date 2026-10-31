"""Shared shape-aware configuration operations for VMC and DMC.

Axis 0 indexes independent walkers. Every subsequent axis is a coordinate
within one complete configuration, e.g. (n_particles, spatial_dimension).
"""

from __future__ import annotations

import numpy as np


def validated_shape(shape) -> tuple[int, ...]:
    if isinstance(shape, (str, bytes)):
        raise ValueError("configuration_shape must be a nonempty tuple of positive integers")
    try:
        shape = tuple(shape)
    except TypeError as exc:
        raise ValueError("configuration_shape must be iterable") from exc
    if not shape or any(
        not isinstance(n, (int, np.integer)) or isinstance(n, (bool, np.bool_)) or n <= 0
        for n in shape
    ):
        raise ValueError("configuration_shape must contain positive integer dimensions")
    return shape


def require_positive_kinetic(value: float) -> float:
    """Validate D>0 for H=-D*Laplacian+V and the associated diffusion."""
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("kinetic_coefficient must be finite and > 0")
    return value


def configuration_norm2(arr: np.ndarray) -> np.ndarray:
    """Squared Euclidean norm of each walker's full configuration."""
    arr = np.asarray(arr)
    if arr.ndim < 2:
        raise ValueError("Expected shape (n_walkers, *configuration_shape)")
    return np.sum(arr * arr, axis=tuple(range(1, arr.ndim)))


def walker_broadcast_mask(mask: np.ndarray, ndim: int) -> np.ndarray:
    """Reshape a walker-wise boolean mask to broadcast over all coordinates."""
    if ndim < 2:
        raise ValueError("ndim must be at least 2")
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 1:
        raise ValueError("mask must be one-dimensional")
    return mask.reshape((len(mask),) + (1,) * (ndim - 1))


def initial_configuration_walkers(
    *, n_walkers: int, configuration_shape, initial_sigma: float, rng,
    initial_walkers=None,
) -> np.ndarray:
    """Initialize or validate batched configuration positions.

    Explicit initial_walkers may contain a different walker count than
    n_walkers, preserving the behavior of the earlier single-body drivers.
    """
    shape = validated_shape(configuration_shape)
    if initial_walkers is None:
        if n_walkers <= 0 or not isinstance(n_walkers, (int, np.integer)):
            raise ValueError("n_walkers must be a positive integer")
        if not np.isfinite(initial_sigma) or initial_sigma <= 0:
            raise ValueError("initial_sigma must be finite and > 0")
        return rng.normal(0.0, initial_sigma, size=(n_walkers, *shape))
    walkers = np.asarray(initial_walkers, dtype=np.float64).copy()
    if walkers.shape[1:] != shape or walkers.ndim != len(shape) + 1 or len(walkers) == 0:
        raise ValueError(
            f"initial_walkers must have shape (n_walkers, {', '.join(map(str, shape))})"
        )
    if not np.all(np.isfinite(walkers)):
        raise ValueError("initial_walkers contains NaN or infinity")
    return walkers
