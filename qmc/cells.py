"""Three-dimensional periodic simulation cells, independent of VMC/DMC.

Each cell is a Bravais *translation lattice*: row i of ``vectors`` is the
Cartesian lattice vector a_i. Cartesian positions use the last array axis,
so operations work on ``(3,)``, ``(N, 3)``, ``(W, N, 3)``, and pair arrays.

For periodic pair separations, always use ``cell.displacement(a, b)`` or
``cell.minimum_image(b - a)`` rather than raw Euclidean differences.
"""

from __future__ import annotations

from itertools import product

import numpy as np


def _cartesian(value, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim < 1 or array.shape[-1] != 3:
        raise ValueError(f"{name} must have last dimension 3; got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite coordinates")
    return array


class PeriodicCell:
    """General 3D Bravais cell specified by three Cartesian row vectors.

    ``wrap()`` maps positions into a primitive *parallelepiped* by default.
    ``wrap(centered=True)`` instead maps them to the Wigner--Seitz cell
    centered at the origin, using the exact minimum-image algorithm.

    For a skew lattice, rounding fractional coordinates component by
    component is NOT generally the nearest image. The general implementation
    searches lattice translations until a singular-value bound proves that
    no unvisited translation can give a shorter displacement. This is robust
    but slower than specialized orthorhombic/BCC methods.
    """

    def __init__(self, vectors):
        basis = np.asarray(vectors, dtype=np.float64)
        if basis.shape != (3, 3) or not np.all(np.isfinite(basis)):
            raise ValueError("vectors must be a finite 3x3 matrix")
        singular_values = np.linalg.svd(basis, compute_uv=False)
        if singular_values[-1] <= 1e-12 * singular_values[0]:
            raise ValueError("cell lattice vectors must be linearly independent")
        determinant = float(np.linalg.det(basis))
        if not np.isfinite(determinant) or determinant == 0:
            raise ValueError("cell volume must be finite and nonzero")

        self._vectors = basis.copy()
        self._inverse = np.linalg.inv(basis)
        self._volume = abs(determinant)
        self._smallest_singular_value = float(singular_values[-1])

    @property
    def vectors(self) -> np.ndarray:
        """Cartesian primitive vectors, stored as rows (defensive copy)."""
        return self._vectors.copy()

    @property
    def volume(self) -> float:
        return self._volume

    @property
    def reciprocal_vectors(self) -> np.ndarray:
        """Rows b_i obey a_i dot b_j = 2*pi*delta_ij."""
        return (2.0 * np.pi * self._inverse.T).copy()

    def to_fractional(self, positions) -> np.ndarray:
        return _cartesian(positions, "positions") @ self._inverse

    def to_cartesian(self, fractional) -> np.ndarray:
        return _cartesian(fractional, "fractional") @ self._vectors

    def wavevectors(self, indices) -> np.ndarray:
        """Reciprocal-lattice k-vectors for integer triples (..., 3)."""
        values = _cartesian(indices, "indices")
        if not np.all(values == np.rint(values)):
            raise ValueError("reciprocal-lattice indices must be integers")
        return values @ self.reciprocal_vectors

    def wrap(self, positions, *, centered: bool = False) -> np.ndarray:
        """Map positions into a periodic representative of the cell.

        ``centered=False`` uses fractional coordinates in [0, 1) per axis.
        ``centered=True`` uses the Wigner--Seitz cell around the origin.
        """
        values = _cartesian(positions, "positions")
        if centered:
            return self.minimum_image(values)
        fractional = values @ self._inverse
        return (fractional - np.floor(fractional)) @ self._vectors

    def displacement(self, a, b) -> np.ndarray:
        """Shortest periodic displacement from ``a`` to ``b``."""
        return self.minimum_image(_cartesian(b, "b") - _cartesian(a, "a"))

    def minimum_image(self, displacement) -> np.ndarray:
        """Return the shortest Cartesian displacement among periodic images.

        A sphere bound from the smallest singular value certifies that the
        expanding integer-translation search has found a global minimum.
        Avoid this generic fallback in a high-throughput pair potential if a
        specialized lattice formulation is available.
        """
        delta = _cartesian(displacement, "displacement")
        if delta.size == 0:
            return delta.copy()

        # First reduce large translations to fractional coordinates in
        # [-1/2, 1/2]. This is only a *starting candidate*, not generally
        # the minimum image in a skew cell.
        fractional = delta @ self._inverse
        centered = fractional - np.rint(fractional)
        best = centered @ self._vectors
        best2 = np.sum(best * best, axis=-1)
        radius = 1
        base_displacement = best.copy()

        while True:
            for index in product(range(-radius, radius + 1), repeat=3):
                if max(abs(x) for x in index) != radius:
                    continue  # previous shell
                shift = np.asarray(index, dtype=np.float64) @ self._vectors
                candidate = base_displacement - shift
                dist2 = np.sum(candidate * candidate, axis=-1)
                shorter = dist2 < best2
                best = np.where(shorter[..., None], candidate, best)
                best2 = np.where(shorter, dist2, best2)

            # For any unvisited n, at least one fractional component of
            # (centered - n) has magnitude >= radius + 0.5. Therefore
            # |(centered - n) @ A| >= s_min(A) * (radius + 0.5).
            lower_bound2 = (self._smallest_singular_value * (radius + 0.5)) ** 2
            if np.all(best2 <= lower_bound2):
                return best
            radius += 1


class OrthorhombicCell(PeriodicCell):
    """Rectangular periodic box with positive Cartesian side lengths."""

    def __init__(self, lengths):
        sides = np.asarray(lengths, dtype=np.float64)
        if sides.shape != (3,) or not np.all(np.isfinite(sides)) or np.any(sides <= 0):
            raise ValueError("lengths must be three finite, positive numbers")
        self.lengths = tuple(float(s) for s in sides)
        super().__init__(np.diag(sides))

    def minimum_image(self, displacement) -> np.ndarray:
        delta = _cartesian(displacement, "displacement")
        sides = np.asarray(self.lengths)
        return delta - sides * np.rint(delta / sides)


class CubicCell(OrthorhombicCell):
    """Simple-cubic periodic box of side ``length``."""

    def __init__(self, length: float):
        super().__init__((length, length, length))
        self.length = float(length)


class BCCCell(PeriodicCell):
    """BCC translation lattice whose Wigner--Seitz cell is a truncated octahedron.

    ``length`` is the *conventional cubic* lattice constant. The primitive
    cell has volume ``length**3 / 2``. Default ``wrap()`` produces the
    primitive parallelepiped; ``wrap(centered=True)`` gives the actual
    truncated-octahedral Wigner--Seitz representation.

    The BCC lattice is the union of integer and half-integer simple-cubic
    translations. Testing those two Cartesian candidate sets gives an exact,
    vectorized minimum-image calculation without a 27-image search.
    """

    def __init__(self, length: float):
        side = float(length)
        if not np.isfinite(side) or side <= 0:
            raise ValueError("length must be finite and positive")
        self.length = side
        super().__init__(
            0.5 * side * np.array([
                [-1.0, 1.0, 1.0],
                [1.0, -1.0, 1.0],
                [1.0, 1.0, -1.0],
            ])
        )

    def minimum_image(self, displacement) -> np.ndarray:
        delta = _cartesian(displacement, "displacement")
        scaled = delta / self.length
        integer_image = delta - self.length * np.rint(scaled)
        half_image = delta - self.length * (np.rint(scaled - 0.5) + 0.5)
        use_half = (
            np.sum(half_image * half_image, axis=-1)
            < np.sum(integer_image * integer_image, axis=-1)
        )
        return np.where(use_half[..., None], half_image, integer_image)
