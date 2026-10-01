from __future__ import annotations

import numpy as np

Array = np.ndarray


class GaussianTrial:
    """Psi_T(r) = exp(-alpha r^2 / 2)."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = float(alpha)

    def log_psi(self, r: Array) -> Array:
        return -0.5 * self.alpha * np.sum(r * r, axis=1)

    def grad_log_psi(self, r: Array) -> Array:
        return -self.alpha * r

    def laplacian_log_psi(self, r: Array) -> Array:
        return np.full(len(r), -3.0 * self.alpha)


class ExponentialTrial:
    """Generic radial trial Psi_T(r) = exp(-alpha r)."""

    def __init__(self, alpha: float = 1.0, r_eps: float = 1e-12):
        self.alpha = float(alpha)
        self.r_eps = float(r_eps)

    def _radius(self, r: Array) -> Array:
        return np.maximum(np.linalg.norm(r, axis=1), self.r_eps)

    def log_psi(self, r: Array) -> Array:
        return -self.alpha * self._radius(r)

    def grad_log_psi(self, r: Array) -> Array:
        radius = self._radius(r)
        return -self.alpha * r / radius[:, None]

    def laplacian_log_psi(self, r: Array) -> Array:
        return -2.0 * self.alpha / self._radius(r)


class YukawaExponentialTrial(ExponentialTrial):
    """
    Psi_T = exp(-alpha r) for the Yukawa potential.

    Provides an analytic local energy. If alpha=g the Coulomb-like cusp at
    r=0 is satisfied and the 1/r singularity in E_L cancels.
    """

    def __init__(
        self,
        *,
        g: float = 1.0,
        mu: float = 0.5,
        alpha: float | None = None,
        r_eps: float = 1e-12,
    ):
        self.g = float(g)
        self.mu = float(mu)
        super().__init__(alpha=g if alpha is None else alpha, r_eps=r_eps)

    def local_energy(self, r: Array) -> Array:
        radius = self._radius(r)
        # alpha - g*exp(-mu*r), written to reduce cancellation when alpha≈g.
        numerator = self.alpha - self.g - self.g * np.expm1(-self.mu * radius)
        return numerator / radius - 0.5 * self.alpha**2


class YukawaCuspTrial:
    """
    Cusp-preserving Yukawa trial wavefunction.

    log Psi_T(r) =
        -alpha*r
        -(g-alpha)*r/(1 + beta*r)

    Properties:

        d log(Psi)/dr | r=0 = -g

    while

        d log(Psi)/dr -> -alpha

    for large r.
    """

    def __init__(
        self,
        g=1.0,
        mu=0.5,
        alpha=0.8,
        beta=1.0,
        r_eps=1e-12,
    ):
        self.g = float(g)
        self.mu = float(mu)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.r_eps = float(r_eps)

    def _radius(self, r):
        radius = np.linalg.norm(
            r,
            axis=1,
        )

        return np.maximum(
            radius,
            self.r_eps,
        )

    def log_psi(self, r):
        radius = self._radius(r)

        c = (
            self.g
            - self.alpha
        )

        return (
            -self.alpha * radius
            - c
            * radius
            / (
                1.0
                + self.beta * radius
            )
        )

    def grad_log_psi(self, r):
        radius = self._radius(r)

        c = (
            self.g
            - self.alpha
        )

        s = (
            1.0
            + self.beta * radius
        )

        dlog_dr = (
            -self.alpha
            - c / s**2
        )

        return (
            dlog_dr[:, None]
            * r
            / radius[:, None]
        )

    def laplacian_log_psi(self, r):
        radius = self._radius(r)

        c = (
            self.g
            - self.alpha
        )

        s = (
            1.0
            + self.beta * radius
        )

        dlog_dr = (
            -self.alpha
            - c / s**2
        )

        d2log_dr2 = (
            2.0
            * self.beta
            * c
            / s**3
        )

        return (
            d2log_dr2
            + 2.0
            * dlog_dr
            / radius
        )