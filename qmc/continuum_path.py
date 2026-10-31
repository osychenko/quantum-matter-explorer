"""Periodic free-particle open-path benchmark for imaginary-time coherence.

The physical system is a noninteracting periodic 3D particle, or N bosons in
an identical factorized positive orbital. This is NOT an interacting continuum
reptation/PIGS implementation. No bosonic permutation sampling is required
for this factorized reference, but it will be required for general systems.

Path weight: phi(x[0]) * prod K_tau(x[j], x[j+1]) * phi(x[-1]).
Independent-path MH and bidirectional reptation satisfy detailed balance for
this exact free kernel. The *blocked* free-particle off-diagonal estimator is
an exact finite-beta pure (two-sided) projector expectation. It cannot be
used unchanged with interacting propagators.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cells import OrthorhombicCell


def _positive(value, label):
    x = float(value)
    if not np.isfinite(x) or x <= 0:
        raise ValueError(f"{label} must be finite and positive")
    return x


def _paths(paths, n_beads):
    x = np.asarray(paths, dtype=float)
    if x.ndim != 3 or x.shape[1:] != (n_beads, 3) or x.shape[0] < 1 or not np.all(np.isfinite(x)):
        raise ValueError(f"paths must be finite (samples, {n_beads}, 3) array")
    return x


class PeriodicFreePropagator:
    r"""Exact free torus kernel K_t(x,y) via convergent Gaussian image sums.

    K_t(delta) = prod_d sum_m exp[-(delta_d+mL_d)^2/(4 D t)] / sqrt(4 pi D t).
    Each one-dimensional sum is truncated beyond at least eight diffusion
    standard deviations (small relative absolute error); no minimum-image
    approximation is used in the kernel itself. The sampling distribution is
    EXACTLY the wrapped normal corresponding to the infinite image sum.
    """

    def __init__(self, cell: OrthorhombicCell, kinetic_coefficient: float = 0.5):
        if not isinstance(cell, OrthorhombicCell):
            raise TypeError("free periodic reference requires an OrthorhombicCell")
        self.cell = cell
        self.kinetic_coefficient = _positive(kinetic_coefficient, "kinetic_coefficient")

    def log_kernel(self, delta, time: float):
        time = _positive(time, "time")
        dr = np.asarray(delta, dtype=float)
        if dr.ndim < 1 or dr.shape[-1] != 3 or not np.all(np.isfinite(dr)):
            raise ValueError("delta must be finite with final dimension 3")
        D = self.kinetic_coefficient
        sigma = np.sqrt(2.0 * D * time)
        total = np.zeros(dr.shape[:-1], dtype=float)
        for axis, L in enumerate(self.cell.lengths):
            x = dr[..., axis] - L * np.rint(dr[..., axis] / L)
            # At least 8 Gaussian standard deviations outside the centered
            # image interval; the neglected normal tail is <~1.3e-15.
            radius = max(1, int(np.ceil(0.5 + 8.0 * sigma / L)))
            shifts = np.arange(-radius, radius + 1, dtype=float) * L
            exponents = -0.5 * ((x[..., None] + shifts) / sigma) ** 2
            largest = np.max(exponents, axis=-1)
            total += (largest + np.log(np.exp(exponents - largest[..., None]).sum(axis=-1))
                      - np.log(np.sqrt(2.0 * np.pi) * sigma))
        return total

    def sample_step(self, positions, time: float, rng: np.random.Generator):
        time = _positive(time, "time")
        positions = np.asarray(positions, dtype=float)
        if positions.shape[-1:] != (3,) or not np.all(np.isfinite(positions)):
            raise ValueError("positions must have final dimension 3 and be finite")
        return self.cell.wrap(positions + np.sqrt(2 * self.kinetic_coefficient * time)
                              * rng.normal(size=positions.shape))


class PeriodicCosineOrbital:
    """Positive periodic trial phi(x)=1+a cos(2 pi mode x_axis/L_axis).

    Amplitude zero gives the exact ideal uniform Bose ground-state orbital.
    The n-boson product of this orbital is exchange symmetric. Other than
    trial-state endpoints, there are no interactions in Patch 14.
    """

    def __init__(self, cell: OrthorhombicCell, amplitude: float = 0.7,
                 axis: int = 0, mode: int = 1):
        if not isinstance(cell, OrthorhombicCell):
            raise TypeError("orbital requires an OrthorhombicCell")
        a = float(amplitude)
        if not np.isfinite(a) or abs(a) >= 1.0:
            raise ValueError("amplitude must be finite with absolute value < 1")
        if isinstance(axis, (bool, np.bool_)) or not isinstance(axis, (int, np.integer)) or axis not in (0, 1, 2):
            raise ValueError("axis must be 0, 1 or 2")
        if isinstance(mode, (bool, np.bool_)) or not isinstance(mode, (int, np.integer)) or mode < 1:
            raise ValueError("mode must be a positive integer")
        self.cell, self.amplitude, self.axis, self.mode = cell, a, int(axis), int(mode)
        self.k = 2.0 * np.pi * mode / cell.lengths[axis]

    def log_amplitude(self, x):
        arr = np.asarray(x, dtype=float)
        if arr.shape[-1:] != (3,) or not np.all(np.isfinite(arr)):
            raise ValueError("positions must have last axis 3 and be finite")
        return np.log1p(self.amplitude * np.cos(self.k * arr[..., self.axis]))

    def exact_coherence(self, beta: float, displacements, kinetic_coefficient: float = 0.5):
        beta = float(beta)
        if not np.isfinite(beta) or beta < 0:
            raise ValueError("beta must be finite and nonnegative")
        D = _positive(kinetic_coefficient, "kinetic_coefficient")
        shifts = np.asarray(displacements, dtype=float)
        if shifts.shape[-1:] != (3,) or not np.all(np.isfinite(shifts)):
            raise ValueError("displacements must be finite (...,3)")
        a_beta = self.amplitude * np.exp(-D * self.k**2 * beta)
        q = 0.5 * a_beta**2
        g1 = (1 + q * np.cos(self.k * shifts[..., self.axis])) / (1 + q)
        return g1, 1.0 / (1.0 + q)

    def exact_radial(self, beta: float, radii, kinetic_coefficient: float = 0.5):
        """Exact spherical orientation average of g1(r) and k=0 fraction."""
        rad = np.asarray(radii, dtype=float)
        if np.any(~np.isfinite(rad)) or np.any(rad < 0):
            raise ValueError("radii must be finite and nonnegative")
        D = _positive(kinetic_coefficient, "kinetic_coefficient")
        if not np.isfinite(beta) or beta < 0:
            raise ValueError("beta must be finite and nonnegative")
        a_beta = self.amplitude * np.exp(-D * self.k**2 * beta)
        q = 0.5*a_beta*a_beta
        return (1 + q*np.sinc(self.k*rad/np.pi))/(1+q), 1/(1+q)


@dataclass(frozen=True)
class ContinuumPathConfig:
    time_step: float = 0.10
    n_links: int = 20  # must be even so beta = n_links*tau/2
    n_steps: int = 6000
    burn_in: int = 1600
    thin: int = 8
    refresh_probability: float = 0.25
    seed: int = 2026

    def validate(self):
        _positive(self.time_step, "time_step")
        for key, minimum in (("n_links", 2), ("n_steps", 2), ("burn_in", 0), ("thin", 1)):
            value = getattr(self, key)
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                    or value < minimum):
                raise ValueError(f"{key} must be an integer >= {minimum}")
        if self.n_links % 2:
            raise ValueError("n_links must be even")
        if self.burn_in >= self.n_steps:
            raise ValueError("burn_in must be smaller than n_steps")
        prob = float(self.refresh_probability)
        if not np.isfinite(prob) or not 0 <= prob <= 1:
            raise ValueError("refresh_probability must be between zero and one")
        if isinstance(self.seed, (bool, np.bool_)) or not isinstance(self.seed, (int, np.integer)) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        return self

    @property
    def beta(self):
        return self.time_step * self.n_links / 2


@dataclass(frozen=True)
class ContinuumPathResult:
    paths: np.ndarray
    refresh_acceptance: float
    reptation_acceptance: float
    beta: float


class FreeOpenPathSampler:
    """Open-path MH sampler for an exact periodic free-particle kernel.

    Refresh proposes the entire path under a uniform-start free random walk;
    reptation grows a new bead at either end. Endpoint phi ratios give exact
    Metropolis-Hastings accept/reject rules. For |a|<1 the refresh proposal
    gives an everywhere-positive, ergodic move, including if reptation mixes
    poorly. No energy offset E_ref is required (it cancels in normalized P).
    """

    def __init__(self, kernel: PeriodicFreePropagator, orbital: PeriodicCosineOrbital,
                 config: ContinuumPathConfig = ContinuumPathConfig()):
        if not isinstance(kernel, PeriodicFreePropagator) or not isinstance(orbital, PeriodicCosineOrbital):
            raise TypeError("kernel and orbital must be periodic free reference objects")
        if not np.array_equal(kernel.cell.vectors, orbital.cell.vectors):
            raise ValueError("kernel and orbital must have the same periodic cell")
        self.kernel, self.orbital, self.config = kernel, orbital, config.validate()
        self.rng = np.random.default_rng(int(config.seed))
        self.path = self._free_path()
        self.accepted_refresh = self.attempted_refresh = 0
        self.accepted_reptation = self.attempted_reptation = 0

    def _free_path(self):
        cfg = self.config
        start = self.kernel.cell.to_cartesian(self.rng.random(3))
        increments = (np.sqrt(2 * self.kernel.kinetic_coefficient * cfg.time_step)
                      * self.rng.normal(size=(cfg.n_links, 3)))
        # A wrapped Gaussian walk can be constructed by wrapping the whole
        # cumulative unwrapped trajectory once; this is exactly the same path
        # proposal density but avoids one Python/NumPy call per bead.
        unwrapped = start + np.vstack((np.zeros(3), np.cumsum(increments, axis=0)))
        return self.kernel.cell.wrap(unwrapped)

    def _end_log(self, x):
        return float(self.orbital.log_amplitude(x[0]) + self.orbital.log_amplitude(x[-1]))

    def step(self, *, refresh: bool | None = None):
        if refresh is None:
            refresh = self.rng.random() < self.config.refresh_probability
        if refresh:
            self.attempted_refresh += 1
            proposal = self._free_path()
            log_accept = self._end_log(proposal) - self._end_log(self.path)
        else:
            self.attempted_reptation += 1
            proposal = np.empty_like(self.path)
            if self.rng.random() < 0.5:
                proposal[:-1] = self.path[1:]
                proposal[-1] = self.kernel.sample_step(self.path[-1], self.config.time_step, self.rng)
            else:
                proposal[1:] = self.path[:-1]
                proposal[0] = self.kernel.sample_step(self.path[0], self.config.time_step, self.rng)
            log_accept = self._end_log(proposal) - self._end_log(self.path)
        accepted = np.log(self.rng.random()) < min(0.0, log_accept)
        if accepted:
            self.path = proposal
            if refresh:
                self.accepted_refresh += 1
            else:
                self.accepted_reptation += 1
        return bool(accepted)

    def run(self):
        cfg = self.config
        samples = []
        for i in range(cfg.n_steps):
            self.step()
            if i >= cfg.burn_in and (i - cfg.burn_in) % cfg.thin == 0:
                samples.append(self.path.copy())
        return ContinuumPathResult(
            paths=np.asarray(samples),
            refresh_acceptance=(self.accepted_refresh / self.attempted_refresh
                                if self.attempted_refresh else float("nan")),
            reptation_acceptance=(self.accepted_reptation / self.attempted_reptation
                                  if self.attempted_reptation else float("nan")),
            beta=cfg.beta,
        )


@dataclass(frozen=True)
class FreeCoherenceResult:
    radii: np.ndarray
    g1: np.ndarray
    n0_fraction: float
    n_paths: int
    estimator: str


def measure_free_coherence(kernel: PeriodicFreePropagator, path_result: ContinuumPathResult,
                           radii, *, draws: int = 8, seed: int = 4000,
                           estimator: str = "endpoint") -> FreeCoherenceResult:
    """Blocked *pure* two-sided projector ratio for the free Hamiltonian.

    endpoint estimator: K_(2beta)(x0, x_end+s)/K_(2beta)(x0,x_end).
    midpoint estimator: K_beta(x0, x_mid+s)/K_beta(x0,x_mid).
    Both exactly integrate out free-path subchains and have the same expected
    projected g1(s), but their finite-MC variances are different. A full-volume
    displacement integral for n0 is analytically Rao-Blackwellized:
    (1/V) integral ds K_t(x,y+s)/K_t(x,y) = 1/(V K_t(x,y)).

    These identities require the *free* translationally invariant propagator.
    Never use this estimator unchanged for interacting systems.
    """
    if estimator not in ("endpoint", "midpoint"):
        raise ValueError("estimator must be 'endpoint' or 'midpoint'")
    if isinstance(draws, (bool, np.bool_)) or not isinstance(draws, (int, np.integer)) or draws < 1:
        raise ValueError("draws must be a positive integer")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be nonnegative integer")
    if not isinstance(path_result, ContinuumPathResult) or not np.isfinite(path_result.beta) or path_result.beta <= 0:
        raise ValueError("path_result must contain a valid positive beta")
    x = np.asarray(path_result.paths)
    x = _paths(x, x.shape[1] if x.ndim == 3 else -1)
    if (x.shape[1] - 1) % 2:
        raise ValueError("path must have an even number of links")
    if estimator == "endpoint":
        left, right, time = x[:, 0], x[:, -1], 2 * path_result.beta
    else:
        left, right, time = x[:, 0], x[:, (x.shape[1]-1)//2], path_result.beta
    r = np.asarray(radii, dtype=float)
    if r.ndim != 1 or len(r) < 2 or r[0] != 0 or np.any(np.diff(r) <= 0) or not np.all(np.isfinite(r)):
        raise ValueError("radii must start at zero and be strictly increasing")
    if r[-1] > 0.5 * min(kernel.cell.lengths) * (1 + 1e-12):
        raise ValueError("radii must not exceed the periodic cell inradius")
    rng = np.random.default_rng(int(seed))
    original = kernel.log_kernel(right - left, time)
    totals = np.ones(len(r), dtype=float)
    for j, radius in enumerate(r[1:], start=1):
        s = rng.normal(size=(draws, len(x), 3))
        lengths = np.linalg.norm(s, axis=-1)
        while np.any(lengths == 0):
            bad = lengths == 0
            s[bad] = rng.normal(size=(int(bad.sum()), 3))
            lengths = np.linalg.norm(s, axis=-1)
        s *= radius / lengths[..., None]
        logratio = kernel.log_kernel(right[None, :, :] + s - left[None, :, :], time) - original[None, :]
        ratio = np.exp(logratio)
        if not np.all(np.isfinite(ratio)):
            raise FloatingPointError("off-diagonal kernel ratio overflow")
        totals[j] = float(np.mean(ratio))
    # Exactly integrate over a *uniform* displacement in the full primitive
    # cell; no noisy random-displacement cubature is needed for n0.
    occupation_terms = np.exp(-original) / kernel.cell.volume
    if not np.all(np.isfinite(occupation_terms)):
        raise FloatingPointError("occupation kernel ratio overflow")
    n0 = float(np.mean(occupation_terms))
    return FreeCoherenceResult(radii=r.copy(), g1=totals, n0_fraction=n0,
                               n_paths=len(x), estimator=estimator)
