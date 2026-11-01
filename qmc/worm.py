"""Canonical finite-temperature Z/G worm-sector benchmark for periodic bosons.

The path and permutation in Z are sampled by BosonPermutationPaths.  The G
sector opens ONE permutation closure link by a displacement ``s``. Its target
measure is

    pi_G(X,P,i,s) ds = eta/(N*V) * pi_Z(X,P) *
        K_tau(x[0,P[i]]+s-x[M-1,i]) / K_tau(x[0,P[i]]-x[M-1,i]) ds.

Averaging the translated link in the canonical bosonic trace gives g1(s).
Thus Z_G/Z_Z=eta*n0/N and n_k/N=<1_G cos(k.s)>/(eta <1_Z>).
These are FINITE-TEMPERATURE canonical expectations for the primitive action,
not a pure projector/zero-temperature condensate fraction. No arbitrary
histogram normalization or unknown worm-sector constant is left unfixed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .boson_permutations import (BosonPermutationPaths, PermutationPathConfig,
                                free_one_particle_partition, _cycles_with_labels)
from .path_staging import periodic_bridge
from .interacting_path import _integer, _positive
from .path_staging import autocorrelation_estimate


def _modes(indices):
    a = np.asarray(indices)
    if (a.ndim != 2 or a.shape[1] != 3 or len(a) < 1
            or not np.issubdtype(a.dtype, np.number)
            or not np.all(np.isfinite(a)) or not np.all(a == np.rint(a))):
        raise ValueError("modes must be a nonempty (n,3) integer array")
    if np.any(np.abs(a) > 100):
        raise ValueError("mode indices exceed supported numerical range")
    b = a.astype(np.int64)
    if len(set(map(tuple, b.tolist()))) != len(b):
        raise ValueError("duplicate reciprocal modes")
    if not np.any(np.all(b == 0, axis=1)):
        raise ValueError("include (0,0,0) to measure n0/N")
    return b


def canonical_free_occupations(cell, n_particles, beta, indices,
                               kinetic_coefficient=0.5):
    """Exact fixed-N ideal-Bose occupations n_k/N on a periodic torus.

    Z_N=(1/N)*sum_(l=1)^N Z1(l*beta)*Z_(N-l), Z0=1;
    <n_k>=sum_(l=1)^N exp(-l*beta*D*k^2)*Z_(N-l)/Z_N.
    """
    from .cells import OrthorhombicCell
    if not isinstance(cell, OrthorhombicCell):
        raise TypeError("orthorhombic cell required")
    n = _integer(n_particles, "n_particles", 1)
    beta, D = _positive(beta, "beta"), _positive(kinetic_coefficient, "kinetic_coefficient")
    modes = _modes(indices)
    z = np.zeros(n + 1, dtype=float)
    z[0] = 1.0
    z1 = np.array([0.0] + [free_one_particle_partition(cell, j*beta, D)
                            for j in range(1, n+1)])
    for j in range(1, n+1):
        z[j] = sum(z1[l]*z[j-l] for l in range(1, j+1))/j
    if not np.all(np.isfinite(z)) or z[n] <= 0:
        raise FloatingPointError("canonical partition recursion overflow")
    k2 = np.sum(cell.wavevectors(modes)**2, axis=1)
    occ = np.zeros(len(modes), dtype=float)
    for l in range(1, n+1):
        occ += np.exp(-l*beta*D*k2)*z[n-l]
    return occ/(n*z[n])


@dataclass(frozen=True)
class WormConfig(PermutationPathConfig):
    sector_fugacity: float = 1.0  # eta; known, dimensionless relative Z/G weight
    toggle_probability: float = 0.35
    collapsed_refresh_probability: float = 0.16
    gibbs_probability: float = 0.80
    local_shift_probability: float = 0.2
    shift_sigma: float = 0.50
    radial_bins: int = 5

    def validate(self):
        super().validate()
        _positive(self.sector_fugacity, "sector_fugacity")
        _positive(self.shift_sigma, "shift_sigma")
        for key in ("toggle_probability", "collapsed_refresh_probability", "gibbs_probability", "local_shift_probability"):
            x = getattr(self, key)
            if not np.isfinite(x) or not 0 <= x <= 1:
                raise ValueError(f"{key} must be in [0,1]")
        if self.toggle_probability <= 0:
            raise ValueError("toggle_probability must be positive")
        if self.toggle_probability + self.collapsed_refresh_probability > 1:
            raise ValueError("toggle and collapsed refresh probabilities must sum to <= 1")
        _integer(self.radial_bins, "radial_bins", 1)
        return self


@dataclass(frozen=True)
class WormResult:
    beta: float
    indices: np.ndarray
    occupation_fractions: np.ndarray
    radius: np.ndarray
    g1_radial: np.ndarray
    n_closed: int
    n_open: int
    closed_occupation_fractions: np.ndarray
    sector_effective_samples: float
    acceptance: dict[str, float]
    attempted: dict[str, int]
    closed_mean_exchange: float

    @property
    def n0_fraction(self):
        idx = np.where(np.all(self.indices == 0, axis=1))[0]
        return float(self.occupation_fractions[int(idx[0])])


class CanonicalWorm:
    """Positive Z/G extension of Patch 17's canonical PIMC sampler.

    In Z: full Patch 17 path/permutation moves. In G: displacement moves with
    fixed underlying paths and permutation. Every G point connects to a Z
    point by close; therefore Z moves plus G displacements form an ergodic
    extended sampler (given an ergodic Z engine).

    Heatbath opening: choose link i uniformly, draw y~K_tau(y), set
    s=y-(x0[P[i]]-xlast[i]) modulo cell; proposal q(s)=K_tau(delta+s).
    Opening MH ratio is eta/(V*K_tau(delta)), independent of sampled s.
    Closing MH is its inverse. G heatbath shift redraws s with acceptance 1.
    Other local-shift proposals use symmetric wrapped Gaussians.
    """

    def __init__(self, cell, potential, n_particles, config: WormConfig = WormConfig(),
                 *, kinetic_coefficient=0.5, indices=((0, 0, 0), (1, 0, 0),
                                                      (0, 1, 0), (0, 0, 1))):
        if not isinstance(config, WormConfig):
            raise TypeError("config must be WormConfig")
        self.config = config.validate()
        self.base = BosonPermutationPaths(cell, potential, n_particles, config,
                                          kinetic_coefficient=kinetic_coefficient)
        self.cell, self.kernel = self.base.cell, self.base.kernel
        self.rng = self.base.rng
        self.n_particles = self.base.n_particles
        self.indices = _modes(indices)
        self.wavevectors = cell.wavevectors(self.indices)
        self.open = False
        self.particle = None
        self.shift = None
        self._counts = {key: [0, 0] for key in ("open", "close", "sector_gibbs", "collapsed_refresh", "shift_heatbath", "shift_local")}

    def _delta(self, i):
        return self.base.path[0, self.base.permutation[i]] - self.base.path[-1, i]

    def log_sector_ratio(self, particle, shift):
        """log(pi_G/(pi_Z * ds)) for a given link and Cartesian displacement.

        The discrete link choice has measure 1/N; shift has measure ds. The
        density includes eta/(N V), and the kernel ratio. Useful for tests.
        """
        i = _integer(particle, "particle", 0)
        if i >= self.n_particles:
            raise IndexError("particle out of range")
        s = np.asarray(shift, dtype=float)
        if s.shape != (3,) or not np.all(np.isfinite(s)):
            raise ValueError("shift must be finite (3,)")
        delta = self._delta(i)
        tau = self.config.time_step
        return float(np.log(self.config.sector_fugacity)
                     - np.log(self.n_particles*self.cell.volume)
                     + self.kernel.log_kernel(delta+s, tau)
                     - self.kernel.log_kernel(delta, tau))

    def _accept(self, key, log_ratio):
        self._counts[key][1] += 1
        if not np.isfinite(log_ratio):
            raise FloatingPointError("nonfinite worm log ratio")
        accepted = np.log(self.rng.random()) < min(0.0, float(log_ratio))
        if accepted:
            self._counts[key][0] += 1
        return bool(accepted)

    def _heatbath_shift(self, particle):
        delta = self._delta(particle)
        y = self.kernel.sample_step(np.zeros(3), self.config.time_step, self.rng)
        return self.cell.wrap(y - delta)

    def move_toggle(self):
        if not self.open:
            i = int(self.rng.integers(self.n_particles))
            s = self._heatbath_shift(i)
            logk = float(self.kernel.log_kernel(self._delta(i), self.config.time_step))
            log_ratio = (np.log(self.config.sector_fugacity)
                         - np.log(self.cell.volume) - logk)
            if self._accept("open", log_ratio):
                self.open, self.particle, self.shift = True, i, s
                return True
            return False
        logk = float(self.kernel.log_kernel(self._delta(self.particle), self.config.time_step))
        log_ratio = (np.log(self.cell.volume) + logk
                     - np.log(self.config.sector_fugacity))
        if self._accept("close", log_ratio):
            self.open, self.particle, self.shift = False, None, None
            return True
        return False

    def conditional_sector_probabilities(self):
        """Exact sector probabilities at the current path/permutation.

        Integrating the G_i displacement leaves weight eta/(N V K_tau(delta_i)).
        This makes a rejection-free sector Gibbs update possible, even for
        interacting potential weights, because the potential action cancels
        conditional on the unchanged underlying path and permutation.
        """
        delta = self.base.path[0, self.base.permutation] - self.base.path[-1]
        logk = np.asarray(self.kernel.log_kernel(delta, self.config.time_step))
        logg = (np.log(self.config.sector_fugacity)
                - np.log(self.n_particles*self.cell.volume) - logk)
        z = np.concatenate(([0.0], logg))
        z -= np.max(z)
        weights = np.exp(z)
        return weights / np.sum(weights)

    def _collapsed_log_normalizer(self, path, perm):
        delta = path[0, perm] - path[-1]
        k = np.asarray(self.kernel.log_kernel(delta, self.config.time_step))
        logweights = np.r_[0., np.log(self.config.sector_fugacity)
                            - np.log(self.n_particles*self.cell.volume) - k]
        m = float(np.max(logweights))
        return m + np.log(float(np.exp(logweights-m).sum()))

    def move_collapsed_refresh(self):
        """Independent exact-free-boson path proposal for the *sector-marginal* target.

        Integrating out the sector and G displacements gives
        pi(X,P) ∝ q_free(X,P) * exp(-tau*sum V) * F(X,P),
        F=1+sum_i eta/(N V K_tau(Delta_i)).  The independent free proposal
        cancels the free kinetic/permutation part.  A conditional sector Gibbs
        resampling immediately after the MH decision restores the joint law.
        This is especially useful when a tiny closure K traps an open sector.
        """
        base = self.base
        ir = int(self.rng.choice(len(base._free_permutations),
                                 p=base._free_probabilities))
        perm = base._free_permutations[ir].copy()
        m = self.config.n_beads
        path = np.empty_like(base.path)
        for cycle in _cycles_with_labels(perm):
            start = self.cell.to_cartesian(self.rng.random((1,3)))
            loop = periodic_bridge(self.kernel, self.rng, start, start,
                                   m*len(cycle), self.config.time_step,
                                   method="staging")
            for offset, strand in enumerate(cycle):
                path[:,strand] = loop[offset*m:(offset+1)*m,0]
        dv = float(np.sum(base._potential(path)-base._potential(base.path)))
        logratio = (-self.config.time_step*dv
                    +self._collapsed_log_normalizer(path,perm)
                    -self._collapsed_log_normalizer(base.path,base.permutation))
        if self._accept("collapsed_refresh", logratio):
            base.path, base.permutation = path, perm
            accepted=True
        else:
            accepted=False
        self.move_sector_gibbs()
        return accepted

    def move_sector_gibbs(self):
        """Rejection-free, exact heatbath over Z and all G_i sectors."""
        probs = self.conditional_sector_probabilities()
        choice = int(self.rng.choice(len(probs), p=probs))
        if choice == 0:
            self.open, self.particle, self.shift = False, None, None
        else:
            self.open, self.particle = True, choice-1
            self.shift = self._heatbath_shift(self.particle)
        self._counts["sector_gibbs"][0] += 1
        self._counts["sector_gibbs"][1] += 1
        return True

    def move_shift(self, *, method="heatbath"):
        if not self.open:
            raise RuntimeError("shift moves require the open sector")
        if method == "heatbath":
            self.shift = self._heatbath_shift(self.particle)
            self._counts["shift_heatbath"][0] += 1
            self._counts["shift_heatbath"][1] += 1
            return True
        if method != "local":
            raise ValueError("method must be 'heatbath' or 'local'")
        trial_shift = self.cell.wrap(self.shift + self.config.shift_sigma*self.rng.normal(size=3))
        delta = self._delta(self.particle)
        log_ratio = (self.kernel.log_kernel(delta+trial_shift, self.config.time_step)
                     - self.kernel.log_kernel(delta+self.shift, self.config.time_step))
        if self._accept("shift_local", float(log_ratio)):
            self.shift = trial_shift
            return True
        return False

    def step(self):
        u = self.rng.random()
        if u < self.config.collapsed_refresh_probability:
            return self.move_collapsed_refresh()
        if u < self.config.collapsed_refresh_probability+self.config.toggle_probability:
            if self.rng.random() < self.config.gibbs_probability:
                return self.move_sector_gibbs()
            return self.move_toggle()
        if self.open:
            method = "local" if self.rng.random() < self.config.local_shift_probability else "heatbath"
            return self.move_shift(method=method)
        return self.base.step()

    def run(self):
        """Measure sector-ratio occupations and radial g1 with no fitted scale.

        Radial bins stop at min(L)/2 so shells have their ordinary 4pi*r²
        volume in the cubic/orthorhombic primitive torus. Histograms are noisy
        near zero, and must not be used alone to infer BEC.
        """
        cfg = self.config
        bins = np.linspace(0.0, 0.5*min(self.cell.lengths), cfg.radial_bins+1)
        volumes = (4*np.pi/3)*(bins[1:]**3-bins[:-1]**3)
        hist = np.zeros(cfg.radial_bins, dtype=float)
        numerators = np.zeros(len(self.indices), dtype=float)
        sectors = []
        nZ = nG = 0
        exchanged = []
        closed_moments = np.zeros(len(self.indices), dtype=float)
        for step in range(cfg.n_steps):
            self.step()
            if step < cfg.burn_in or (step-cfg.burn_in) % cfg.thin:
                continue
            if self.open:
                nG += 1
                numerators += np.cos(self.wavevectors @ self.shift)
                r = float(np.linalg.norm(self.cell.minimum_image(self.shift)))
                if r < bins[-1]:
                    j = min(int(np.searchsorted(bins, r, side="right")-1), cfg.radial_bins-1)
                    hist[j] += 1
                sectors.append(1.)
            else:
                nZ += 1
                closed_moments += closed_link_occupations(self.base, self.indices)
                sectors.append(0.)
                p = self.base.permutation
                exchanged.append(float(np.any(p != np.arange(self.n_particles))))
        if min(nZ, nG) < max(10, int(0.01*(nZ+nG))):
            raise RuntimeError("insufficient Z/G visits; increase steps, improve mixing or tune sector_fugacity")
        occ = numerators/(cfg.sector_fugacity*nZ)
        g1 = (hist/(cfg.sector_fugacity*nZ))*(self.cell.volume/volumes)
        a = autocorrelation_estimate(sectors)
        acceptance = {name: accepted/attempted if attempted else float("nan")
                      for name, (accepted, attempted) in {**self.base._counts, **self._counts}.items()}
        attempted = {name: v[1] for name, v in {**self.base._counts, **self._counts}.items()}
        return WormResult(beta=cfg.beta, indices=self.indices.copy(),
                          occupation_fractions=occ, radius=(bins[1:]+bins[:-1])/2,
                          g1_radial=g1, n_closed=nZ, n_open=nG,
                          closed_occupation_fractions=closed_moments/nZ,
                          sector_effective_samples=a.effective_samples,
                          acceptance=acceptance, attempted=attempted,
                          closed_mean_exchange=float(np.mean(exchanged)))


def closed_link_occupations(sampler: BosonPermutationPaths, indices):
    """Conditional, non-worm estimator at the current *closed* configuration.

    Integrating the translated closure kernel against exp(-ik.s) exactly gives
       exp(-tau*D*k²) * cos(k.delta)/(V*K_tau(delta)).
    Average this over independently sampled Z configurations for a second
    estimator of the SAME finite-temperature primitive-action occupations.
    """
    modes = _modes(indices)
    if not isinstance(sampler, BosonPermutationPaths):
        raise TypeError("sampler must be BosonPermutationPaths")
    p, x = sampler.permutation, sampler.path
    delta = x[0, p] - x[-1]
    k = sampler.cell.wavevectors(modes)
    tau, D = sampler.config.time_step, sampler.kernel.kinetic_coefficient
    denominator = sampler.cell.volume * np.exp(sampler.kernel.log_kernel(delta, tau))
    if np.any(denominator == 0) or not np.all(np.isfinite(denominator)):
        raise FloatingPointError("invalid closure weight")
    phase = np.cos(delta @ k.T)
    return (phase / denominator[:, None]).mean(axis=0) * np.exp(-tau*D*np.sum(k*k, axis=1))
