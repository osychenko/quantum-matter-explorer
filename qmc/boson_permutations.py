"""Canonical finite-temperature 3D boson ring polymers with permutation closure.

This is a *closed-worldline* PIMC reference, not a pure PIGS/OBDM method.
For M beads the primitive-action target is

    pi(R,P) ~ [prod_{s=0}^{M-2} K_tau(R_s,R_{s+1})]
             [prod_i K_tau(R_(M-1),i, R_0,P(i))]
             exp(-tau * sum_s V(R_s)).

Summing over P and integrating R gives the bosonic canonical partition
function (up to the same N! factor for every sector). P is a permutation
of particle indices in the *imaginary-time closure*, not a particle-label
swap of an entire open path. Exchanges here are equilibrium cycles at
finite temperature; the sampler does not measure condensate fraction.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

import numpy as np

from .cells import OrthorhombicCell
from .continuum_path import PeriodicFreePropagator
from .interacting_path import _integer, _positive
from .path_staging import periodic_bridge


def permutation_cycles(permutation) -> tuple[int, ...]:
    """Lengths of disjoint cycles, sorted largest-first."""
    p = np.asarray(permutation)
    n = len(p) if p.ndim == 1 else -1
    if (n < 1 or not np.issubdtype(p.dtype, np.integer)
            or not np.array_equal(np.sort(p), np.arange(n))):
        raise ValueError("permutation must contain exactly 0,..,N-1")
    seen = set()
    lengths = []
    for start in range(n):
        if start in seen:
            continue
        j, length = start, 0
        while j not in seen:
            seen.add(j)
            length += 1
            j = int(p[j])
        lengths.append(length)
    return tuple(sorted(lengths, reverse=True))


def free_two_boson_exchange_probability(cell: OrthorhombicCell, beta: float,
                                        kinetic_coefficient: float = 0.5) -> float:
    """Exact canonical free N=2 probability of an exchange closure.

    P(transposition) = Z1(2 beta) / (Z1(beta)^2 + Z1(2 beta)),
    with Z1(beta)=sum_{k in reciprocal lattice} exp(-beta*D*|k|^2).
    Each 1D reciprocal sum is truncated at a Gaussian tail of <1e-14.
    """
    if not isinstance(cell, OrthorhombicCell):
        raise TypeError("cell must be orthorhombic")
    beta, D = _positive(beta, "beta"), _positive(kinetic_coefficient, "kinetic_coefficient")

    one = free_one_particle_partition(cell, beta, D)
    two = free_one_particle_partition(cell, 2.0*beta, D)
    return float(two / (one * one + two))


def free_one_particle_partition(cell, beta, kinetic_coefficient=0.5) -> float:
    """One-particle periodic heat-kernel trace, reciprocal Gaussian series."""
    beta = _positive(beta, "beta")
    D = _positive(kinetic_coefficient, "kinetic_coefficient")
    total = 1.0
    for L in cell.lengths:
        a = beta * D * (2 * np.pi / L)**2
        if a <= np.pi:
            # Poisson transformation avoids huge reciprocal grids at small beta.
            dual = np.pi**2/a
            radius = max(1, int(np.ceil(np.sqrt(37.0 / dual))))
            k = np.arange(1, radius + 1, dtype=float)
            term = np.sqrt(np.pi/a)*(1.0 + 2.0*float(np.sum(np.exp(-dual*k*k))))
        else:
            radius = max(1, int(np.ceil(np.sqrt(37.0 / a))))
            k = np.arange(1, radius + 1, dtype=float)
            term = 1.0 + 2.0*float(np.sum(np.exp(-a*k*k)))
        total *= term
    return total


def _cycles_with_labels(permutation):
    p = np.asarray(permutation)
    seen = set()
    cycles = []
    for i in range(len(p)):
        if i in seen:
            continue
        seq = []
        j = i
        while j not in seen:
            seen.add(j)
            seq.append(j)
            j = int(p[j])
        cycles.append(seq)
    return cycles


@dataclass(frozen=True)
class PermutationPathConfig:
    time_step: float = 0.2
    n_beads: int = 8
    n_steps: int = 5000
    burn_in: int = 1000
    thin: int = 5
    move_sigma: float = 0.38
    swap_probability: float = 0.35
    bridge_probability: float = 0.30
    refresh_probability: float = 0.15
    translate_probability: float = 0.05
    bridge_lengths: tuple[int, ...] = (2, 4)
    seed: int = 2026

    def validate(self):
        _positive(self.time_step, "time_step")
        _positive(self.move_sigma, "move_sigma")
        for name, least in (("n_beads", 4), ("n_steps", 2),
                            ("burn_in", 0), ("thin", 1), ("seed", 0)):
            _integer(getattr(self, name), name, least)
        if self.burn_in >= self.n_steps:
            raise ValueError("burn_in must be less than n_steps")
        if (self.n_steps - self.burn_in + self.thin - 1) // self.thin < 4:
            raise ValueError("at least four production measurements required")
        probabilities = (self.swap_probability, self.bridge_probability,
                         self.translate_probability, self.refresh_probability)
        if any(not np.isfinite(p) or p < 0 or p > 1 for p in probabilities):
            raise ValueError("move probabilities must be finite and in [0,1]")
        if sum(probabilities) > 1 + 1e-12:
            raise ValueError("move probabilities must sum to <= 1")
        if not isinstance(self.bridge_lengths, (tuple, list)) or not self.bridge_lengths:
            raise ValueError("bridge_lengths must be nonempty")
        for length in self.bridge_lengths:
            _integer(length, "bridge length", 2)
            if length >= self.n_beads:
                raise ValueError("bridge length must be < n_beads (no closure crossing)")
        if len(set(self.bridge_lengths)) != len(self.bridge_lengths):
            raise ValueError("bridge lengths must be distinct")
        return self

    @property
    def beta(self) -> float:
        return self.n_beads * self.time_step


@dataclass(frozen=True)
class PermutationPathResult:
    beta: float
    midpoint_potential: np.ndarray
    exchange_indicator: np.ndarray
    longest_cycle_fraction: np.ndarray
    acceptance: dict[str, float]
    attempted: dict[str, int]
    permutation: np.ndarray
    final_path: np.ndarray


class BosonPermutationPaths:
    """Primitive closed-worldline PIMC, with swap and free-bridge updates.

    For a closure permutation p, the final bead of strand i is linked to
    first bead p[i].  Interior bridges do not cross this closure; updating P
    with a transposition changes only two closing kernels.  All selectors
    are independent of the state; detailed-balance ratios are exact for the
    *discretized primitive* target. No sign problem for nonnegative K and V.
    """

    def __init__(self, cell, potential, n_particles: int,
                 config: PermutationPathConfig = PermutationPathConfig(),
                 *, kinetic_coefficient: float = 0.5):
        if not isinstance(cell, OrthorhombicCell):
            raise TypeError("cell must be orthorhombic")
        self.cell = cell
        self.config = config.validate()
        self.n_particles = _integer(n_particles, "n_particles", 2)
        if self.n_particles > 5:
            raise ValueError("Patch 17 exact free-sector proposals are restricted to N<=5")
        self.potential = potential
        self.kernel = PeriodicFreePropagator(cell, kinetic_coefficient)
        if hasattr(potential, "cell") and not np.allclose(
                potential.cell.vectors, cell.vectors, rtol=1e-12, atol=1e-12):
            raise ValueError("potential cell must match simulation cell")
        self.rng = np.random.default_rng(self.config.seed)
        self.permutation = np.arange(self.n_particles, dtype=int)
        first = self.cell.to_cartesian(self.rng.random((self.n_particles, 3)))
        increments = np.sqrt(2 * self.kernel.kinetic_coefficient * self.config.time_step) * self.rng.normal(
            size=(self.config.n_beads - 1, self.n_particles, 3))
        self.path = self.cell.wrap(first[None, :, :] + np.concatenate(
            [np.zeros((1, self.n_particles, 3)), np.cumsum(increments, axis=0)]))
        self._counts = {name: [0, 0] for name in ("bead", "swap", "bridge", "translate", "refresh")}
        # Enumerate the free permutation sector weights only for small N.
        ps = np.asarray(list(permutations(range(self.n_particles))), dtype=int)
        weights = []
        for p in ps:
            weights.append(sum(np.log(free_one_particle_partition(
                self.cell, len(cycle)*self.config.beta,
                self.kernel.kinetic_coefficient)) for cycle in _cycles_with_labels(p)))
        weights = np.asarray(weights)
        weights = np.exp(weights - weights.max())
        self._free_permutations = ps
        self._free_probabilities = weights/weights.sum()
        self.log_weight()  # validate the potential and starting state

    def _potential(self, path):
        x = np.asarray(path, dtype=float)
        values = np.asarray(self.potential(x), dtype=float)
        if values.shape != (len(x),) or not np.all(np.isfinite(values)):
            raise ValueError("potential must return one finite value per configuration")
        return values

    def log_weight(self, path=None, permutation=None) -> float:
        x = self.path if path is None else np.asarray(path, dtype=float)
        p = self.permutation if permutation is None else np.asarray(permutation)
        if (x.shape != (self.config.n_beads, self.n_particles, 3)
                or not np.all(np.isfinite(x))):
            raise ValueError("path must be finite (n_beads,N,3)")
        permutation_cycles(p)  # validates closure exactly
        tau = self.config.time_step
        interior = self.kernel.log_kernel(x[1:] - x[:-1], tau).sum()
        closure = self.kernel.log_kernel(x[0, p] - x[-1], tau).sum()
        return float(interior + closure - tau * self._potential(x).sum())

    def _accept(self, kind, candidate_path, candidate_perm, log_ratio):
        self._counts[kind][1] += 1
        if not np.isfinite(log_ratio):
            raise FloatingPointError("nonfinite acceptance ratio")
        if np.log(self.rng.random()) < min(0.0, float(log_ratio)):
            self.path = candidate_path
            self.permutation = candidate_perm
            self._counts[kind][0] += 1
            return True
        return False

    def move_swap(self, a=None, b=None):
        if a is None:
            a, b = self.rng.choice(self.n_particles, size=2, replace=False)
        a, b = int(a), int(b)
        if a == b or not (0 <= a < self.n_particles and 0 <= b < self.n_particles):
            raise ValueError("swap requires two different valid particle indices")
        p = self.permutation
        q = p.copy()
        q[a], q[b] = q[b], q[a]
        x, tau = self.path, self.config.time_step
        delta = (self.kernel.log_kernel(x[0, q[[a,b]]] - x[-1, [a,b]], tau).sum()
                 - self.kernel.log_kernel(x[0, p[[a,b]]] - x[-1, [a,b]], tau).sum())
        return self._accept("swap", x, q, float(delta))

    def move_bead(self, bead=None, particle=None):
        m, n = self.config.n_beads, self.n_particles
        j = int(self.rng.integers(m)) if bead is None else int(bead)
        i = int(self.rng.integers(n)) if particle is None else int(particle)
        if not (0 <= j < m and 0 <= i < n):
            raise IndexError("bead or particle out of range")
        x = self.path
        proposed = x.copy()
        proposed[j, i] = self.cell.wrap(x[j, i] + self.config.move_sigma * self.rng.normal(size=3))
        tau = self.config.time_step
        dv = float(self._potential(proposed[j:j+1])[0] - self._potential(x[j:j+1])[0])
        prev = x[m-1, np.argsort(self.permutation)[i]] if j == 0 else x[j-1, i]
        next_ = x[0, self.permutation[i]] if j == m-1 else x[j+1, i]
        old = (self.kernel.log_kernel(x[j,i] - prev, tau)
               + self.kernel.log_kernel(next_ - x[j,i], tau))
        new = (self.kernel.log_kernel(proposed[j,i] - prev, tau)
               + self.kernel.log_kernel(next_ - proposed[j,i], tau))
        return self._accept("bead", proposed, self.permutation, float(new - old - tau*dv))

    def move_bridge(self, *, length=None, start=None, particle=None):
        m, n = self.config.n_beads, self.n_particles
        if length is None:
            length = int(self.rng.choice(self.config.bridge_lengths))
        length = _integer(length, "length", 2)
        if length >= m:
            raise ValueError("bridge cannot cross the permutation closure")
        if start is None:
            start = int(self.rng.integers(m - length))
        if particle is None:
            particle = int(self.rng.integers(n))
        if not (0 <= start < m - length and 0 <= particle < n):
            raise IndexError("bridge position or particle out of range")
        x = self.path
        proposed = x.copy()
        bridge = periodic_bridge(
            self.kernel, self.rng, x[start, particle:particle+1],
            x[start+length, particle:particle+1], length,
            self.config.time_step, method="staging")
        proposed[start+1:start+length, particle] = bridge[1:-1,0]
        old = self._potential(x[start+1:start+length])
        new = self._potential(proposed[start+1:start+length])
        delta = -self.config.time_step * float(np.sum(new-old))
        return self._accept("bridge", proposed, self.permutation, delta)

    def move_translate(self):
        # Uniform displacement of ALL particles and beads is exactly free
        # and interaction invariant. This tests global translation invariance.
        shift = self.config.move_sigma * self.rng.normal(size=3)
        proposed = self.cell.wrap(self.path + shift)
        # For general translationally invariant pair potentials, delta = 0.
        # Evaluate it in full so any supplied noninvariant external potential
        # still respects detailed balance.
        delta = self.log_weight(proposed, self.permutation) - self.log_weight()
        return self._accept("translate", proposed, self.permutation, delta)

    def move_refresh(self):
        """Independent exact *free bosonic* path proposal, MH only on V.

        A permutation P is drawn with weight prod_cycles Z1(length*beta),
        then each exchange cycle is drawn as a uniform-start periodic free
        Brownian loop of length length*M.  This is the correctly normalized
        free bosonic canonical path ensemble; in the interacting target the
        free proposal factors cancel and only tau*sum(V) remains.
        """
        index = int(self.rng.choice(len(self._free_permutations),
                                    p=self._free_probabilities))
        proposal_p = self._free_permutations[index].copy()
        proposed = np.empty_like(self.path)
        M = self.config.n_beads
        for cycle in _cycles_with_labels(proposal_p):
            start = self.cell.to_cartesian(self.rng.random((1,3)))
            loop = periodic_bridge(self.kernel, self.rng, start, start,
                                   M*len(cycle), self.config.time_step,
                                   method="staging")
            for offset, strand in enumerate(cycle):
                proposed[:,strand] = loop[offset*M:(offset+1)*M, 0]
        delta = -self.config.time_step * float(
            self._potential(proposed).sum()-self._potential(self.path).sum())
        return self._accept("refresh", proposed, proposal_p, delta)

    def step(self):
        u = self.rng.random()
        cfg = self.config
        if u < cfg.swap_probability:
            return self.move_swap()
        if u < cfg.swap_probability + cfg.bridge_probability:
            return self.move_bridge()
        u -= cfg.swap_probability + cfg.bridge_probability
        if u < cfg.refresh_probability:
            return self.move_refresh()
        if u < cfg.refresh_probability + cfg.translate_probability:
            return self.move_translate()
        return self.move_bead()

    def run(self) -> PermutationPathResult:
        potential, exchanged, longest = [], [], []
        for step in range(self.config.n_steps):
            self.step()
            if step >= self.config.burn_in and (step-self.config.burn_in) % self.config.thin == 0:
                potential.append(self._potential(self.path[self.config.n_beads//2:self.config.n_beads//2+1])[0])
                cycles = permutation_cycles(self.permutation)
                exchanged.append(float(cycles[0] > 1))
                longest.append(cycles[0] / self.n_particles)
        return PermutationPathResult(
            beta=self.config.beta, midpoint_potential=np.asarray(potential),
            exchange_indicator=np.asarray(exchanged), longest_cycle_fraction=np.asarray(longest),
            acceptance={k:v[0]/v[1] if v[1] else float("nan") for k,v in self._counts.items()},
            attempted={k:v[1] for k,v in self._counts.items()},
            permutation=self.permutation.copy(), final_path=self.path.copy())
