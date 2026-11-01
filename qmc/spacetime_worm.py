"""Exactly normalized spacetime-worm *lattice benchmark* for canonical bosons.

A fixed-N Bose-Hubbard ring is the Z sector. A G configuration has a creation
insertion at (tail_site,tail_time) and an annihilation at (head_site,head_time),
with N+1 bosons between them. The independent endpoint times differ by
s=1,...,M-1 imaginary-time lattice links. The exact matrix exponentials are
small finite-basis transfer kernels, not a scalable continuum PIMC solver.

Closed weight Z_N=Tr(exp(-beta H_N)). For fixed s,i,j, the *integrated* G weight
is F[s,i,j]=Tr(T_N^(M-s) a_i T_(N+1)^s a_j^dagger).
One G state (tail time t, separation s, head site i, tail site j) has weight
eta * F[s,i,j] / [M*(M-1)*L**2]. Z has weight Z_N.

The exact Green function is F/Z_N; normalized MC estimator for a displacement
d=(i-j) mod L averages the L origins and M tail time positions:
  G(d,s*tau) = L*(M-1)/eta * count(G with d,s) / count(Z).
This is the *particle-addition* time-ordered Green function, not the equal-time
one-body density matrix, condensate fraction, or a pure PIGS estimator.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt

import numpy as np

from .pure_projector import BoseHubbardRing, ExactBosonProjector


def _integer(x, name, minimum):
    if isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)) or x < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(x)


def _positive(x, name):
    try:
        x = float(x)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must be finite and positive") from exc
    if not np.isfinite(x) or x <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return x


@dataclass(frozen=True)
class SpacetimeWormConfig:
    time_step: float = 0.20
    n_slices: int = 8
    n_steps: int = 16000
    burn_in: int = 3000
    thin: int = 3
    fugacity: float = 1.0
    toggle_probability: float = 0.35
    time_move_probability: float = 0.40
    space_move_probability: float = 0.25
    seed: int = 2026

    def validate(self):
        _positive(self.time_step, "time_step")
        _positive(self.fugacity, "fugacity")
        for key, lower in (("n_slices", 4), ("n_steps", 8),
                           ("burn_in", 0), ("thin", 1), ("seed", 0)):
            _integer(getattr(self, key), key, lower)
        if self.burn_in >= self.n_steps:
            raise ValueError("burn_in must precede end of simulation")
        if (self.n_steps - self.burn_in + self.thin - 1)//self.thin < 10:
            raise ValueError("at least ten production measurements required")
        ps = [self.toggle_probability, self.time_move_probability, self.space_move_probability]
        if any(not np.isfinite(p) or p < 0 or p > 1 for p in ps) or not np.isclose(sum(ps), 1.0, atol=1e-12, rtol=0):
            raise ValueError("toggle/time/space probabilities must sum to one")
        if ps[0] <= 0 or ps[1] <= 0 or ps[2] <= 0:
            raise ValueError("all three move families must have positive probability")
        return self

    @property
    def beta(self):
        return self.time_step * self.n_slices


@dataclass(frozen=True)
class OccupationPath:
    """Occupation-number worldline with an N+1 segment for the G sector.

    For G: extra[0] = background[-1] + e_tail and
    background[0] = extra[-1] - e_head. The segments each include their end
    configurations and cover M time links in total. In Z, only background is
    populated and first=last. Time origin in G is the sampled tail_time.
    """
    open: bool
    tail_time: int | None
    head_time: int | None
    tail_site: int | None
    head_site: int | None
    extra: np.ndarray | None
    background: np.ndarray


@dataclass(frozen=True)
class SpacetimeWormResult:
    separation_time: np.ndarray
    displacement: np.ndarray
    green: np.ndarray
    n_closed: int
    n_open: int
    observed_open_fraction: float
    exact_open_fraction: float
    acceptance: dict[str, float]
    attempted: dict[str, int]
    time_counts: np.ndarray
    site_counts: np.ndarray


class ExactSpacetimeWorm:
    """Collapsed-endpoint MH worm with exact conditional occupation-path draws.

    Endpoints are updated independently in time and lattice position. Internal
    worldline configurations are integrated out *exactly* in the MH ratio;
    sample_path() draws the full discrete path from its conditional law. Thus
    this is a small-system reference, not a production worm with local bead
    updates or an approximate continuum short-time kernel.
    """

    def __init__(self, model: BoseHubbardRing,
                 config: SpacetimeWormConfig = SpacetimeWormConfig()):
        if not isinstance(model, BoseHubbardRing):
            raise TypeError("model must be BoseHubbardRing")
        if not isinstance(config, SpacetimeWormConfig):
            raise TypeError("config must be SpacetimeWormConfig")
        self.config = config.validate()
        self.model = model
        self.L, self.N, self.M = model.n_sites, model.n_bosons, config.n_slices
        self.n_projector = ExactBosonProjector(model)
        plus_model = BoseHubbardRing(
            n_sites=model.n_sites, n_bosons=model.n_bosons + 1,
            hopping=model.hopping, onsite_u=model.onsite_u,
            nearest_v=model.nearest_v, max_dimension=model.max_dimension)
        self.plus_projector = ExactBosonProjector(plus_model)
        self._n_basis = self.n_projector.basis
        self._p_basis = self.plus_projector.basis
        self._n_index = self.n_projector._index
        self._p_index = self.plus_projector._index
        self.rng = np.random.default_rng(config.seed)

        # A *common* shift preserves relative N and N+1 sector energies.
        shift = min(self.n_projector.energies[0], self.plus_projector.energies[0])
        self.tn = self._transfer(self.n_projector, shift)
        self.tp = self._transfer(self.plus_projector, shift)
        self.npowers = self._powers(self.tn)
        self.ppowers = self._powers(self.tp)
        self.z = float(np.trace(self.npowers[self.M]))
        if not np.isfinite(self.z) or self.z <= 0:
            raise FloatingPointError("invalid closed-sector partition function")

        dn, dp = len(self._n_basis), len(self._p_basis)
        self.create = np.zeros((self.L, dp, dn))
        self._raise_index = np.empty((self.L, dn), dtype=int)
        self._raise_amp = np.empty((self.L, dn), dtype=float)
        self._lower_index = np.full((self.L, dp), -1, dtype=int)
        self._lower_amp = np.zeros((self.L, dp), dtype=float)
        for j in range(self.L):
            for b, occ in enumerate(self._n_basis):
                new = occ.copy()
                new[j] += 1
                a = self._p_index[tuple(new)]
                value = sqrt(int(occ[j]) + 1)
                self.create[j, a, b] = value
                self._raise_index[j, b] = a
                self._raise_amp[j, b] = value
            for a, occ in enumerate(self._p_basis):
                if occ[j]:
                    old = occ.copy()
                    old[j] -= 1
                    self._lower_index[j, a] = self._n_index[tuple(old)]
                    self._lower_amp[j, a] = sqrt(int(occ[j]))

        self.f = np.zeros((self.M, self.L, self.L), dtype=float)
        for s in range(1, self.M):
            for i in range(self.L):
                for j in range(self.L):
                    value = np.trace(self.npowers[self.M-s] @ self.create[i].T
                                     @ self.ppowers[s] @ self.create[j])
                    if value < -1e-11 or not np.isfinite(value):
                        raise FloatingPointError("nonpositive or nonfinite Green weight")
                    self.f[s, i, j] = max(0.0, float(value))
        self.g_sector_weight = (config.fugacity * self.f.sum()
                                / ((self.M-1)*self.L**2))
        self.open = False
        self.tail_time = self.separation = self.head_site = self.tail_site = None
        self._counts = {k: [0, 0] for k in
                        ("open", "close", "head_time", "tail_time", "head_site", "tail_site")}

    def _transfer(self, projector, shift):
        dt = self.config.time_step
        v = projector.eigenvectors
        weights = np.exp(-dt * (projector.energies - shift))
        matrix = (v * weights[None, :]) @ v.T
        if not np.all(np.isfinite(matrix)) or np.min(matrix) < -1e-11:
            raise FloatingPointError("invalid stoquastic transfer matrix")
        return np.maximum(matrix, 0.0)

    def _powers(self, matrix):
        result = [np.eye(len(matrix))]
        for _ in range(self.M):
            result.append(result[-1] @ matrix)
        return np.asarray(result)

    def exact_green(self):
        return self.f / self.z

    def exact_open_fraction(self):
        w = self.g_sector_weight
        return float(w/(self.z+w))

    def exact_obdm(self):
        """Independent equal-time *thermal* OBDM, never inferred from G(dt)."""
        weights = np.exp(-self.config.beta*(self.n_projector.energies-
                                            min(self.n_projector.energies[0], self.plus_projector.energies[0])))
        rho = np.zeros((self.L, self.L), dtype=float)
        for idx, prob in enumerate(weights):
            rho += float(prob)*np.asarray(self.n_projector.obdm(
                self.n_projector.eigenvectors[:, idx]), dtype=float)
        return rho / weights.sum()

    def exact_zero_momentum_fraction(self):
        return float(np.sum(self.exact_obdm()) / (self.L*self.N))

    def _accept(self, kind, ratio):
        self._counts[kind][1] += 1
        if not np.isfinite(ratio) or ratio < 0:
            raise FloatingPointError("invalid worm Metropolis probability ratio")
        ok = self.rng.random() < min(1.0, float(ratio))
        if ok:
            self._counts[kind][0] += 1
        return bool(ok)

    def move_toggle(self):
        """Uniform G-label proposal from Z; deterministic inverse from G."""
        if not self.open:
            t = int(self.rng.integers(self.M))
            s = int(self.rng.integers(1, self.M))
            i, j = int(self.rng.integers(self.L)), int(self.rng.integers(self.L))
            if self._accept("open", self.config.fugacity*self.f[s,i,j]/self.z):
                self.open = True
                self.tail_time, self.separation = t, s
                self.head_site, self.tail_site = i, j
                return True
            return False
        ratio = self.z/(self.config.fugacity*self.f[self.separation,self.head_site,self.tail_site])
        if self._accept("close", ratio):
            self.open = False
            self.tail_time = self.separation = self.head_site = self.tail_site = None
            return True
        return False

    @property
    def head_time(self):
        return (self.tail_time+self.separation)%self.M if self.open else None

    def _move_time(self, who, direction=None):
        if not self.open:
            raise RuntimeError("time moves require the worm sector")
        if direction is None:
            direction = int(self.rng.choice((-1, 1)))
        if direction not in (-1, 1):
            raise ValueError("direction must be -1 or +1")
        s = self.separation + (direction if who == "head" else -direction)
        kind = who+"_time"
        if not (1 <= s < self.M):
            self._counts[kind][1] += 1
            return False
        ratio = self.f[s,self.head_site,self.tail_site]/self.f[self.separation,self.head_site,self.tail_site]
        if self._accept(kind, ratio):
            if who == "tail":
                self.tail_time = (self.tail_time+direction)%self.M
            self.separation = s
            return True
        return False

    def move_head_time(self, direction=None):
        return self._move_time("head", direction)

    def move_tail_time(self, direction=None):
        return self._move_time("tail", direction)

    def _move_site(self, who, destination=None):
        if not self.open:
            raise RuntimeError("spatial moves require the worm sector")
        old = self.head_site if who == "head" else self.tail_site
        if destination is None:
            # Symmetric all-to-all proposal among the L-1 alternative sites.
            k = int(self.rng.integers(self.L-1))
            destination = k + (k >= old)
        if not isinstance(destination, (int, np.integer)) or not 0 <= destination < self.L or destination == old:
            raise ValueError("destination must be a different valid site")
        i,j = ((destination,self.tail_site) if who=="head" else (self.head_site,destination))
        ratio = self.f[self.separation,i,j]/self.f[self.separation,self.head_site,self.tail_site]
        if self._accept(who+"_site", ratio):
            if who=="head": self.head_site=destination
            else: self.tail_site=destination
            return True
        return False

    def move_head_site(self, destination=None):
        return self._move_site("head", destination)

    def move_tail_site(self, destination=None):
        return self._move_site("tail", destination)

    def step(self):
        cfg = self.config
        u = self.rng.random()
        if u < cfg.toggle_probability:
            return self.move_toggle()
        if not self.open:
            return False  # Z has no endpoint to move; preserve symmetric toggle schedule.
        u -= cfg.toggle_probability
        if u < cfg.time_move_probability:
            if self.rng.integers(2):
                return self.move_head_time()
            return self.move_tail_time()
        if self.rng.integers(2):
            return self.move_head_site()
        return self.move_tail_site()

    def _draw(self, probabilities):
        w = np.asarray(probabilities, dtype=float).reshape(-1)
        if np.any(w < -1e-12) or not np.all(np.isfinite(w)):
            raise FloatingPointError("invalid bridge probabilities")
        w = np.maximum(w, 0)
        z = float(w.sum())
        if z <= 0:
            raise FloatingPointError("conditional path has zero weight")
        return int(self.rng.choice(len(w), p=w/z))

    def _bridge(self, t, power, start, end, links):
        sites = [int(start)]
        here = int(start)
        for remaining in range(links, 0, -1):
            weights = t[here,:]*power[remaining-1][:,end]
            here = self._draw(weights)
            sites.append(here)
        if sites[-1] != end:
            raise RuntimeError("bridge endpoint mismatch")
        return np.asarray(sites, dtype=int)

    def sample_path(self):
        """Draw genuine occupation-number worldlines conditional on Z or G.

        In G, the two separate segments contain N+1 and N particles,
        connected by the *actual* a^dagger and a insertion matrix elements.
        """
        if not self.open:
            diag = np.diag(self.npowers[self.M])
            first = self._draw(diag)
            indices = self._bridge(self.tn,self.npowers,first,first,self.M)
            return OccupationPath(False,None,None,None,None,None,self._n_basis[indices].copy())
        s,i,j = self.separation,self.head_site,self.tail_site
        dN,dP = len(self._n_basis),len(self._p_basis)
        y0 = self._raise_index[j]
        xend = self._lower_index[i]
        factors = self._raise_amp[j][:,None] * self._lower_amp[i][None,:]
        weights = (factors * self.ppowers[s][y0[:,None],np.arange(dP)[None,:]]
                   * self.npowers[self.M-s][np.maximum(xend,0)[None,:],np.arange(dN)[:,None]])
        weights[:,xend<0]=0.0
        total = float(weights.sum())
        if not np.isclose(total,self.f[s,i,j],atol=1e-10,rtol=1e-9):
            raise RuntimeError("conditional path normalization inconsistent with Green weight")
        picked=self._draw(weights)
        x0,ys=divmod(picked,dP)
        y0_state=int(y0[x0]); xs_state=int(xend[ys])
        extra_indices=self._bridge(self.tp,self.ppowers,y0_state,ys,s)
        background_indices=self._bridge(self.tn,self.npowers,xs_state,x0,self.M-s)
        return OccupationPath(True,self.tail_time,self.head_time,j,i,
                              self._p_basis[extra_indices].copy(),
                              self._n_basis[background_indices].copy())

    def run(self):
        cfg = self.config
        hist = np.zeros((self.M-1,self.L), dtype=int)
        tail_hist = np.zeros(self.M,dtype=int)
        site_hist = np.zeros((self.L,self.L),dtype=int)
        nZ=nG=0
        for step in range(cfg.n_steps):
            self.step()
            if step < cfg.burn_in or (step-cfg.burn_in)%cfg.thin:
                continue
            if self.open:
                nG+=1
                d=(self.head_site-self.tail_site)%self.L
                hist[self.separation-1,d]+=1
                tail_hist[self.tail_time]+=1
                site_hist[self.head_site,self.tail_site]+=1
            else:
                nZ+=1
        if nZ==0 or nG==0:
            raise RuntimeError("no Z/G mixing; increase steps or adjust fugacity")
        green = ((self.M-1)*self.L/cfg.fugacity)*hist/nZ
        accept={k: a/t if t else float("nan") for k,(a,t) in self._counts.items()}
        tried={k:t for k,(a,t) in self._counts.items()}
        return SpacetimeWormResult(
            separation_time=cfg.time_step*np.arange(1,self.M),
            displacement=np.arange(self.L),green=green,
            n_closed=nZ,n_open=nG,
            observed_open_fraction=nG/(nG+nZ),
            exact_open_fraction=self.exact_open_fraction(),
            acceptance=accept,attempted=tried,time_counts=tail_hist,site_counts=site_hist)
