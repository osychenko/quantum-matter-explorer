"""Auxiliary-path Gibbs sector updates and Rao-Blackwellization for Patch 21.

The auxiliary (cut, length, extra path) is retained even when the sampler is
closed.  In the closed sector it follows the EXACT normalized free proposal q.
The augmented target is q_host q_extra [1_Z + R(host,extra)*1_G].  Integrating
out the auxiliary data in Z and the sectors recovers the original Patch 21
partition-function ratio. No fitted normalization and no changed Hamiltonian.

This addresses avoidable Bernoulli-sector noise, but neither fixes the
primitive-action time-step bias nor guarantees decorrelation of rare open
configurations. Report replica errors and autocorrelation diagnostics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .continuum_spacetime_worm import (ContinuumSpacetimeWorm, ContinuumSpacetimeConfig)
from .path_staging import autocorrelation_estimate


@dataclass(frozen=True)
class ReliableSpacetimeConfig(ContinuumSpacetimeConfig):
    ghost_refresh_probability: float = 0.55  # within non-toggle, closed moves

    def validate(self):
        super().validate()
        p = self.ghost_refresh_probability
        if not np.isfinite(p) or p < 0 or p > 1:
            raise ValueError("ghost_refresh_probability must lie in [0,1]")
        return self


@dataclass(frozen=True)
class ReliableGreenResult:
    beta: float
    imaginary_times: np.ndarray
    indices: np.ndarray
    green: np.ndarray
    green_raw: np.ndarray
    n_closed: int
    n_open_by_length: np.ndarray
    closed_effective_samples: float
    conditional_effective_samples: float
    effective_samples_by_length: np.ndarray
    mean_closed_probability: float
    acceptance: dict[str, float]
    attempted: dict[str, int]


class ReliableContinuumSpacetimeWorm(ContinuumSpacetimeWorm):
    """Patch 21 target with auxiliary paths and Gibbs sector sampling.

    Closed-sector auxiliary variables are not physical worldlines. The
    conditional sector Bernoulli probability is pG=R/(1+R).  Replacing sector
    indicators by this known conditional probability leaves their expectation
    unchanged and removes conditional Bernoulli noise.
    """

    def __init__(self, cell, potential, n_background, config=None,
                 *, kinetic_coefficient=0.5, indices=((0,0,0),(1,0,0))):
        if config is None:
            config = ReliableSpacetimeConfig()
        if not isinstance(config, ReliableSpacetimeConfig):
            raise TypeError("config must be ReliableSpacetimeConfig")
        super().__init__(cell, potential, n_background, config,
                         kinetic_coefficient=kinetic_coefficient, indices=indices)
        self.cut, self.length, self.extra = self._draw_ghost()
        self._counts["gibbs"] = [0, 0]
        self._counts["ghost_refresh"] = [0, 0]

    def _draw_ghost(self):
        c = int(self.rng.integers(self.config.n_links))
        ell = int(self.rng.integers(1, self.config.n_links))
        return c, ell, self._new_extra(ell)

    def conditional_open_probability(self):
        """Stable logistic(log R), conditional on all auxiliary variables."""
        log_r = self.log_relative_weight(self.host, self.cut, self.length, self.extra)
        if log_r >= 0:
            return float(1/(1+np.exp(-log_r)))
        z = np.exp(log_r)
        return float(z/(1+z))

    def move_sector_gibbs(self):
        probability = self.conditional_open_probability()
        self.open = bool(self.rng.random() < probability)
        self._counts["gibbs"][0] += 1
        self._counts["gibbs"][1] += 1
        return True

    def move_ghost_refresh(self):
        if self.open:
            raise RuntimeError("ghost refresh is only unconditional in Z")
        self.cut, self.length, self.extra = self._draw_ghost()
        self._counts["ghost_refresh"][0] += 1
        self._counts["ghost_refresh"][1] += 1
        return True

    def move_global_refresh(self):
        if self.open:
            # G proposal is q_host*q_extra; Metropolis uses R'/R.
            return super().move_global_refresh()
        self.host = self._new_host()
        self.cut, self.length, self.extra = self._draw_ghost()
        self._counts["global_refresh"][0] += 1
        self._counts["global_refresh"][1] += 1
        return True

    def step(self):
        cfg = self.config
        u = self.rng.random()
        if u < cfg.toggle_probability:
            return self.move_sector_gibbs()
        if u < cfg.toggle_probability + cfg.global_refresh_probability:
            return self.move_global_refresh()
        if not self.open:
            if self.rng.random() < cfg.ghost_refresh_probability:
                return self.move_ghost_refresh()
            return self.move_host_refresh()
        v = self.rng.random()
        if v < cfg.time_move_probability:
            return self.move_time(which="head" if self.rng.random()<0.5 else "tail",
                                  direction="extend" if self.rng.random()<0.5 else "retract")
        if v < cfg.time_move_probability + cfg.endpoint_move_probability:
            return self.move_endpoint("head" if self.rng.random()<0.5 else "tail")
        if v < (cfg.time_move_probability + cfg.endpoint_move_probability
                + cfg.exchange_bridge_probability):
            return self.move_exchange_bridge()
        return self.move_extra_refresh()

    def run(self):
        """Return raw sector-ratio and conditional-expectation Green functions.

        Both use the same eta/(M-1) normalization. The conditional estimator
        is a Rao-Blackwellization with respect to Z/G, *not* an exact solution
        of the path autocorrelation or an off-diagonal pure ground-state OBDM.
        """
        M, cfg = self.config.n_links, self.config
        z_count, g_count = 0, np.zeros(M-1, dtype=int)
        raw = np.zeros((M-1, len(self.indices)))
        rb = np.zeros_like(raw)
        pz_sum = 0.
        z_trace, p_trace = [], []
        length_traces = [[] for _ in range(M-1)]
        for step in range(cfg.n_steps):
            self.step()
            if step < cfg.burn_in or (step-cfg.burn_in)%cfg.thin:
                continue
            pG = self.conditional_open_probability()
            pZ = 1-pG
            pz_sum += pZ
            p_trace.append(pG)
            z_trace.append(float(not self.open))
            ell = self.length-1
            phase = np.cos(self.wavevectors@(self.extra[-1]-self.extra[0]))
            rb[ell] += pG*phase
            for j in range(M-1):
                length_traces[j].append(pG if j == ell else 0.0)
            if self.open:
                g_count[ell] += 1
                raw[ell] += phase
            else:
                z_count += 1
        n = len(p_trace)
        if n < 4 or pz_sum <= 0 or not np.isfinite(pz_sum):
            raise RuntimeError("insufficient or nonfinite conditional sector statistics")
        if z_count < 4:
            raise RuntimeError("too few closed-sector samples; increase steps or tune eta")
        prefactor = (M-1)/cfg.sector_fugacity
        result = prefactor*rb/pz_sum
        raw_result = prefactor*raw/z_count
        # The per-length autocorrelation measures the weighted indicator
        # at every time point, including zeros when a different length is active.
        p_all = np.asarray(p_trace)
        ell_ess = [autocorrelation_estimate(trace).effective_samples
                   if np.any(np.asarray(trace) > 0) else 0.0
                   for trace in length_traces]
        return ReliableGreenResult(
            beta=cfg.beta, imaginary_times=cfg.time_step*np.arange(1,M),
            indices=self.indices.copy(), green=result, green_raw=raw_result,
            n_closed=z_count, n_open_by_length=g_count,
            closed_effective_samples=autocorrelation_estimate(z_trace).effective_samples,
            conditional_effective_samples=autocorrelation_estimate(p_trace).effective_samples,
            effective_samples_by_length=np.asarray(ell_ess),
            mean_closed_probability=pz_sum/n,
            acceptance={k:a/t if t else float("nan") for k,(a,t) in self._counts.items()},
            attempted={k:t for k,(_,t) in self._counts.items()},
        )
