"""Patch 27: exact two-replica interaction tempering for the canonical worm.

One replica samples the *physical* N=1 -> N=2 continuum auxiliary-path
Z/G target; a second, at a weaker interaction, is an auxiliary mixing aid.
Both have the same beta, cell, kinetic term, sector fugacity and normalized
free-path reference. Only the soft-core interaction amplitude differs.

Swapping complete auxiliary states (host, cut, length, extra, Z/G sector)
with a Metropolis ratio built from the **open-sector potential action**
leaves the product target invariant. Closed sectors carry zero interaction
action. No hot-replica samples enter the reported physical Green function.

This is not thermodynamic temperature tempering or a larger-N solution.
Performance must be measured *including both replicas* and swap attempts.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .bosons import SoftCorePairPotential
from .reliable_spacetime_worm import ReliableContinuumSpacetimeWorm, ReliableSpacetimeConfig


@dataclass(frozen=True)
class ReplicaExchangeConfig(ReliableSpacetimeConfig):
    hot_interaction_fraction: float = 0.25
    swap_probability: float = 0.3

    def validate(self):
        super().validate()
        for field in ("hot_interaction_fraction", "swap_probability"):
            value = getattr(self, field)
            try:
                numeric = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{field} must be a finite probability") from exc
            if not np.isfinite(numeric) or not 0 <= numeric <= 1:
                raise ValueError(f"{field} must lie in [0,1]")
        return self


class _ScaledPairPotential:
    """Same physical potential family, with a scalar coupling multiplier."""

    def __init__(self, potential: SoftCorePairPotential, fraction: float):
        self.base, self.fraction, self.cell = potential, fraction, potential.cell

    def __call__(self, positions):
        return self.fraction * self.base(positions)


class ReplicaExchangeContinuumWorm(ReliableContinuumSpacetimeWorm):
    """Cold worm with a weaker-interaction companion and reversible swaps.

    The Markov transition first updates *both* replicas with their existing
    invariant kernels, then optionally swaps their entire auxiliary states.
    The cold `step()` remains compatible with Patch 23 `audit_worm_chain`.
    The hot replica never supplies a measurement to the Green estimator.
    """

    def __init__(self, cell, potential, n_background=1,
                 config: ReplicaExchangeConfig | None = None, *,
                 kinetic_coefficient=0.5, indices=((0, 0, 0), (1, 0, 0))):
        if config is None:
            config = ReplicaExchangeConfig()
        if not isinstance(config, ReplicaExchangeConfig):
            raise TypeError("ReplicaExchangeConfig required")
        config.validate()
        if not isinstance(potential, SoftCorePairPotential):
            raise TypeError("replica exchange requires SoftCorePairPotential")
        if n_background != 1:
            raise ValueError("replica exchange benchmark requires one background boson")
        super().__init__(cell, potential, n_background, config,
                         kinetic_coefficient=kinetic_coefficient, indices=indices)
        self._physical_potential = potential
        self._hot_fraction = float(config.hot_interaction_fraction)
        hot_seed = int(config.seed) + 1000003
        hot_cfg = replace(config, seed=hot_seed)
        self.hot = ReliableContinuumSpacetimeWorm(
            cell, _ScaledPairPotential(potential, self._hot_fraction),
            n_background, hot_cfg, kinetic_coefficient=kinetic_coefficient,
            indices=indices)
        self._counts["exchange"] = [0, 0]
        self._open_exchange = [0, 0]

    def interaction_action(self, replica):
        """Dimensionless *full-strength* interaction action U for one state.

        The closed sector has no inserted two-boson path, hence U=0.
        In the open sector the primitive potential has half weight at both
        endpoints and full weight at interior beads. This must match the
        potential contribution in `log_relative_weight` exactly.
        """
        if not replica.open:
            return 0.0
        ell = replica.length
        idx = (replica.cut + np.arange(ell + 1)) % self.config.n_links
        positions = np.stack((replica.host[idx, 0, :], replica.extra), axis=1)
        values = np.asarray(self._physical_potential(positions), dtype=float)
        if values.shape != (ell + 1,) or not np.all(np.isfinite(values)):
            raise ValueError("nonfinite or wrongly shaped pair interaction action")
        return float(self.config.time_step * (0.5 * (values[0] + values[-1])
                                               + values[1:-1].sum()))

    def log_exchange_ratio(self):
        """Exact MH log ratio for swapping the full states of the two replicas.

        pi_lambda(x) = q(x) * [1_Z + R_free(x)*exp(-lambda U(x))*1_G].
        With lambda_cold=1 and lambda_hot=f, all q and free Bose kernel
        factors cancel; log alpha = (1-f)*(U_cold-U_hot).
        """
        return ((1.0 - self._hot_fraction)
                * (self.interaction_action(self) - self.interaction_action(self.hot)))

    @staticmethod
    def _get_state(replica):
        return (replica.host, replica.cut, replica.length, replica.extra, replica.open)

    @staticmethod
    def _set_state(replica, state):
        replica.host, replica.cut, replica.length, replica.extra, replica.open = state

    def move_exchange(self):
        """State-independent symmetric swap proposal with correct MH ratio."""
        lr = self.log_exchange_ratio()
        is_open = self.open or self.hot.open
        self._counts["exchange"][1] += 1
        if is_open:
            self._open_exchange[1] += 1
        if np.log(self.rng.random()) < min(0.0, lr):
            cold_state = self._get_state(self)
            hot_state = self._get_state(self.hot)
            self._set_state(self, hot_state)
            self._set_state(self.hot, cold_state)
            self._counts["exchange"][0] += 1
            if is_open:
                self._open_exchange[0] += 1
            return True
        return False

    @property
    def open_swap_acceptance(self):
        return (self._open_exchange[0] / self._open_exchange[1]
                if self._open_exchange[1] else float("nan"))

    def step(self):
        # Both updates are individually invariant for their respective
        # temperatures. The subsequent exchange is invariant for the joint
        # product target; only the physical (cold) replica is measured.
        super().step()
        self.hot.step()
        if self.rng.random() < self.config.swap_probability:
            return self.move_exchange()
        return False
