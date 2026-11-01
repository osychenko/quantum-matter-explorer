"""Patch 24: defensive multiple-try *independence* updates for Patch 22 worms.

For the auxiliary-path G target pi_G(x) proportional to q(x) R(x),
q is the *normalized* independent free-worldline proposal from Patch 21.
A multiple-try step draws K candidates from q and resamples by R. The
reverse set contains the old state and K-1 independent q draws. The exact
multiple-try Metropolis acceptance is min(1, sum R_fwd / sum R_rev).

This is a reversible kernel for the SAME interacting primitive-action target.
It neither reweights physical measurements nor changes the worm fugacity.
It can expose high-weight rare configurations but is more expensive per step;
measure ESS / CPU-second and independent-replica agreement before use.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .reliable_spacetime_worm import (ReliableContinuumSpacetimeWorm,
                                      ReliableSpacetimeConfig)
from .interacting_path import _integer


def independent_multiple_try(current, log_importance, draw_reference, rng,
                             *, candidates=4):
    """Exact independence multiple-try MH step, including the reverse reference set.

    ``draw_reference`` must sample from a *normalized* probability density q,
    independent of the current configuration. ``log_importance(x)`` is
    log[pi(x)/q(x)] up to a state-independent constant, and must be finite.
    For K=1 this reduces exactly to ordinary independence Metropolis.

    Returns (state, accepted, log_acceptance). No proposal-density evaluation
    is needed because all q factors cancel in the forward/reverse construction.
    The scheme works for mixed discrete/continuous reference measures.
    """
    k = _integer(candidates, "candidates", 1)
    if k > 64:
        raise ValueError("candidates must be <=64 (bounded CPU work)")
    old_log = float(log_importance(current))
    if not np.isfinite(old_log):
        raise ValueError("current log importance must be finite")
    forward = [draw_reference() for _ in range(k)]
    log_forward = np.asarray([log_importance(x) for x in forward], dtype=float)
    if not np.all(np.isfinite(log_forward)):
        raise ValueError("all forward log importance values must be finite")
    log_sum_forward = float(np.logaddexp.reduce(log_forward))
    probs = np.exp(log_forward-log_sum_forward)
    chosen = int(rng.choice(k, p=probs / probs.sum()))
    # The selected forward candidate is not in the reverse reference set;
    # the old configuration is. This is essential for detailed balance.
    reverse = [draw_reference() for _ in range(k-1)]
    log_reverse = np.array([old_log] + [log_importance(x) for x in reverse], dtype=float)
    if not np.all(np.isfinite(log_reverse)):
        raise ValueError("all reverse log importance values must be finite")
    log_alpha = log_sum_forward - float(np.logaddexp.reduce(log_reverse))
    if not np.isfinite(log_alpha):
        raise FloatingPointError("invalid multiple-try MH ratio")
    accepted = bool(np.log(rng.random()) < min(0.0, log_alpha))
    return (forward[chosen] if accepted else current), accepted, float(log_alpha)


@dataclass(frozen=True)
class RareWormConfig(ReliableSpacetimeConfig):
    multiple_try_probability: float = 0.80  # when an open-sector global move occurs
    multiple_try_candidates: int = 4

    def validate(self):
        super().validate()
        p = float(self.multiple_try_probability)
        if not np.isfinite(p) or not 0 <= p <= 1:
            raise ValueError("multiple_try_probability must be in [0,1]")
        k = _integer(self.multiple_try_candidates, "multiple_try_candidates", 1)
        if k > 64:
            raise ValueError("multiple_try_candidates must be <=64")
        return self


class RareEventContinuumWorm(ReliableContinuumSpacetimeWorm):
    """Same Gibbs-sector worm target as Patch 22, with G-sector MTM refreshes.

    `multiple_try_probability=0` gives the original Patch 22 sampler and is
    the proper baseline. For K>1, one MTM attempt evaluates 2K-1 R values,
    and may be slower; compare *ESS per CPU time* rather than acceptance rates.
    Z-sector updates, conditional Gibbs choices, physical parameters, and
    estimators are otherwise untouched.
    """
    def __init__(self, cell, potential, n_background, config=None,
                 *, kinetic_coefficient=0.5, indices=((0,0,0),(1,0,0))):
        if config is None:
            config = RareWormConfig()
        if not isinstance(config, RareWormConfig):
            raise TypeError("RareWormConfig is required")
        super().__init__(cell, potential, n_background, config,
                         kinetic_coefficient=kinetic_coefficient, indices=indices)
        self._counts["multiple_try"] = [0, 0]

    def _independent_state(self):
        host = self._new_host()
        cut, length, extra = self._draw_ghost()
        return host, cut, length, extra

    def _importance(self, state):
        host, cut, length, extra = state
        return self.log_relative_weight(host, cut, length, extra)

    def move_multiple_try(self):
        if not self.open:
            raise RuntimeError("multiple-try refresh requires open sector")
        current = (self.host, self.cut, self.length, self.extra)
        state, accepted, _ = independent_multiple_try(
            current, self._importance, self._independent_state, self.rng,
            candidates=self.config.multiple_try_candidates)
        self._counts["multiple_try"][1] += 1
        if accepted:
            self.host, self.cut, self.length, self.extra = state
            self._counts["multiple_try"][0] += 1
        return accepted

    def move_global_refresh(self):
        if self.open and self.rng.random() < self.config.multiple_try_probability:
            return self.move_multiple_try()
        return super().move_global_refresh()


def contribution_concentration(numerator_trace, *, tail_fraction=0.01):
    """Absolute share of numerator mass from largest 1% of time observations.

    A *warning diagnostic*, not a convergence proof: for signed Fourier modes
    use absolute contributions, since cancellation makes signed shares unstable.
    Returns an array of shape (imaginary-time separation, mode).
    """
    a = np.asarray(numerator_trace, dtype=float)
    f = float(tail_fraction)
    if (a.ndim != 3 or len(a) < 4 or not np.all(np.isfinite(a))
            or not np.isfinite(f) or not 0 < f <= 1):
        raise ValueError("finite (time,separation,mode) array and 0<tail_fraction<=1 required")
    k = max(1, int(np.ceil(f*len(a))))
    absolute = np.abs(a)
    total = absolute.sum(axis=0)
    largest = np.sort(absolute, axis=0)[-k:].sum(axis=0)
    return np.divide(largest, total, out=np.zeros_like(total), where=total>0)


@dataclass(frozen=True)
class IndependentGreenResult:
    """i.i.d. free-reference importance estimates, separated by time interval."""
    imaginary_times: np.ndarray
    indices: np.ndarray
    green: np.ndarray
    stderr: np.ndarray
    top_one_pct: np.ndarray
    samples_per_interval: int


def independent_reference_green(sampler: RareEventContinuumWorm, *,
                                draws_per_interval=500):
    """Independent *stratified* q-reference estimator for Patch 21's target.

    Because N_background<=1, the *closed* background is exactly free.  For
    each length ell draw an independent closed host, uniform cut, and free
    extra strand, using q from Patch 21. Then

       G_k(ell*tau) = E_q[R(host,cut,ell,extra)/eta * cos(k*(head-tail))].

    No Markov-chain ratio, sector residency, or unknown partition function.
    Samples across lengths and independent draws are independent; however R
    may be heavy-tailed and ordinary standard errors can badly underestimate
    unreached rare events. This is an *independent cross-check*, not a cure for
    importance-sampling degeneracy or an extension to interacting N>=2 host.
    """
    if not isinstance(sampler, RareEventContinuumWorm):
        raise TypeError("requires RareEventContinuumWorm")
    n = _integer(draws_per_interval, "draws_per_interval", 4)
    M = sampler.config.n_links
    result = np.empty((M-1,len(sampler.indices)),dtype=float)
    stderr = np.empty_like(result)
    concentration = np.empty_like(result)
    for ell in range(1,M):
        measurements = np.empty((n,len(sampler.indices)),dtype=float)
        for j in range(n):
            host = sampler._new_host()
            cut = int(sampler.rng.integers(M))
            extra = sampler._new_extra(ell)
            log_value = sampler.log_relative_weight(host,cut,ell,extra) - np.log(sampler.config.sector_fugacity)
            if log_value > 700:
                raise FloatingPointError("large importance weight; increase reference sampling stability")
            measurements[j] = np.exp(log_value)*np.cos(sampler.wavevectors@(extra[-1]-extra[0]))
        result[ell-1] = measurements.mean(axis=0)
        stderr[ell-1] = measurements.std(axis=0,ddof=1)/np.sqrt(n)
        concentration[ell-1] = contribution_concentration(measurements[:,None,:])[0]
    return IndependentGreenResult(sampler.config.time_step*np.arange(1,M),
                                  sampler.indices.copy(),result,stderr,concentration,n)
