"""Patch 25: exactly reweighted, interaction-guided continuum insertion proposals.

For an N=1 free background, the G-sector auxiliary reference is
q0(host,cut,length,extra) = q_host * q_cut * q_length *
                            (1/V) prod_t K_tau(extra[t+1]-extra[t]).

Replace only the extra-path distribution by a normalized mixture
q_mix = (1-p) q0(extra) + p q_drift(extra|host,cut,length), where the
latter is a sequence of normalized *wrapped* Gaussian heat kernels with
bounded deterministic repulsive drifts.  We can evaluate both densities
exactly. The target is still q0 R, so independence MH uses the weight
R * q0/q_mix. Neither the Hamiltonian nor the Green-function estimator
normalization is changed.

This is designed for SoftCorePairPotential and N_background==1, NOT a
fully general N-body worm. It is exploratory until efficiency, overlap,
and correlated-ratio diagnostics have been demonstrated on held-out runs.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import numpy as np

from .bosons import SoftCorePairPotential
from .interacting_path import _positive
from .rare_worm import RareEventContinuumWorm, RareWormConfig, contribution_concentration


def _probability(value, name, *, strictly_below_one=False):
    try:
        x = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a probability") from exc
    if not np.isfinite(x) or x < 0 or (x >= 1 if strictly_below_one else x > 1):
        raise ValueError(f"{name} must be in {'[0,1)' if strictly_below_one else '[0,1]'}")
    return x


class GuidedExtraProposal:
    """Normalized q_mix(extra|host,cut,ell) with evaluated log(q0/qmix).

    The guide is an Euler-type drift **proposal**, not a potential-action
    approximation. At each next bead, a repulsive vector based on the next
    host position moves the wrapped-normal proposal center away from the host.
    Its magnitude is capped. In the q_drift density the drift must be
    evaluated using the *previous extra bead*, never the proposed next bead.

    `free_fraction=1-guided_fraction` guarantees a globally bounded
    importance ratio q0/qmix <= 1/free_fraction for guided_fraction < 1.
    """

    def __init__(self, kernel, potential: SoftCorePairPotential, *,
                 guided_fraction=0.75, drift_strength=0.9, max_drift=0.4,
                 initial_tilt_strength=1.0):
        if not isinstance(potential, SoftCorePairPotential):
            raise TypeError("guided proposal currently requires SoftCorePairPotential")
        if not np.allclose(kernel.cell.vectors, potential.cell.vectors, atol=1e-12, rtol=0):
            raise ValueError("potential and heat kernel must share a cell")
        self.kernel = kernel
        self.cell = kernel.cell
        self.potential = potential
        self.guided_fraction = _probability(guided_fraction, "guided_fraction", strictly_below_one=True)
        self.drift_strength = float(drift_strength)
        if not np.isfinite(self.drift_strength) or self.drift_strength < 0:
            raise ValueError("drift_strength must be finite and nonnegative")
        self.max_drift = _positive(max_drift, "max_drift")
        self.initial_tilt_strength = float(initial_tilt_strength)
        if not np.isfinite(self.initial_tilt_strength) or self.initial_tilt_strength < 0:
            raise ValueError("initial_tilt_strength must be finite and nonnegative")

    @lru_cache(maxsize=16)
    def _initial_log_normalizer(self, time_step):
        """Exact (up to bounded Gauss-Legendre quadrature) initial tilt norm.

        The finite-support radial pair potential fits within the minimum-image
        inradius; translating the host changes neither the integral nor Z.
        Thus q_initial(e|h) = exp(-alpha*V(e-h))/(V*Z), Z=E_uniform exp(-alpha V).
        This *known* Z is required in the mixture q0/qmix correction.
        """
        tau = _positive(time_step, "time_step")
        amplitude = self.initial_tilt_strength * 0.5 * tau * self.potential.height
        if amplitude == 0:
            return 0.0
        nodes, weights = np.polynomial.legendre.leggauss(64)
        radius = self.potential.cutoff * (nodes + 1) / 2
        weight_r = self.potential.cutoff * weights / 2
        profile = np.maximum(0., 1-(radius/self.potential.cutoff)**2)**3
        integral = np.sum(weight_r*radius**2 * (-np.expm1(-amplitude*profile)))
        norm = 1. - 4*np.pi/self.cell.volume * integral
        if not 0 < norm <= 1:
            raise FloatingPointError("invalid initial guided proposal normalizer")
        return float(np.log(norm))

    def _initial_log_guide_over_free(self, first, host_at_cut, time_step):
        tau = _positive(time_step, "time_step")
        if self.initial_tilt_strength == 0 or self.potential.height == 0:
            return 0.0
        disp = self.cell.minimum_image(np.asarray(first)-np.asarray(host_at_cut))
        profile = max(0.0,1-np.dot(disp,disp)/self.potential.cutoff**2)**3
        return (-self.initial_tilt_strength*.5*tau*self.potential.height*profile
                -self._initial_log_normalizer(tau))

    def _check(self, host, cut, extra):
        h = np.asarray(host, dtype=float)
        e = np.asarray(extra, dtype=float)
        if (h.ndim != 3 or h.shape[1:] != (1, 3) or len(h) < 3
                or e.ndim != 2 or e.shape[1:] != (3,) or not 2 <= len(e) <= len(h)
                or not isinstance(cut, (int, np.integer)) or not 0 <= cut < len(h)
                or not np.all(np.isfinite(h)) or not np.all(np.isfinite(e))):
            raise ValueError("host (M,1,3), cut, extra (ell+1,3) are invalid")
        return h, e

    def _drift(self, previous, host_next, time_step):
        delta = self.cell.minimum_image(np.asarray(previous) - np.asarray(host_next))
        r2 = float(np.dot(delta, delta))
        rc = self.potential.cutoff
        if r2 >= rc * rc or self.potential.height == 0 or self.drift_strength == 0:
            return np.zeros(3)
        factor = 6 * self.potential.height / rc**2 * (1 - r2 / rc**2)**2
        # Repulsive force: -grad_extra V = +factor*(extra-host).
        drift = self.drift_strength * self.kernel.kinetic_coefficient * time_step * factor * delta
        norm = float(np.linalg.norm(drift))
        return drift * min(1.0, self.max_drift / norm) if norm > 0 else drift

    def sample(self, host, cut, length, time_step, rng):
        h = np.asarray(host, dtype=float)
        if h.ndim != 3 or h.shape[1:] != (1, 3):
            raise ValueError("host must be (M,1,3)")
        if (not isinstance(length, (int, np.integer)) or not 1 <= length < len(h)
                or not isinstance(cut, (int, np.integer)) or not 0 <= cut < len(h)):
            raise ValueError("invalid cut or length")
        tau = _positive(time_step, "time_step")
        e = np.empty((length + 1, 3), dtype=float)
        use_guide = bool(rng.random() < self.guided_fraction)
        if use_guide and self.initial_tilt_strength > 0 and self.potential.height > 0:
            host0 = h[cut,0]
            tau_alpha = self.initial_tilt_strength*.5*tau*self.potential.height
            for _ in range(100000):  # bounded rejection sampler; nonzero floor
                initial = self.cell.to_cartesian(rng.random(3))
                delta = self.cell.minimum_image(initial-host0)
                profile = max(0.,1.-np.dot(delta,delta)/self.potential.cutoff**2)**3
                if np.log(rng.random()) < -tau_alpha*profile:
                    e[0] = initial
                    break
            else:
                raise RuntimeError("initial guided rejection sampler did not terminate")
        else:
            e[0] = self.cell.to_cartesian(rng.random(3))
        for t in range(length):
            center = e[t]
            if use_guide:
                center = center + self._drift(e[t], h[(cut+t+1) % len(h), 0], tau)
            e[t + 1] = self.kernel.sample_step(center, tau, rng)
        return e, use_guide

    def log_guide_over_free(self, host, cut, extra, time_step):
        h, e = self._check(host, cut, extra)
        tau = _positive(time_step, "time_step")
        out = self._initial_log_guide_over_free(e[0], h[cut,0], tau)
        for t in range(len(e)-1):
            delta = e[t+1]-e[t]
            drift = self._drift(e[t], h[(cut+t+1) % len(h), 0], tau)
            out += float(self.kernel.log_kernel(delta-drift, tau) - self.kernel.log_kernel(delta, tau))
        return out

    def log_reference_over_mixture(self, host, cut, extra, time_step):
        """log(q0/qmix), stable at large positive/negative guide weights."""
        p = self.guided_fraction
        if (p == 0 or self.potential.height == 0
                or (self.drift_strength == 0 and self.initial_tilt_strength == 0)):
            return 0.0
        log_ratio = self.log_guide_over_free(host, cut, extra, time_step)
        return -float(np.logaddexp(np.log1p(-p), np.log(p) + log_ratio))


@dataclass(frozen=True)
class GuidedWormConfig(RareWormConfig):
    multiple_try_probability: float = 0.0
    guided_move_probability: float = 0.8  # fraction of open global moves
    guided_fraction: float = 0.8         # qmix = (1-p)qfree + p*qdrift
    drift_strength: float = 0.9
    max_drift: float = 0.4
    initial_tilt_strength: float = 1.0

    def validate(self):
        super().validate()
        _probability(self.guided_move_probability, "guided_move_probability")
        _probability(self.guided_fraction, "guided_fraction", strictly_below_one=True)
        if not np.isfinite(self.drift_strength) or self.drift_strength < 0:
            raise ValueError("drift_strength must be finite and nonnegative")
        _positive(self.max_drift, "max_drift")
        if not np.isfinite(self.initial_tilt_strength) or self.initial_tilt_strength < 0:
            raise ValueError("initial_tilt_strength must be finite and nonnegative")
        return self


class GuidedContinuumWorm(RareEventContinuumWorm):
    """Same exact G-sector target as Patch 24 with corrected guided refreshes."""

    def __init__(self, cell, potential, n_background, config=None, *,
                 kinetic_coefficient=0.5, indices=((0,0,0),(1,0,0))):
        if config is None:
            config = GuidedWormConfig()
        if not isinstance(config, GuidedWormConfig):
            raise TypeError("GuidedWormConfig required")
        if n_background != 1:
            raise ValueError("interaction guide currently supports one background boson")
        super().__init__(cell, potential, n_background, config,
                         kinetic_coefficient=kinetic_coefficient, indices=indices)
        self.guide = GuidedExtraProposal(self.kernel, potential,
             guided_fraction=config.guided_fraction,
             drift_strength=config.drift_strength, max_drift=config.max_drift,
             initial_tilt_strength=config.initial_tilt_strength)
        self._counts["guided_refresh"] = [0, 0]

    def _draw_guided_state(self):
        host = self._new_host()
        cut = int(self.rng.integers(self.config.n_links))
        length = int(self.rng.integers(1,self.config.n_links))
        extra, _ = self.guide.sample(host,cut,length,self.config.time_step,self.rng)
        return host,cut,length,extra

    def _corrected_log_weight(self,state):
        host,cut,length,extra = state
        return (self.log_relative_weight(host,cut,length,extra)
                + self.guide.log_reference_over_mixture(
                    host,cut,extra,self.config.time_step))

    def move_guided_refresh(self):
        if not self.open:
            raise RuntimeError("guided refresh requires open sector")
        old = (self.host,self.cut,self.length,self.extra)
        new = self._draw_guided_state()
        logratio = self._corrected_log_weight(new) - self._corrected_log_weight(old)
        if self._mh("guided_refresh",logratio):
            self.host,self.cut,self.length,self.extra = new
            return True
        return False

    def move_global_refresh(self):
        p = self.config.guided_move_probability
        if self.open and p > 0 and (p == 1 or self.rng.random() < p):
            return self.move_guided_refresh()
        return super().move_global_refresh()


@dataclass(frozen=True)
class GuidedIndependentResult:
    imaginary_times: np.ndarray
    indices: np.ndarray
    green: np.ndarray
    stderr: np.ndarray
    top_one_pct: np.ndarray
    weight_ess: np.ndarray
    samples_per_interval: int


def independent_guided_green(sampler: GuidedContinuumWorm, *, draws_per_interval=400):
    """i.i.d. stratified qmix-reweighted G(k,t), with weight-tail diagnostics.

    Each draw reweights by R/eta * q0/qmix, preserving Patch 21's exact
    normalization. For k!=0 signed Fourier phases may give unstable means;
    weight_ess uses the positive importance weights, not an MCMC ESS.
    """
    if not isinstance(sampler, GuidedContinuumWorm):
        raise TypeError("requires GuidedContinuumWorm")
    if (isinstance(draws_per_interval, (bool,np.bool_))
            or not isinstance(draws_per_interval, (int,np.integer))
            or draws_per_interval < 4):
        raise ValueError("draws_per_interval must be integer >=4")
    M, tau, eta = sampler.config.n_links, sampler.config.time_step, sampler.config.sector_fugacity
    n = int(draws_per_interval)
    green = np.zeros((M-1,len(sampler.indices)))
    stderr = np.zeros_like(green)
    tails = np.zeros_like(green)
    weight_ess = np.zeros(M-1)
    for ell in range(1,M):
        vals = np.empty((n,len(sampler.indices)))
        weights = np.empty(n)
        for j in range(n):
            h = sampler._new_host()
            cut = int(sampler.rng.integers(M))
            e, _ = sampler.guide.sample(h,cut,ell,tau,sampler.rng)
            logw = (sampler.log_relative_weight(h,cut,ell,e) - np.log(eta)
                    + sampler.guide.log_reference_over_mixture(h,cut,e,tau))
            if logw > 700:
                raise FloatingPointError("importance weight overflow: recheck overlap")
            weights[j] = np.exp(logw)
            vals[j] = weights[j]*np.cos(sampler.wavevectors @ (e[-1]-e[0]))
        green[ell-1] = np.mean(vals,axis=0)
        stderr[ell-1] = np.std(vals,axis=0,ddof=1)/np.sqrt(n)
        tails[ell-1] = contribution_concentration(vals[:,None,:])[0]
        weight_ess[ell-1] = float(weights.sum()**2/np.dot(weights,weights)) if np.any(weights) else 0.
    return GuidedIndependentResult(tau*np.arange(1,M),sampler.indices.copy(),green,
                                   stderr,tails,weight_ess,n)
