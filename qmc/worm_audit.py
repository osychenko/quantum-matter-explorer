"""Patch 23: autocorrelation-aware audits of normalized canonical worm ratios.

For every imaginary-time separation and momentum mode the worm observable is

    G = (M-1)/eta * E[p(G|aux) * 1(length=ell) * cos(k.dr)] / E[1-p(G|aux)].

Numerator and denominator are measured on the SAME correlated Markov chain.
Uncertainty uses block-jackknife of the *ratio*, not an incorrect error bar
on a sector count. Independent chains and split-Rhat remain separate checks.

This does not change the finite-tau primitive path action, the Hamiltonian,
or the scope N_background <= 1 of Patch 21/22.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .reliable_spacetime_worm import ReliableContinuumSpacetimeWorm
from .path_staging import autocorrelation_estimate


def _integer(x, name, least):
    if isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)) or x < least:
        raise ValueError(f"{name} must be integer >= {least}")
    return int(x)


def ratio_block_jackknife(numerator, denominator, *, scale=1.0, n_blocks=8):
    """Ratio estimator and deletion-of-time-block jackknife standard errors.

    Observations must be in time order. Adjacent observations are blocked
    together; *walkers or worm endpoints are never independent samples*.
    The jackknife quantifies finite-chain noise only when block lengths
    comfortably exceed the observable's autocorrelation time.
    """
    num = np.asarray(numerator, dtype=float)
    den = np.asarray(denominator, dtype=float)
    b = _integer(n_blocks, "n_blocks", 4)
    if (num.ndim != 3 or num.shape[0] < 8*b or den.shape != (num.shape[0],)
            or not np.all(np.isfinite(num)) or not np.all(np.isfinite(den))
            or np.any(den < 0) or not np.isfinite(scale) or scale <= 0):
        raise ValueError("need finite (time, separation, mode) numerator and nonnegative denominator, >=8 samples per block")
    if den.sum() <= 0:
        raise ValueError("zero total denominator")
    full = scale * num.sum(axis=0) / den.sum()
    indices = np.array_split(np.arange(len(den)), b)
    leave = []
    for part in indices:
        rest_den = den.sum()-den[part].sum()
        if rest_den <= 0:
            raise ValueError("one time block contains all denominator weight")
        leave.append(scale * (num.sum(axis=0)-num[part].sum(axis=0))/rest_den)
    loo = np.asarray(leave)
    stderr = np.sqrt((b-1)/b * np.sum((loo-loo.mean(axis=0))**2,axis=0))
    return full, stderr


def split_rhat(influence_chains):
    """Rank-free split-Rhat diagnostic on stationary ratio influence traces.

    This uses the conventional split-chain variance estimate, not rank
    normalization. A value near 1 is necessary but NOT sufficient for mixing.
    Only comparable, independently seeded chains may be supplied.
    """
    x = [np.asarray(chain, dtype=float) for chain in influence_chains]
    if len(x) < 2 or any(a.ndim != 1 or len(a) < 16 or not np.all(np.isfinite(a)) for a in x):
        raise ValueError("require >=2 finite independent chains with >=16 observations each")
    half = min(len(a) for a in x)//2
    parts = np.asarray([part for a in x for part in (a[:half],a[-half:])])
    within = np.var(parts,axis=1,ddof=1)
    W = float(np.mean(within))
    if W == 0:
        return 1.0 if np.all(parts == parts[0,0]) else float("inf")
    B = half*float(np.var(parts.mean(axis=1),ddof=1))
    return float(np.sqrt(((half-1)/half*W+B/half)/W))


@dataclass(frozen=True)
class AuditedWormResult:
    imaginary_times: np.ndarray
    indices: np.ndarray
    green: np.ndarray
    green_raw: np.ndarray
    jackknife_stderr: np.ndarray
    half_block_stderr: np.ndarray
    ratio_ess: np.ndarray
    closed_count: int
    open_counts: np.ndarray
    sector_ess: float
    conditional_ess: float
    acceptance: dict
    n_measurements: int
    influence: np.ndarray
    numerator_trace: np.ndarray
    denominator_trace: np.ndarray
    block_length: int


def audit_worm_chain(sampler: ReliableContinuumSpacetimeWorm, *, n_blocks=8):
    """Run an unmodified Patch-22 sampler and audit the correlated ratio.

    Return the complete influence-function traces for cross-replica Rhat.
    RB and raw estimates use exactly the same normalization as Patch 22.
    """
    if not isinstance(sampler,ReliableContinuumSpacetimeWorm):
        raise TypeError("sampler must be ReliableContinuumSpacetimeWorm")
    cfg = sampler.config
    M = cfg.n_links
    measurements = (cfg.n_steps-cfg.burn_in+cfg.thin-1)//cfg.thin
    _integer(n_blocks,"n_blocks",4)
    if measurements < 8*n_blocks:
        raise ValueError("not enough measurements for requested jackknife blocks")
    rb = np.zeros((measurements,M-1,len(sampler.indices)),float)
    raw = np.zeros_like(rb)
    den = np.zeros(measurements,float)
    den_raw = np.zeros(measurements,float)
    p_trace=np.zeros(measurements,float)
    open_counts=np.zeros(M-1,int)
    n_closed=0
    position=0
    for step in range(cfg.n_steps):
        sampler.step()
        if step<cfg.burn_in or (step-cfg.burn_in)%cfg.thin:
            continue
        pG=sampler.conditional_open_probability()
        if not np.isfinite(pG) or not 0<=pG<=1:
            raise FloatingPointError("invalid conditional worm probability")
        ell=sampler.length-1
        phase=np.cos(sampler.wavevectors @ (sampler.extra[-1]-sampler.extra[0]))
        rb[position,ell]=pG*phase
        den[position]=1-pG
        p_trace[position]=pG
        if sampler.open:
            raw[position,ell]=phase
            open_counts[ell]+=1
        else:
            den_raw[position]=1.
            n_closed+=1
        position+=1
    if position != measurements:
        raise RuntimeError("unexpected production measurement count")
    if n_closed<4 or np.sum(den)<=0:
        raise RuntimeError("insufficient closed-sector or conditional samples")
    scale=(M-1)/cfg.sector_fugacity
    estimate,stderr=ratio_block_jackknife(rb,den,scale=scale,n_blocks=n_blocks)
    _,coarse=ratio_block_jackknife(rb,den,scale=scale,n_blocks=4)
    raw_est=scale*raw.sum(axis=0)/den_raw.sum()
    # The ratio's influence function includes covariance with its denominator.
    influence=scale*rb-estimate[None,:,:]*den[:,None,None]
    ess=np.zeros_like(estimate)
    for it in range(M-1):
        for ik in range(len(sampler.indices)):
            ess[it,ik]=autocorrelation_estimate(influence[:,it,ik]).effective_samples
    return AuditedWormResult(cfg.time_step*np.arange(1,M),sampler.indices.copy(),
                              estimate,raw_est,stderr,coarse,ess,n_closed,open_counts,
                              autocorrelation_estimate(den_raw).effective_samples,
                              autocorrelation_estimate(p_trace).effective_samples,
                              {name:a/n if n else float('nan') for name,(a,n) in sampler._counts.items()},
                              measurements,influence,rb,den,measurements//n_blocks)


def replica_summary(values):
    """Mean and between-independent-chain SEM (NOT within-chain MC SEM)."""
    x=np.asarray(values,dtype=float)
    if x.ndim!=1 or len(x)<2 or not np.all(np.isfinite(x)):
        raise ValueError("need at least two finite independent replica means")
    return float(x.mean()),float(np.std(x,ddof=1)/np.sqrt(len(x)))
