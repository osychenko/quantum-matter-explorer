"""Patch 28: pilot-calibrated fixed-step CPU comparisons, with held-out chains.

CPU calibration and validation are separate; no time-dependent stopping of a
measurement chain. Achieved CPU time is reported, never assumed equal.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
import time
import numpy as np
from .reliable_spacetime_worm import ReliableContinuumSpacetimeWorm, ReliableSpacetimeConfig
from .replica_exchange_worm import ReplicaExchangeContinuumWorm, ReplicaExchangeConfig
from .worm_audit import audit_worm_chain

@dataclass(frozen=True)
class BudgetPlan:
    method: str
    cpu_target: float
    pilot_steps: int
    pilot_seconds: float
    validation_steps: int
    seed: int
    calibration_factor: float = 1.0

def make_sampler(method, cell, potential, cfg):
    if method == 'baseline':
        return ReliableContinuumSpacetimeWorm(cell, potential, 1, cfg, indices=((0,0,0),))
    if method == 'exchange':
        return ReplicaExchangeContinuumWorm(cell, potential, 1, replace(ReplicaExchangeConfig(),**vars(cfg)), indices=((0,0,0),))
    raise ValueError('method must be baseline or exchange')

def plan_budget(method, cell, potential, base, *, target_seconds, pilot_steps=30, seed=1120,
                min_validation_steps=160, max_validation_steps=200000, calibration_factor=1.25):
    """Estimate fixed n_steps from a separate pilot; warmup overhead included.

    `target_seconds` is a soft target; wall clock and CPU share resources and
    scheduler variation. The pilot is discarded and uses disjoint seed space.
    """
    if method not in ('baseline','exchange'):
        raise ValueError('unknown method')
    if not np.isfinite(target_seconds) or target_seconds <= 0:
        raise ValueError('target_seconds must be positive finite')
    if type(pilot_steps) is not int or pilot_steps < 8:
        raise ValueError('pilot_steps must be integer >=8')
    if not np.isfinite(calibration_factor) or calibration_factor <= 0:
        raise ValueError("calibration_factor must be finite and positive")
    if type(seed) is not int or seed < 0:
        raise ValueError('seed must be nonnegative integer')
    # For realistic pilots, time the same observable/diagnostic workload as
    # validation. Short legacy pilots remain supported for smoke tests.
    realistic = pilot_steps >= 160
    cfg = replace(base, seed=seed,
                  n_steps=pilot_steps if realistic else max(200,pilot_steps+20),
                  burn_in=min(pilot_steps//4, 400) if realistic else 10,
                  thin=2)
    sampler = make_sampler(method,cell,potential,cfg)
    t=time.process_time()
    if realistic:
        audit_worm_chain(sampler, n_blocks=4)
    else:
        for _ in range(pilot_steps):
            sampler.step()
    dt=time.process_time()-t
    if not np.isfinite(dt) or dt <= 0:
        raise RuntimeError('invalid measured CPU time')
    n=int(round(target_seconds*pilot_steps/(dt*calibration_factor)))
    n=max(min_validation_steps,min(max_validation_steps,n))
    return BudgetPlan(method,float(target_seconds),pilot_steps,float(dt),n,seed,float(calibration_factor))

def run_validation(method,cell,potential,base,plan,*, seed, n_blocks=8):
    """One new fixed-length chain. Returns observable and CPU diagnostics."""
    if method != plan.method or seed == plan.seed or type(seed) is not int or seed < 0:
        raise ValueError('validation method/seed must differ from calibration')
    n=plan.validation_steps
    burn=min(n//4,400)
    cfg=replace(base,seed=seed,n_steps=n,burn_in=burn,thin=2)
    sampler=make_sampler(method,cell,potential,cfg)
    start=time.process_time()
    measurements=(n-burn+cfg.thin-1)//cfg.thin
    used_blocks=min(n_blocks,max(4,measurements//8))
    audit=audit_worm_chain(sampler,n_blocks=used_blocks)
    dt=max(time.process_time()-start,np.finfo(float).tiny)
    midpoint=cfg.n_links//2-1
    g=float(audit.green[midpoint,0]); sem=float(audit.jackknife_stderr[midpoint,0])
    ess=float(audit.ratio_ess[midpoint,0]); raw=float(audit.green_raw[midpoint,0])
    influence=audit.influence[:,midpoint,0]
    # 4, 8 and (when enough measurements) 16 time blocks provide genuinely
    # distinct blocking scales, without re-running the chain.
    from .worm_audit import ratio_block_jackknife
    from .worm_audit import _integer
    errors = {}
    for blocks in (4, 8, 16):
        if audit.n_measurements >= 8*blocks:
            _, e = ratio_block_jackknife(audit.numerator_trace,
                                         audit.denominator_trace,
                                         scale=(cfg.n_links-1)/cfg.sector_fugacity,
                                         n_blocks=blocks)
            errors[blocks] = float(e[midpoint,0])
    positive = [v for v in errors.values() if v > 1e-12]
    spread = (max(positive)/min(positive)) if len(positive) >= 2 else float('nan')
    return dict(method=method,seed=seed,steps=n,cpu_seconds=dt,goal_cpu_seconds=plan.cpu_target,
                cpu_ratio=dt/plan.cpu_target,green=g,jackknife_sem=sem,raw_green=raw,
                ratio_ess=ess,ess_per_cpu=ess/dt,
                max_abs_influence_fraction=float(np.max(np.abs(influence))/max(np.abs(influence).sum(),np.finfo(float).tiny)),
                closed=audit.closed_count,open=int(audit.open_counts[midpoint]),
                block_stability=float(spread),
                block_sem_4=errors.get(4,float('nan')),
                block_sem_8=errors.get(8,float('nan')),
                block_sem_16=errors.get(16,float('nan')),
                block_scales=','.join(map(str,sorted(errors))))

def summarize_validation(rows, reference, *, min_ess=10, cpu_tolerance=0.35):
    """No binary convergence certificate: explicit flags and replicate discrepancy."""
    xs=list(rows)
    if len(xs)<2 or len(set(r['seed'] for r in xs))!=len(xs):
        raise ValueError('require at least two independently seeded chains')
    if len(set(r['method'] for r in xs))!=1 or not np.isfinite(reference):
        raise ValueError('inconsistent methods or reference')
    means=np.array([r['green'] for r in xs],float)
    sem=float(np.std(means,ddof=1)/np.sqrt(len(xs)))
    flags=[]
    if min(r['ratio_ess'] for r in xs)<min_ess: flags.append('low_ess')
    if max(r['max_abs_influence_fraction'] for r in xs)>0.1: flags.append('rare_contribution')
    if any(not np.isfinite(r['block_stability']) or r['block_stability']>3 for r in xs): flags.append('blocking_unstable')
    budget_mismatch=max(abs(r['cpu_ratio']-1) for r in xs)>cpu_tolerance
    if abs(means.mean()-reference)>2*max(sem,1e-12): flags.append('reference_discrepancy')
    return dict(method=xs[0]['method'],replicas=len(xs),mean_green=float(means.mean()),replica_sem=sem,
                spectral_reference=float(reference),reference_difference=float(means.mean()-reference),
                mean_cpu_seconds=float(np.mean([r['cpu_seconds'] for r in xs])),
                mean_ess_per_cpu=float(np.mean([r['ess_per_cpu'] for r in xs])),
                min_ess=float(min(r['ratio_ess'] for r in xs)),status='inconclusive' if flags else 'screen_pass_not_proof',
                flags=';'.join(flags) if flags else 'no_flags',
                cpu_status='budget_mismatch' if budget_mismatch else 'within_tolerance',
                cpu_flags='cpu_budget_mismatch' if budget_mismatch else 'no_flags')
