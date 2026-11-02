"""Patch 26: frozen pilot selection and held-out validation for continuum worms.

No adaptive MCMC kernels are used during the measurement stage. The pilot
compares fixed, correctly reweighted Patch 24/25 proposal families. It chooses
one configuration; *new* seeds evaluate the chosen and comparator families.

Selection against a spectral reference is suitable for a known-answer
algorithm benchmark, NOT unbiased evidence about an unknown physical system.
The spectral K=1/K=2 spread is reported separately; short-chain diagnostics
remain heuristic and no threshold certifies convergence.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import time

import numpy as np

from .guided_worm import GuidedContinuumWorm, GuidedWormConfig
from .worm_audit import audit_worm_chain


@dataclass(frozen=True)
class ProposalCandidate:
    name: str
    guided_move_probability: float = 0.0
    guided_fraction: float = 0.0
    drift_strength: float = 0.9
    initial_tilt_strength: float = 1.0
    multiple_try_probability: float = 0.0
    multiple_try_candidates: int = 1

    def validate(self):
        if not isinstance(self.name, str) or not self.name or any(c in self.name for c in ',\n'):
            raise ValueError('candidate needs a nonempty CSV-safe name')
        # Delegate proposal-specific validation to the underlying exact sampler.
        GuidedWormConfig(guided_move_probability=self.guided_move_probability,
                         guided_fraction=self.guided_fraction,
                         drift_strength=self.drift_strength,
                         initial_tilt_strength=self.initial_tilt_strength,
                         multiple_try_probability=self.multiple_try_probability,
                         multiple_try_candidates=self.multiple_try_candidates).validate()
        if self.multiple_try_probability > 0 and self.guided_move_probability > 0:
            raise ValueError('use only one optional global move family per candidate')
        return self

    def configure(self, base: GuidedWormConfig, *, seed: int, steps: int):
        self.validate()
        if type(seed) is not int or seed < 0 or type(steps) is not int or steps < 160:
            raise ValueError('seed >=0 and steps >=160 required')
        result = replace(base, seed=seed, n_steps=steps,
                         burn_in=min(steps // 4, 400),
                         guided_move_probability=self.guided_move_probability,
                         guided_fraction=self.guided_fraction,
                         drift_strength=self.drift_strength,
                         initial_tilt_strength=self.initial_tilt_strength,
                         multiple_try_probability=self.multiple_try_probability,
                         multiple_try_candidates=self.multiple_try_candidates)
        return result.validate()


def candidate_set(*, quick: bool = False):
    candidates = (
        ProposalCandidate('baseline'),
        ProposalCandidate('guided_moderate', .70, .50, .90, 1.0),
        ProposalCandidate('multiple_try_4', multiple_try_probability=.80,
                          multiple_try_candidates=4),
    )
    if not quick:
        candidates += (ProposalCandidate('guided_strong', .85, .80, 1.25, 1.5),)
    return candidates


def separated_seed(*, stage: str, model: int, links: int, candidate: int, replica: int):
    """Disjoint seed namespaces for selection and validation; no warm starts."""
    if stage not in ('pilot', 'validation'):
        raise ValueError('stage must be pilot or validation')
    for name, value in (('model', model), ('links', links), ('candidate', candidate), ('replica', replica)):
        if type(value) is not int or not 0 <= value < 100:
            raise ValueError(f'{name} must be an integer in [0,100)')
    return (1_000_000 if stage == 'pilot' else 7_000_000) + model*1_000_000 + links*10_000 + candidate*100 + replica


def run_candidate(cell, potential, base_config, candidate, *, seed, steps,
                  spectral_reference, spectral_basis_spread=0.0, n_blocks=4):
    """Run exactly one fresh chain. One target observable: G_(k=0)(beta/2)."""
    cfg = candidate.configure(base_config, seed=seed, steps=steps)
    if cfg.n_links % 2:
        raise ValueError('midpoint requires an even number of links')
    if not np.isfinite(spectral_reference) or spectral_reference <= 0:
        raise ValueError('spectral reference must be finite and positive')
    if not np.isfinite(spectral_basis_spread) or spectral_basis_spread < 0:
        raise ValueError('spectral spread must be nonnegative')
    model = GuidedContinuumWorm(cell, potential, 1, cfg,
                                indices=((0, 0, 0), (1, 0, 0)))
    t0 = time.process_time()
    audit = audit_worm_chain(model, n_blocks=n_blocks)
    seconds = max(time.process_time()-t0, np.finfo(float).tiny)
    index = cfg.n_links//2-1
    estimate = float(audit.green[index, 0])
    error = float(audit.jackknife_stderr[index, 0])
    half_error = float(audit.half_block_stderr[index, 0])
    ess = float(audit.ratio_ess[index, 0])
    return dict(candidate=candidate.name, seed=seed, steps=steps,
                beta=cfg.beta, M=cfg.n_links, tau=cfg.time_step,
                green=estimate, jackknife_sem=error, halfblock_sem=half_error,
                ratio_ess=ess, cpu_seconds=seconds, ess_per_cpu=ess/seconds,
                spectral_reference=float(spectral_reference),
                spectral_basis_spread=float(spectral_basis_spread),
                raw_green=float(audit.green_raw[index, 0]),
                closed_count=int(audit.closed_count),
                open_count=int(audit.open_counts[index]),
                global_refresh_acceptance=float(audit.acceptance.get('global_refresh',float('nan'))),
                guided_refresh_acceptance=float(audit.acceptance.get('guided_refresh',float('nan'))),
                multiple_try_acceptance=float(audit.acceptance.get('multiple_try',float('nan'))))


def summarize_runs(rows, *, reference: float, basis_spread: float, min_ess: float = 5.0):
    """Pilot ranking score and conservative quality flags for independent replicas.

    Score is diagnostic MSE proxy *relative CPU cost*:
        (squared reference discrepancy + between-chain variance of mean
         + mean within-chain block variance + basis_spread**2) * mean CPU.
    A candidate with any low-ESS, unstable blocking, or nonfinite estimate is
    marked ineligible. This heuristic is fixed BEFORE the held-out phase.
    """
    data = list(rows)
    if len(data) < 2 or any(type(x) is not dict for x in data):
        raise ValueError('at least two independent replica dictionaries required')
    if not np.isfinite(reference) or not np.isfinite(basis_spread) or basis_spread < 0:
        raise ValueError('invalid references')
    if not np.isfinite(min_ess) or min_ess <= 0:
        raise ValueError('min_ess must be positive')
    fields = ('green','jackknife_sem','halfblock_sem','ratio_ess','cpu_seconds')
    x = np.asarray([[row[k] for k in fields] for row in data], dtype=float)
    if not np.all(np.isfinite(x)) or np.any(x[:,1:3] < 0) or np.any(x[:,3:] <= 0):
        raise ValueError('replicas have invalid statistics')
    if len({r['seed'] for r in data}) != len(data):
        raise ValueError('replicas must have distinct seeds')
    if len({r['candidate'] for r in data}) != 1:
        raise ValueError('replica candidates must match')
    estimate = float(np.mean(x[:,0]))
    between_sem = float(np.std(x[:,0],ddof=1)/np.sqrt(len(data)))
    within = float(np.mean(x[:,1]**2))
    cost = float(np.mean(x[:,4]))
    penalty = (estimate-reference)**2 + between_sem**2 + within + basis_spread**2
    score = float(penalty*cost)
    flags=[]
    if np.min(x[:,3]) < min_ess:
        flags.append('low_ratio_ess')
    # A different blocking scale must not reveal >3x worse errors.
    if np.any(x[:,2] > 3*np.maximum(x[:,1],1e-12)):
        flags.append('blocking_unstable')
    if abs(estimate-reference) > 2*max(between_sem,float(np.sqrt(within/len(data))),1e-12) + basis_spread:
        flags.append('reference_discrepancy')
    return dict(candidate=data[0]['candidate'], replicas=len(data), green_mean=estimate,
                replica_sem=between_sem, mean_jackknife_sem=float(np.mean(x[:,1])),
                min_ratio_ess=float(np.min(x[:,3])), mean_cpu_seconds=cost,
                mean_ess_per_cpu=float(np.mean(x[:,3]/x[:,4])),
                spectral_reference=float(reference), spectral_basis_spread=float(basis_spread),
                pilot_score=score, eligible=('low_ratio_ess' not in flags and 'blocking_unstable' not in flags),
                diagnostics=(';'.join(flags) if flags else 'no_flag'))


def choose_candidate(pilot_summaries, *, baseline='baseline'):
    """Freeze exactly one candidate using PILOT results only; fallback baseline."""
    records = list(pilot_summaries)
    if not records or len({r['candidate'] for r in records}) != len(records):
        raise ValueError('nonempty summaries with distinct candidates required')
    if baseline not in {r['candidate'] for r in records}:
        raise ValueError('baseline missing')
    eligible = [r for r in records if r['eligible'] and np.isfinite(r['pilot_score'])]
    if not eligible:
        return baseline, 'fallback_no_eligible_candidate'
    chosen = min(eligible, key=lambda r:(r['pilot_score'], r['candidate']))
    return chosen['candidate'], 'pilot_minimum_screened_mse_per_cpu'
