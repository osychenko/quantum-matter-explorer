"""Patch 31: evidence-based, conservative readiness screening for educational presets.

This is an *engineering* decision aid, not a mathematical convergence proof.
It only evaluates the benchmark's beta=2.4, N=1->2 midpoint zero-mode
observable. A failed control cannot be overridden by a successful R-hat.
"""
from __future__ import annotations

from collections import defaultdict
from math import sqrt
import numpy as np

from .worm_trace_diagnostics import group_diagnostics, blocking_curve, autocorrelation_curve

DEFAULT_REFERENCE = 1.6910255956087066  # independent spectral K=2, finite basis


def _number(x, key):
    y = float(x[key])
    if not np.isfinite(y):
        raise ValueError(f"nonfinite {key}")
    return y


def _records(rows):
    groups = defaultdict(list)
    for row in rows:
        key = (int(row['M']), row['method'])
        if key[0] < 4 or key[0] % 2 or key[1] not in ('baseline', 'exchange'):
            raise ValueError('unsupported grid/method')
        _number(row, 'green')
        if (key[0], key[1], int(row['seed'])) in {(k[0], k[1], int(x['seed'])) for k, v in groups.items() for x in v}:
            raise ValueError('duplicate method/grid/seed')
        groups[key].append(row)
    return groups


def _trace_checks(traces):
    if not traces:
        return {'trace_evidence': 'missing', 'split_rhat': float('nan'),
                'trace_tail_warnings': 0, 'blocking_unresolved': True}
    diagnostic = group_diagnostics(traces)
    # Each chain's observed maximum *valid* contiguous block length is checked
    # against the autocorrelation time of the same chain, measured in samples.
    # A plateau is not inferred from one threshold alone.
    pooled = diagnostic['pooled_green']
    incomplete = 0
    windows = 0
    for t in traces:
        influence = t['scale']*t['numerator']-pooled*t['denominator']
        ac = autocorrelation_curve(influence)
        windows += int(ac['warning'] != 'none')
        blocks = blocking_curve(t)
        # Largest block, containing >=8 samples and at least 4 blocks.
        longest = max((b['block_length'] for b in blocks), default=0.)
        if longest < 10*max(ac['tau_int'], .5):
            incomplete += 1
    rh = max(diagnostic['split_rhat_influence'],
             diagnostic['split_rhat_numerator'], diagnostic['split_rhat_denominator'])
    return {'trace_evidence':'available', 'split_rhat':float(rh),
            'trace_tail_warnings':windows, 'blocking_unresolved':incomplete > 0,
            'blocking_unresolved_chains':incomplete,
            'median_tau_int':float(diagnostic['median_tau_int'])}


def assess_preset(rows, *, reference=DEFAULT_REFERENCE, traces=None,
                  min_replicas=8, max_z=2.0, max_rhat=1.05,
                  max_relative_sem=.05, min_ess=100):
    """Evaluate one M/method without pooling replicates with distinct targets.

    Distinguish `blocked` (resolved failure), `needs_evidence` (no trace files
    or too little precision) and `candidate` (all operational screens pass).
    A candidate is not a physics certification.
    """
    rows = list(rows)
    if not rows:
        raise ValueError('empty replica group')
    keys={(int(r['M']),r['method']) for r in rows}
    if len(keys)!=1:
        raise ValueError('cannot mix M or method')
    M,method=next(iter(keys))
    seeds=[int(r['seed']) for r in rows]
    if len(seeds)!=len(set(seeds)):
        raise ValueError('duplicate validation seed')
    values=np.asarray([_number(r,'green') for r in rows],float)
    if not np.isfinite(reference) or reference <= 0:
        raise ValueError('reference must be finite positive')
    mean=float(values.mean())
    sem=float(values.std(ddof=1)/sqrt(len(rows))) if len(rows)>=2 else float('nan')
    z=abs(mean-reference)/sem if np.isfinite(sem) and sem>0 else float('inf')
    ess=min(_number(r,'ratio_ess') for r in rows)
    rel=sem/abs(mean) if mean else float('inf')
    if traces:
        if {(int(t['M']),t['method']) for t in traces}!={(M,method)}:
            raise ValueError('trace method/grid mismatch')
        if {t['seed'] for t in traces}!={*seeds}:
            raise ValueError('missing/extra traces or wrong seeds')
        for r in rows:
            t=next(t for t in traces if t['seed']==int(r['seed']))
            if not np.isclose(float(r['green']),t['green'],atol=1e-9,rtol=1e-9):
                raise ValueError('CSV vs trace Green value mismatch')
    tc=_trace_checks(traces)
    flags=[]; warnings=[]
    if z>max_z: flags.append('reference_discrepancy')
    if ess < min_ess: flags.append('low_reported_ess')
    if tc['trace_evidence']=='available' and tc['split_rhat']>max_rhat:
        flags.append('split_rhat_high')
    if len(rows)<min_replicas: warnings.append('too_few_replicas')
    if rel > max_relative_sem: warnings.append('imprecise')
    if tc['trace_evidence']=='missing': warnings.append('traces_missing')
    else:
        if tc['trace_tail_warnings']: warnings.append('autocorrelation_window_warning')
        if tc['blocking_unresolved']: warnings.append('blocking_length_insufficient')
    status = 'blocked' if flags else ('needs_evidence' if warnings else 'candidate')
    return dict(M=M,method=method,replicas=len(rows),mean_green=mean,replica_sem=sem,
                reference=float(reference),z_reference=float(z),relative_sem=float(rel),
                min_reported_ess=ess,status=status,
                blocking_flags=';'.join(flags) if flags else 'none',
                evidence_warnings=';'.join(warnings) if warnings else 'none',**tc)


def compare_methods(results, *, max_disagreement_z=2.0):
    """Flag two *independently seeded* methods for same grid.

    The check assumes independent baselines and exchange validation chains.
    """
    by_m=defaultdict(dict)
    for row in results:
        by_m[int(row['M'])][row['method']]=row
    output=[]
    for M,methods in sorted(by_m.items()):
        if not {'baseline','exchange'}<=set(methods):
            continue
        a,b=methods['baseline'],methods['exchange']
        denom=sqrt(float(a['replica_sem'])**2+float(b['replica_sem'])**2)
        z=abs(float(a['mean_green'])-float(b['mean_green']))/denom if denom>0 else float('inf')
        output.append(dict(M=M,baseline=float(a['mean_green']),exchange=float(b['mean_green']),
                           disagreement_z=z,flag='method_disagreement' if z>max_disagreement_z else 'none'))
    return output


def apply_cross_checks(results, comparisons):
    """A cross-method inconsistency blocks *both* estimators on that grid."""
    bad={c['M'] for c in comparisons if c['flag']!='none'}
    for r in results:
        if r['M'] in bad:
            old=r['blocking_flags']
            r['blocking_flags']=(old+';' if old!='none' else '')+'method_disagreement'
            r['status']='blocked'
    return results


def chain_influence_report(rows, *, traces=None):
    """Chain-level leave-one-out sensitivity, plus optional split-half ratios.

    Half ratios are descriptive diagnostics, NOT independent uncertainty tests.
    """
    data=list(rows)
    if len(data)<3:
        raise ValueError('at least three chains required')
    means=np.array([_number(x,'green') for x in data])
    full=float(means.mean())
    lookup={int(t['seed']):t for t in traces or []}
    if traces and set(lookup)!={int(x['seed']) for x in data}:
        raise ValueError('trace seeds incomplete or duplicate')
    output=[]
    for i,r in enumerate(data):
        kept=np.delete(means,i)
        row=dict(M=int(r['M']),method=r['method'],seed=int(r['seed']),
                 green=float(means[i]),leave_one_out_mean=float(kept.mean()),
                 leave_one_out_shift=float(kept.mean()-full),
                 first_half_green=float('nan'),last_half_green=float('nan'),
                 second_minus_first=float('nan'))
        t=lookup.get(int(r['seed']))
        if t is not None:
            half=len(t['numerator'])//2
            first=t['numerator'][:half],t['denominator'][:half]
            second=t['numerator'][-half:],t['denominator'][-half:]
            def ratio(n,d):
                return float(t['scale']*np.sum(n)/np.sum(d)) if np.sum(d)>0 else float('nan')
            row['first_half_green']=ratio(*first)
            row['last_half_green']=ratio(*second)
            row['second_minus_first']=row['last_half_green']-row['first_half_green']
        output.append(row)
    return output
