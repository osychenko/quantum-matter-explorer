"""Patch 26: train-on-pilot, freeze, then validate proposal mixtures.

    python3 -m examples.bosons.run_adaptive_worm --quick
    python3 -m examples.bosons.run_adaptive_worm
    python3 -m examples.bosons.run_adaptive_worm --links 4 6 --pilot-steps 2000 \\
        --validation-steps 4000 --pilot-replicas 4 --validation-replicas 6

Known-reference algorithm benchmarks ONLY. No on-the-fly adaptation during
measurement; all methods target the same finite-tau interacting Green function.
"""
from __future__ import annotations

import argparse
import csv
import json
from contextlib import nullcontext
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from qmc.adaptive_worm import (candidate_set, choose_candidate, run_candidate,
                               separated_seed, summarize_runs)
from qmc.bosons import SoftCorePairPotential
from qmc.cells import CubicCell
from qmc.continuum_spacetime_worm import exact_free_addition_green
from qmc.green_spectral import TwoBosonAdditionReference
from qmc.guided_worm import GuidedWormConfig


def _csv(path, rows):
    if not rows:
        raise ValueError('cannot write empty CSV')
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        # A failed chain has fewer fields than a completed chain.  Preserve
        # every diagnostic column irrespective of which row came first.
        columns=list(dict.fromkeys(key for row in rows for key in row))
        writer=csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def spectral_reference(cell, height, beta, D=0.5):
    if height == 0:
        mode=((0,0,0),)
        exact=float(exact_free_addition_green(cell,1,beta,np.array([beta/2]),mode,D)[0,0])
        return exact,exact
    try:
        from threadpoolctl import threadpool_limits
        limits=threadpool_limits(limits=1)
    except ImportError:
        limits=nullcontext()
    with limits:
        values=[]
        for k in (1,2):
            ref=TwoBosonAdditionReference(cell_length=cell.length,
                    kinetic_coefficient=D,height=height,cutoff=1.2,
                    k_max=k,n_grid=16)
            values.append(float(ref.measure(beta,[beta/2],((0,0,0),)).green[0,0]))
    return tuple(values)


def _screen_failure(candidate, count, *, reason):
    return dict(candidate=candidate,replicas=count,green_mean=float('nan'),
        replica_sem=float('nan'),mean_jackknife_sem=float('nan'),
        min_ratio_ess=0.,mean_cpu_seconds=float('nan'),
        mean_ess_per_cpu=float('nan'),spectral_reference=float('nan'),
        spectral_basis_spread=float('nan'),pilot_score=float('inf'),
        eligible=False,diagnostics=reason)


def run_experiment(output_dir='outputs/bosons/adaptive_worm', *, quick=False,
                   links=None,pilot_steps=None,validation_steps=None,
                   pilot_replicas=None,validation_replicas=None,
                   min_ess=None, include_ideal=False):
    out=Path(output_dir)
    out.mkdir(parents=True,exist_ok=True)
    grids=(4,) if quick else (4,6)
    if links is not None:
        if not isinstance(links,(tuple,list)) or not links or any(type(m) is not int or m<4 or m%2 or m>=100 for m in links):
            raise ValueError('links require even integers in [4,100)')
        grids=tuple(links)
    ps=(400 if quick else 1200) if pilot_steps is None else pilot_steps
    vs=(600 if quick else 2400) if validation_steps is None else validation_steps
    pr=(2 if quick else 3) if pilot_replicas is None else pilot_replicas
    vr=(2 if quick else 4) if validation_replicas is None else validation_replicas
    threshold=(3.0 if quick else 7.0) if min_ess is None else min_ess
    if (any(type(v) is not int for v in (ps,vs,pr,vr)) or min(ps,vs)<160
        or pr<2 or vr<2 or not np.isfinite(threshold) or threshold<=0):
        raise ValueError('steps>=160, replicas>=2 and positive min_ess required')
    n_blocks=4 if quick else 8
    for steps in (ps,vs):
        production=(steps-min(steps//4,400)+2)//3
        if production<8*n_blocks:
            raise ValueError('not enough production samples for block jackknife')
    beta,D=2.4,0.5
    cell=CubicCell(2.8)
    cases=(('ideal',0.),('repulsive',3.)) if include_ideal else (('repulsive',3.),)
    candidates=candidate_set(quick=quick)
    cfg=GuidedWormConfig(n_links=4,time_step=beta/4,n_steps=vs,
             burn_in=vs//4,thin=3,seed=2026,sector_fugacity=.9,
             toggle_probability=.42,global_refresh_probability=.30,
             ghost_refresh_probability=.6,time_move_probability=.45,
             endpoint_move_probability=.20,exchange_bridge_probability=.25,
             max_drift=.4)
    pilot_runs=[]; pilot_summary=[]; choices=[]; validation_runs=[]
    validation_summary=[]; spectral=[]
    for model_index,(case,height) in enumerate(cases):
        k1,k2=spectral_reference(cell,height,beta,D)
        for M in grids:
            spread=abs(k2-k1)
            spectral.append(dict(case=case,M=M,beta=beta,
                           spectral_K1=k1,spectral_K2=k2,basis_difference=spread))
            pot=SoftCorePairPotential(cell,height=height,cutoff=1.2)
            config=replace(cfg,n_links=M,time_step=beta/M)
            # Phase one: pilot. Never reuse a pilot path, RNG, or seed later.
            this_pilot=[]
            for index,candidate in enumerate(candidates):
                successful=[]
                for ir in range(pr):
                    seed=separated_seed(stage='pilot',model=model_index,links=M,
                                        candidate=index,replica=ir)
                    try:
                        entry=run_candidate(cell,pot,config,candidate,seed=seed,
                                steps=ps,spectral_reference=k2,
                                spectral_basis_spread=spread,n_blocks=n_blocks)
                        entry.update(case=case,stage='pilot',replica=ir,status='ok',error='')
                        successful.append(entry)
                    except (RuntimeError,FloatingPointError,ValueError) as exc:
                        entry=dict(candidate=candidate.name,case=case,stage='pilot',replica=ir,
                                   seed=seed,status='failed',error=str(exc),M=M,
                                   green=float('nan'),ratio_ess=0.,cpu_seconds=float('nan'),
                                   spectral_reference=k2)
                    pilot_runs.append(entry)
                if len(successful)==pr:
                    stat=summarize_runs(successful,reference=k2,
                            basis_spread=spread,min_ess=threshold)
                else:
                    stat=_screen_failure(candidate.name,len(successful),reason='failed_pilot_chain')
                stat.update(case=case,M=M,stage='pilot')
                this_pilot.append(stat)
                pilot_summary.append(stat)
            name,reason=choose_candidate(this_pilot)
            choices.append(dict(case=case,M=M,selected=name,selection_reason=reason,
                 pilot_steps=ps,pilot_replicas=pr,validation_steps=vs,
                 validation_replicas=vr,min_ratio_ess=threshold,
                 spectral_K2=k2,spectral_K1=k1))
            print(f'{case}, M={M}: pilot selected {name} ({reason})')
            # Phase two: frozen choices, all arms measured on held-out RNGs.
            # The selection is NOT reevaluated with held-out estimates.
            for index,candidate in enumerate(candidates):
                successes=[]
                for ir in range(vr):
                    seed=separated_seed(stage='validation',model=model_index,links=M,
                                        candidate=index,replica=ir)
                    try:
                        entry=run_candidate(cell,pot,config,candidate,seed=seed,
                               steps=vs,spectral_reference=k2,
                               spectral_basis_spread=spread,n_blocks=n_blocks)
                        entry.update(case=case,stage='validation',replica=ir,status='ok',error='',
                                     pilot_selected=name,selected=(candidate.name==name))
                        successes.append(entry)
                    except (RuntimeError,FloatingPointError,ValueError) as exc:
                        entry=dict(candidate=candidate.name,case=case,stage='validation',
                                   replica=ir,seed=seed,status='failed',error=str(exc),
                                   M=M,green=float('nan'),ratio_ess=0.,
                                   cpu_seconds=float('nan'),spectral_reference=k2,
                                   pilot_selected=name,selected=(candidate.name==name))
                    validation_runs.append(entry)
                if len(successes)==vr:
                    stat=summarize_runs(successes,reference=k2,basis_spread=spread,
                                        min_ess=threshold)
                else:
                    stat=_screen_failure(candidate.name,len(successes),reason='failed_validation_chain')
                stat.update(case=case,M=M,stage='validation',pilot_selected=name,
                            selected=(candidate.name==name),reference_error=(stat['green_mean']-k2))
                validation_summary.append(stat)
                if np.isfinite(stat['green_mean']):
                    print(f'  held-out {candidate.name:17s} G0={stat["green_mean"]:.3f}'
                          f' +/- {stat["replica_sem"]:.3f} ESS/CPU={stat["mean_ess_per_cpu"]:.1f}'
                          f' selected={candidate.name==name}')
    _csv(out/'pilot_runs.csv',pilot_runs)
    _csv(out/'pilot_summary.csv',pilot_summary)
    _csv(out/'selected.csv',choices)
    _csv(out/'validation_runs.csv',validation_runs)
    _csv(out/'validation_summary.csv',validation_summary)
    _csv(out/'spectral_reference.csv',spectral)
    with (out/'protocol.json').open('w',encoding='utf-8') as f:
        json.dump(dict(patch=26,model='canonical N=1 -> 2 continuum soft-core insertion',
              beta=beta,D=D,cell_length=cell.length, cases=cases, links=grids,
              pilot_steps=ps,validation_steps=vs,pilot_replicas=pr,
              validation_replicas=vr,minimum_ratio_ess=threshold,
              candidates=[asdict(c) for c in candidates],
              base_config=asdict(cfg), selected=choices,
              scoring='pilot (squared spectral discrepancy + replica SEM^2 + mean jackknife variance + Fourier basis spread^2)*mean process CPU time; eligibility ESS and blocking',
              methodology='selection uses pilot only; fresh held-out seeds for all proposals; no online chain tuning',
              limitations='inconclusive short chains; reference selection unsuitable for unknown systems; CPU hardware dependent; primitive tau bias; finite K spectral systematic; do not certify convergence'),f,indent=2)
    fig,ax=plt.subplots(figsize=(7.5,4.5))
    for case,_ in cases:
        ref=next(x['spectral_K2'] for x in spectral if x['case']==case)
        ax.axhline(ref,ls='--',alpha=.45,label=f'{case} spectral K2')
        for c in candidates:
            rows=sorted((r for r in validation_summary if r['case']==case and r['candidate']==c.name),key=lambda r:r['M'])
            ok=[r for r in rows if np.isfinite(r['green_mean'])]
            if ok:
                ax.errorbar([r['M'] for r in ok],[r['green_mean'] for r in ok],
                    yerr=[r['replica_sem'] for r in ok],marker='o',capsize=2,
                    label=f'{case}: {c.name}')
    ax.set(xlabel='Number of imaginary-time links M',ylabel=r'Held-out $G_{k=0}(\beta/2)$',
           title='Pilot-selected proposals: untouched held-out accuracy')
    ax.grid(alpha=.2);ax.legend(fontsize=7);fig.tight_layout()
    fig.savefig(out/'validation_accuracy.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.3,4.4))
    for case,_ in cases:
        for c in candidates:
            rows=sorted((r for r in validation_summary if r['case']==case and r['candidate']==c.name),key=lambda r:r['M'])
            ok=[r for r in rows if np.isfinite(r['mean_ess_per_cpu'])]
            if ok:
                ax.plot([r['M'] for r in ok],[r['mean_ess_per_cpu'] for r in ok],
                    marker='o',label=f'{case}: {c.name}')
    ax.set(xlabel='Imaginary-time links M',ylabel='Held-out ratio ESS / process CPU second',
           title='Pilot choices do not guarantee held-out speedups')
    ax.grid(alpha=.2);ax.legend(fontsize=7);fig.tight_layout()
    fig.savefig(out/'validation_efficiency.png',dpi=160);plt.close(fig)
    print('Saved Patch 26 pilot/validation audit to',out)
    return choices,validation_summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick',action='store_true')
    parser.add_argument('--output',default='outputs/bosons/adaptive_worm')
    parser.add_argument('--links',nargs='+',type=int)
    parser.add_argument('--pilot-steps',type=int)
    parser.add_argument('--validation-steps',type=int)
    parser.add_argument('--pilot-replicas',type=int)
    parser.add_argument('--validation-replicas',type=int)
    parser.add_argument('--min-ess',type=float)
    parser.add_argument('--include-ideal',action='store_true')
    args=parser.parse_args()
    run_experiment(args.output,quick=args.quick,links=args.links,
                   pilot_steps=args.pilot_steps,validation_steps=args.validation_steps,
                   pilot_replicas=args.pilot_replicas,
                   validation_replicas=args.validation_replicas,
                   min_ess=args.min_ess,include_ideal=args.include_ideal)


if __name__=='__main__':
    main()
