"""Patch 27: fixed, physically identical baseline vs interaction-replica exchange.

    python3 -m examples.bosons.run_replica_exchange_worm --quick
    python3 -m examples.bosons.run_replica_exchange_worm
    python3 -m examples.bosons.run_replica_exchange_worm --links 4 6 --steps 5000 --replicas 6

No proposal adaptation or spectral-reference-based tuning occurs during a run.
All CPU timings include both physical and auxiliary replica work. Only the
physical interaction-strength chain contributes to the Green estimator.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from threadpoolctl import threadpool_limits

from qmc.bosons import SoftCorePairPotential
from qmc.cells import CubicCell
from qmc.green_spectral import TwoBosonAdditionReference
from qmc.reliable_spacetime_worm import ReliableContinuumSpacetimeWorm
from qmc.replica_exchange_worm import ReplicaExchangeConfig, ReplicaExchangeContinuumWorm
from qmc.worm_audit import audit_worm_chain


def _write_csv(path, rows):
    if not rows:
        raise ValueError('cannot write an empty table')
    with Path(path).open('w', encoding='utf-8', newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def run_experiment(output_dir='outputs/bosons/replica_exchange_worm', *,
                   quick=False, steps=None, replicas=None, links=None,
                   hot_fraction=0.25, height=3.0):
    output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
    if links is None:
        links=(4,) if quick else (4,6)
    links=tuple(links)
    if not links or any(type(v) is not int or v<4 or v%2 for v in links):
        raise ValueError('links must be positive even integers >=4')
    if steps is None:
        steps=600 if quick else 2200
    if type(steps) is not int or steps<320:
        raise ValueError('steps must be >=320')
    if replicas is None:
        replicas=2 if quick else 3
    if type(replicas) is not int or replicas<2:
        raise ValueError('replicas must be >=2')
    if not np.isfinite(height) or height<0:
        raise ValueError('height must be nonnegative and finite')
    cell=CubicCell(2.8);beta=2.4;D=.5;cutoff=1.2
    potential=SoftCorePairPotential(cell, height=height, cutoff=cutoff)
    base=ReplicaExchangeConfig(n_steps=steps,burn_in=min(steps//4,400),thin=3,
           sector_fugacity=.9,toggle_probability=.42,
           global_refresh_probability=.30,ghost_refresh_probability=.55,
           hot_interaction_fraction=hot_fraction,swap_probability=.35)
    references=[];rows=[];summary=[]
    with threadpool_limits(limits=1, user_api='blas'):
        for M in links:
            ref_values=[]
            for K in (1,2):
                reference=TwoBosonAdditionReference(
                    cell_length=cell.length,kinetic_coefficient=D,height=height,
                    cutoff=cutoff,k_max=K,n_grid=12 if quick else 16)
                value=float(reference.measure(beta,[beta/2],((0,0,0),)).green[0,0])
                references.append(dict(n_links=M,tau=beta/M,beta=beta,k_max=K,
                                       n_grid=12 if quick else 16,green_reference=value))
                ref_values.append(value)
            best_ref,spread=ref_values[-1],abs(ref_values[-1]-ref_values[0])
            for kind in ('baseline','replica_exchange'):
                group=[]
                for ir in range(replicas):
                    seed=202627+M*100000+ir*100+(0 if kind=='baseline' else 10000)
                    cfg=replace(base,seed=seed,n_links=M,time_step=beta/M)
                    if kind=='baseline':
                        sampler=ReliableContinuumSpacetimeWorm(cell,potential,1,cfg,
                                            kinetic_coefficient=D,indices=((0,0,0),(1,0,0)))
                    else:
                        sampler=ReplicaExchangeContinuumWorm(cell,potential,1,cfg,
                                            kinetic_coefficient=D,indices=((0,0,0),(1,0,0)))
                    started=time.process_time()
                    audit=audit_worm_chain(sampler,n_blocks=4)
                    seconds=max(time.process_time()-started,np.finfo(float).tiny)
                    midpoint=M//2-1
                    item=dict(sampler=kind,n_links=M,tau=beta/M,beta=beta,
                        height=height,replica=ir,seed=seed,n_steps=steps,
                        green=float(audit.green[midpoint,0]),
                        jackknife_sem=float(audit.jackknife_stderr[midpoint,0]),
                        ratio_ess=float(audit.ratio_ess[midpoint,0]),
                        cpu_seconds=seconds,ess_per_cpu=float(audit.ratio_ess[midpoint,0]/seconds),
                        spectral_reference=best_ref,basis_spread=spread,
                        closed_count=int(audit.closed_count),
                        open_count=int(audit.open_counts[midpoint]),
                        global_acceptance=float(audit.acceptance.get('global_refresh',float('nan'))),
                        exchange_acceptance=(float(audit.acceptance['exchange'])
                             if kind=='replica_exchange' else ''),
                        exchange_open_acceptance=(float(sampler.open_swap_acceptance)
                             if kind=='replica_exchange' else ''),
                        exchange_attempts=(int(sampler._counts['exchange'][1])
                             if kind=='replica_exchange' else 0))
                    rows.append(item);group.append(item)
                    print(f'{kind:16s} M={M} replica={ir} G0={item["green"]:.3f} '
                          f'ESS/cpu={item["ess_per_cpu"]:.1f}, ref={best_ref:.3f}')
                v=np.asarray([x['green'] for x in group])
                mean=float(v.mean());sem=float(v.std(ddof=1)/np.sqrt(replicas))
                avg_cpu=float(np.mean([x['cpu_seconds'] for x in group]))
                avg_esscpu=float(np.mean([x['ess_per_cpu'] for x in group]))
                flags=[]
                if min(x['ratio_ess'] for x in group)<10:
                    flags.append('low_ess')
                if abs(mean-best_ref)>2*max(sem,1e-12)+spread:
                    flags.append('reference_discrepancy')
                summary.append(dict(sampler=kind,n_links=M,tau=beta/M,beta=beta,
                      height=height,replicas=replicas,green_mean=mean,replica_sem=sem,
                      spectral_reference=best_ref,basis_spread=spread,
                      reference_difference=mean-best_ref,
                      mean_cpu_seconds=avg_cpu,mean_ess_per_cpu=avg_esscpu,
                      min_ratio_ess=min(x['ratio_ess'] for x in group),
                      status='inconclusive' if flags or quick else 'requires_long_chain_validation',
                      warnings=';'.join(flags) if flags else 'none_detected'))
    _write_csv(output/'replicas.csv',rows)
    _write_csv(output/'summary.csv',summary)
    _write_csv(output/'spectral_reference.csv',references)
    protocol=dict(model='N=1 background, N+1=2 periodic primitive-action boson worm',
                  beta=beta,cell_length=cell.length,height=height,cutoff=cutoff,D=D,
                  links=links,steps=steps,replicas=replicas,quick=quick,
                  config=asdict(base),
                  note=('Frozen two-replica interaction tempering; cold physical estimator only. '
                        'Both replica CPU costs included. No tuning on test results. '
                        'Short chains and naive ESS diagnostics cannot certify convergence.'))
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    fig,ax=plt.subplots(figsize=(7.0,4.4))
    for kind,title in [('baseline','Baseline'),('replica_exchange','Replica exchange')]:
        pts=sorted([r for r in summary if r['sampler']==kind],key=lambda r:r['tau'])
        ax.errorbar([r['tau'] for r in pts],[r['green_mean'] for r in pts],
                    yerr=[r['replica_sem'] for r in pts],label=title,marker='o',capsize=3)
    ax.axhline(references[-1]['green_reference'],color='gray',ls='--',label='Spectral K=2')
    ax.set(xlabel='Primitive time step',ylabel=r'$G_{k=0}(\beta/2)$',
           title='Physical Green-function agreement (independent replicas)')
    ax.legend();ax.grid(alpha=.25);fig.tight_layout()
    fig.savefig(output/'green_accuracy.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.0,4.4))
    xs=sorted(set(r['n_links'] for r in summary));x=np.arange(len(xs))
    for i,(kind,title) in enumerate([('baseline','Baseline'),('replica_exchange','Replica exchange')]):
        ys=[next(r['mean_ess_per_cpu'] for r in summary if r['n_links']==M and r['sampler']==kind) for M in xs]
        ax.bar(x+(i-.5)*.35,ys,width=.35,label=title)
    ax.set_xticks(x,[f'M={M}' for M in xs]);ax.set(ylabel='Ratio ESS / process CPU second',
           title='CPU-adjusted efficiency (both chains charged)')
    ax.legend();ax.grid(axis='y',alpha=.2);fig.tight_layout()
    fig.savefig(output/'cpu_efficiency.png',dpi=150);plt.close(fig)
    return rows,summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick',action='store_true')
    parser.add_argument('--output',default='outputs/bosons/replica_exchange_worm')
    parser.add_argument('--steps',type=int)
    parser.add_argument('--replicas',type=int)
    parser.add_argument('--links',type=int,nargs='+')
    parser.add_argument('--hot-fraction',type=float,default=.25)
    parser.add_argument('--height',type=float,default=3.)
    args=parser.parse_args()
    run_experiment(args.output,quick=args.quick,steps=args.steps,
                   replicas=args.replicas,links=args.links,
                   hot_fraction=args.hot_fraction,height=args.height)


if __name__=='__main__':
    main()
