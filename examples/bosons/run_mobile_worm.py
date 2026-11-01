"""Patch 19: time-cut mobility and finite-temperature winding response.

    python3 -m examples.bosons.run_mobile_worm --quick
    python3 -m examples.bosons.run_mobile_worm

This is an equal-time canonical worm with a *mobile time-slice cut*, not a
fully spacetime-separated head/tail worm and not a zero-temperature estimator.
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from qmc.bosons import SoftCorePairPotential
from qmc.cells import CubicCell
from qmc.mobile_worm import MobileCutWorm,MobileWormConfig
from qmc.winding import canonical_free_winding
from qmc.worm import canonical_free_occupations


def _write_csv(path, rows, fieldnames):
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mean_sem(values):
    x=np.asarray(values,dtype=float)
    if len(x)<2 or not np.all(np.isfinite(x)):
        raise ValueError('need >=2 finite independent replicas')
    return float(x.mean()),float(x.std(ddof=1)/np.sqrt(len(x)))


def run_experiment(output_dir='outputs/bosons/mobile_worm',*,quick=False):
    out=Path(output_dir)
    out.mkdir(parents=True,exist_ok=True)
    cell=CubicCell(3.6)
    N=2;cutoff=1.5;D=.5;beta=2.1
    steps=(.35,) if quick else (.35,.2625)
    heights=(0.,3.)
    replicas=2 if quick else 3
    config=MobileWormConfig(
        n_beads=6,time_step=.35,n_steps=520 if quick else 1800,
        burn_in=100 if quick else 400,thin=4 if quick else 6,
        refresh_probability=.12,translate_probability=.05,
        swap_probability=.20,bridge_probability=.25,
        bridge_lengths=(2,3),collapsed_refresh_probability=.13,
        sector_fugacity=1.,toggle_probability=.42,gibbs_probability=.90,
        relocate_probability=.65,local_shift_probability=.1)
    ref_occ=canonical_free_occupations(cell,N,beta,((0,0,0),(1,0,0)),D)
    ref_w=canonical_free_winding(cell,N,beta,D)
    replica_rows=[];summary_rows=[];cut_rows=[]
    for it,tau in enumerate(steps):
        M=round(beta/tau)
        if M<4 or not np.isclose(M*tau,beta):
            raise ValueError('time-step grid must preserve fixed beta')
        for ih,height in enumerate(heights):
            if height==0:
                name='free_bosons'
            else:
                name='soft_core'
            occ=[];winding=[];closed=[];visits=[]
            hist=np.zeros(M,dtype=float)
            potential=SoftCorePairPotential(cell,height=height,cutoff=cutoff)
            for ir in range(replicas):
                seed=29000+it*100000+ih*10000+ir*107
                cfg=replace(config,time_step=tau,n_beads=M,seed=seed)
                sampler=MobileCutWorm(cell,potential,N,cfg,kinetic_coefficient=D)
                result=sampler.run()
                occ.append(result.n0_fraction)
                winding.append(result.superfluid_fraction_isotropic)
                closed.append(float(result.closed_occupation_fractions[0]))
                visits.append(float(result.n_closed))
                hist+=result.time_cut_counts
                replica_rows.append(dict(model=name,tau=tau,beta=beta,links=M,
                    replica=ir,seed=seed,n0_worm=result.n0_fraction,
                    n0_closed_link=float(result.closed_occupation_fractions[0]),
                    winding_squared_x=float(result.winding_squared[0]),
                    winding_squared_y=float(result.winding_squared[1]),
                    winding_squared_z=float(result.winding_squared[2]),
                    winding_response=result.superfluid_fraction_isotropic,
                    n_closed=result.n_closed,n_open=result.n_open,
                    sector_effective_samples=result.sector_effective_samples,
                    cut_relocation_acceptance=result.acceptance['relocate_cut'],
                    collapsed_refresh_acceptance=result.acceptance['collapsed_refresh']))
            n0,sem0=_mean_sem(occ)
            wc,semw=_mean_sem(winding)
            cc,semc=_mean_sem(closed)
            summary_rows.append(dict(model=name,tau=tau,beta=beta,links=M,
                replicas=replicas,n0_worm=n0,n0_stderr=sem0,
                n0_closed_link=cc,n0_closed_link_stderr=semc,
                winding_response=wc,winding_stderr=semw,
                exact_free_n0=float(ref_occ[0]) if height==0 else '',
                exact_free_winding_response=ref_w.fraction_isotropic if height==0 else ''))
            if hist.sum():
                for t,v in enumerate(hist):
                    cut_rows.append(dict(model=name,tau=tau,cut=t,
                                         open_visits=int(v),open_visit_fraction=float(v/hist.sum())))
            print(f'{name:11s} tau={tau:.3f}: n0/N={n0:.4f} ± {sem0:.4f} '
                  f'winding={wc:.4f} ± {semw:.4f}')
    _write_csv(out/'replicas.csv',replica_rows,replica_rows[0].keys())
    _write_csv(out/'summary.csv',summary_rows,summary_rows[0].keys())
    _write_csv(out/'time_cuts.csv',cut_rows,cut_rows[0].keys())
    with (out/'protocol.json').open('w',encoding='utf-8') as f:
        json.dump(dict(beta=beta,n_particles=N,cutoff=cutoff,kinetic_coefficient=D,
                       heights=heights,time_steps=steps,replicas=replicas,
                       config=asdict(config),
                       interpretation='finite-temperature equal-time mobile-cut worm; '
                                      'conditional winding image integration; not a mobile head/tail spacetime worm',
                       limitations='finite tau, finite beta, finite N, autocorrelation, '
                                   'replica uncertainty, no ground-state or macroscopic superfluid claim'),f,indent=2)
    fig,ax=plt.subplots(figsize=(7,4.7))
    for name,label in [('free_bosons','Ideal bosons'),('soft_core','Soft-core repulsion')]:
        rows=sorted((r for r in summary_rows if r['model']==name),key=lambda r:r['tau'])
        ax.errorbar([r['tau'] for r in rows],[r['winding_response'] for r in rows],
                    yerr=[r['winding_stderr'] for r in rows],marker='o',capsize=3,label=label)
    ax.axhline(ref_w.fraction_isotropic,color='gray',ls='--',label='Exact ideal Bose reference')
    ax.set(xlabel='Primitive time step τ',ylabel='Finite-size winding response',
           title='Winding-number response at fixed β')
    ax.grid(alpha=.2);ax.legend();fig.tight_layout()
    fig.savefig(out/'winding_response.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4.7))
    for name,label in [('free_bosons','Ideal bosons'),('soft_core','Soft-core repulsion')]:
        rows=sorted((r for r in summary_rows if r['model']==name),key=lambda r:r['tau'])
        ax.errorbar([r['tau'] for r in rows],[r['n0_worm'] for r in rows],
                    yerr=[r['n0_stderr'] for r in rows],marker='o',capsize=3,label=label+' (worm)')
        ax.plot([r['tau'] for r in rows],[r['n0_closed_link'] for r in rows],
                'x--',label=label+' (closed link)')
    ax.axhline(ref_occ[0],color='gray',ls=':',label='Exact ideal Bose reference')
    ax.set(xlabel='Primitive time step τ',ylabel='Finite-cell n₀/N (estimator)',
           title='Mobile-cut coherence with a normalization cross-check')
    ax.grid(alpha=.2);ax.legend(fontsize=8);fig.tight_layout()
    fig.savefig(out/'coherence.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4.1))
    for t in steps:
        rows=[r for r in cut_rows if r['model']=='free_bosons' and r['tau']==t]
        ax.plot([r['cut']/len(rows) for r in rows],
                [r['open_visit_fraction']*len(rows) for r in rows],
                'o-',label=f'τ={t:g}')
    ax.axhline(1,color='gray',ls='--')
    ax.set(xlabel='Cut index / number of links',ylabel='Open-cut frequency / uniform expectation',
           title='Imaginary-time cut mobility (ideal bosons)')
    ax.grid(alpha=.2);ax.legend();fig.tight_layout()
    fig.savefig(out/'cut_mobility.png',dpi=150);plt.close(fig)
    print('Saved Patch 19 outputs:',out)
    return replica_rows,summary_rows,cut_rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick',action='store_true')
    parser.add_argument('--output',default='outputs/bosons/mobile_worm')
    args=parser.parse_args()
    run_experiment(args.output,quick=args.quick)

if __name__=='__main__':main()
