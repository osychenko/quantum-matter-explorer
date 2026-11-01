"""Patch 25: exactly reweighted interaction-guided insertion path benchmark.

    python3 -m examples.bosons.run_guided_worm --quick
    python3 -m examples.bosons.run_guided_worm
    python3 -m examples.bosons.run_guided_worm --steps 3000 --replicas 6

Same Hamiltonian and ratio estimator for all MCMC methods. Reports independent
q-free and q-mix importance checks, spectral K=1/K=2 references, and CPU cost.
A quick run is an algorithm smoke test; no row is certified as converged.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from contextlib import nullcontext
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qmc.bosons import SoftCorePairPotential
from qmc.cells import CubicCell
from qmc.continuum_spacetime_worm import exact_free_addition_green
from qmc.green_spectral import TwoBosonAdditionReference
from qmc.guided_worm import GuidedWormConfig, GuidedContinuumWorm, independent_guided_green
from qmc.rare_worm import RareEventContinuumWorm
from qmc.worm_audit import audit_worm_chain, replica_summary


def _write_csv(path, rows):
    if not rows:
        raise ValueError("cannot save empty results")
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _one_thread():
    try:
        from threadpoolctl import threadpool_limits
        return threadpool_limits(limits=1)
    except ImportError:
        return nullcontext()


def _reference(cell, height, beta, time_mid):
    modes=((0,0,0),(1,0,0))
    if height==0:
        values=exact_free_addition_green(cell,1,beta,np.array([time_mid]),modes)[0]
        return values,values
    with _one_thread():
        a=TwoBosonAdditionReference(cell_length=cell.length,height=height,
                cutoff=1.2,k_max=1,n_grid=16).measure(beta,[time_mid],modes).green[0]
        b=TwoBosonAdditionReference(cell_length=cell.length,height=height,
                cutoff=1.2,k_max=2,n_grid=16).measure(beta,[time_mid],modes).green[0]
    return a,b


def run_experiment(output_dir="outputs/bosons/guided_worm", *, quick=False,
                   steps=None, replicas=None, links=None, iid_draws=None):
    out=Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    links_values=(4,) if quick else (4,6)
    if links is not None:
        if not isinstance(links,(tuple,list)) or not links or any(type(k) is not int or k<4 or k%2 for k in links):
            raise ValueError("links must contain even integers >=4")
        links_values=tuple(links)
    nsteps=(500 if quick else 1600) if steps is None else steps
    nrep=(2 if quick else 3) if replicas is None else replicas
    n_iid=(250 if quick else 650) if iid_draws is None else iid_draws
    if any(type(x) is not int for x in (nsteps,nrep,n_iid)) or nsteps<160 or nrep<2 or n_iid<16:
        raise ValueError("steps>=160, replicas>=2, iid_draws>=16 required")
    beta=2.4
    cell=CubicCell(2.8)
    modes=((0,0,0),(1,0,0))
    cases=(('ideal',0.),('repulsive',3.))
    methods=(('baseline',0.,0.,1), ('guided',.7,.5,1),('multiple_try_4',0.,0.,4))
    base=GuidedWormConfig(n_links=4,time_step=beta/4,
                          n_steps=nsteps,burn_in=min(nsteps//4,400),thin=3,seed=2026,
                          sector_fugacity=.9,toggle_probability=.42,
                          global_refresh_probability=.30,ghost_refresh_probability=.6,
                          time_move_probability=.45,endpoint_move_probability=.20,
                          exchange_bridge_probability=.25,drift_strength=.9,
                          initial_tilt_strength=1.,max_drift=.4)
    n_production=(nsteps-min(nsteps//4,400)+2)//3
    if n_production < 32:
        raise ValueError("not enough production measurements for four-block jackknife")
    n_blocks=8 if not quick and n_production>=128 else 4
    reference_rows=[]
    replica_rows=[]
    iid_rows=[]
    summary_rows=[]
    for M in links_values:
        mid=M//2-1
        for ic,(case,height) in enumerate(cases):
            potential=SoftCorePairPotential(cell,height=height,cutoff=1.2)
            k1,k2=_reference(cell,height,beta,beta/2)
            for ik,mode in enumerate(modes):
                reference_rows.append(dict(case=case,height=height,M=M,beta=beta,
                       mode_x=mode[0],mode_y=mode[1],mode_z=mode[2],
                       spectral_K1=float(k1[ik]),spectral_K2=float(k2[ik]),
                       basis_difference=float(k2[ik]-k1[ik])))
            for method,move_p,fraction,tries in methods:
                runs=[]
                for ir in range(nrep):
                    seed=200000+M*10000+ic*1000+ir*41+methods.index((method,move_p,fraction,tries))*100
                    cfg=replace(base,n_links=M,time_step=beta/M,seed=seed,
                                guided_move_probability=move_p,guided_fraction=fraction,
                                multiple_try_probability=1. if tries>1 else 0.,
                                multiple_try_candidates=tries)
                    # Always use the identical augmented canonical target.
                    sim=GuidedContinuumWorm(cell,potential,1,cfg,indices=modes)
                    start=time.process_time()
                    audit=audit_worm_chain(sim,n_blocks=n_blocks)
                    cpu=max(float(time.process_time()-start),np.finfo(float).tiny)
                    estimates=np.asarray(audit.green[mid],dtype=float)
                    essence=np.asarray(audit.ratio_ess[mid],dtype=float)
                    record=dict(case=case,height=height,M=M,beta=beta,tau=beta/M,
                        method=method,replica=ir,seed=seed,n_measurements=audit.n_measurements,
                        green_k0=float(estimates[0]),green_k1=float(estimates[1]),
                        block_stderr_k0=float(audit.jackknife_stderr[mid,0]),
                        half_block_stderr_k0=float(audit.half_block_stderr[mid,0]),
                        ratio_ess_k0=float(essence[0]),cpu_seconds=cpu,
                        ess_per_cpu=float(essence[0]/cpu),
                        spectral_k0=float(k2[0]),closed_visits=audit.closed_count,
                        guided_attempts=sim._counts['guided_refresh'][1],
                        guided_acceptance=audit.acceptance.get('guided_refresh',float('nan')),
                        multiple_attempts=sim._counts['multiple_try'][1],
                        multiple_acceptance=audit.acceptance.get('multiple_try',float('nan')),
                        global_acceptance=audit.acceptance.get('global_refresh',float('nan')))
                    replica_rows.append(record)
                    runs.append(record)
                    if method in ('baseline','guided'):
                        # Independent cross-check on independent random stream.
                        icfg=replace(cfg,seed=seed+900_000)
                        independ=GuidedContinuumWorm(cell,potential,1,icfg,indices=modes)
                        t0=time.process_time()
                        ind=independent_guided_green(independ,draws_per_interval=n_iid)
                        elapsed=max(float(time.process_time()-t0),np.finfo(float).tiny)
                        iid_rows.append(dict(case=case,height=height,M=M,method=method,
                            replica=ir,seed=icfg.seed,draws=n_iid,
                            green_k0=float(ind.green[mid,0]),iid_stderr_k0=float(ind.stderr[mid,0]),
                            weight_ess_k0=float(ind.weight_ess[mid]),
                            iid_ess_per_cpu=float(ind.weight_ess[mid]/elapsed),
                            top_one_pct_k0=float(ind.top_one_pct[mid,0]),
                            cpu_seconds=elapsed,spectral_k0=float(k2[0])))
                mean,sem=replica_summary([x['green_k0'] for x in runs])
                avg_ess=float(np.mean([x['ess_per_cpu'] for x in runs]))
                summary_rows.append(dict(case=case,height=height,M=M,tau=beta/M,
                    beta=beta,method=method,replicas=nrep,
                    green_k0=mean,replica_sem_k0=sem,spectral_K2_k0=float(k2[0]),
                    spectral_K1_k0=float(k1[0]),difference=mean-float(k2[0]),
                    mean_ratio_ess_per_cpu=avg_ess,
                    screening=('reference_discrepancy' if sem>0 and abs(mean-k2[0])>2*sem else 'inconclusive')))
                print(f"{case:9s} M={M} {method:14s}: G0(beta/2)={mean:.3f} ±{sem:.3f} "
                      f"(reference {k2[0]:.3f}), ESS/CPU={avg_ess:.2f}")
    _write_csv(out/'spectral_reference.csv',reference_rows)
    _write_csv(out/'replicas.csv',replica_rows)
    _write_csv(out/'independent_reference.csv',iid_rows)
    _write_csv(out/'summary.csv',summary_rows)
    with (out/'protocol.json').open('w',encoding='utf-8') as f:
        json.dump(dict(title='Patch 25: interaction-guided and exactly corrected insertion',
                       model='Canonical continuum N=1 -> 2, primitive-action G(k,t), beta=2.4',
                       cases=cases,methods=methods,links_values=links_values,
                       number_of_replicas=nrep,number_of_iid_draws=n_iid,
                       jackknife_blocks=n_blocks,
                       base_config=asdict(base),
                       guidance=('qmix=(1-p)qfree+p*qdrift; initial repulsive tilt with '
                                 'radial normalizer, drifted wrapped Gaussian transitions; '
                                 'MH and independent sampling use qfree/qmix'),
                       interpretation=('All estimates exploratory; no screening result certifies '
                                       'convergence. Exact ideal reference, truncated interacting K1/K2. '
                                       'CPU measurements depend on hardware; compare on held-out chains.')),
                  f,indent=2)
    fig,ax=plt.subplots(figsize=(7.5,4.5))
    for case,height in cases:
        for method,*_ in methods:
            rows=[r for r in summary_rows if r['case']==case and r['method']==method]
            ax.errorbar([r['tau'] for r in rows],[r['green_k0'] for r in rows],
                yerr=[r['replica_sem_k0'] for r in rows],marker='o',capsize=3,
                label=f'{case}, {method}')
        ref=next(r['spectral_K2_k0'] for r in summary_rows if r['case']==case)
        ax.axhline(ref,ls='--',alpha=.5,label=f'{case} spectral K2')
    ax.set(xlabel='Primitive timestep τ',ylabel=r'$G_0(\beta/2)$',
           title='Interaction-guided worm vs unchanged Green-function reference')
    ax.grid(alpha=.2);ax.legend(fontsize=7);fig.tight_layout()
    fig.savefig(out/'green_accuracy.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.3,4.3))
    for case,height in cases:
        for method,*_ in methods:
            rows=[r for r in summary_rows if r['case']==case and r['method']==method]
            ax.plot([r['M'] for r in rows],[r['mean_ratio_ess_per_cpu'] for r in rows],
                    'o-',label=f'{case}, {method}')
    ax.set(xlabel='Imaginary-time links M',ylabel='Ratio ESS / process CPU second',
           title='Guided-refresh efficiency (not equal CPU budgets)')
    ax.grid(alpha=.2);ax.legend(fontsize=7);fig.tight_layout()
    fig.savefig(out/'cpu_efficiency.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.2))
    for case,height in cases:
        for method in ('baseline','guided'):
            rows=[r for r in iid_rows if r['case']==case and r['method']==method]
            mm=sorted(set(x['M'] for x in rows))
            axes[0].plot(mm,[np.mean([x['weight_ess_k0']/x['draws'] for x in rows if x['M']==m]) for m in mm],
                         'o-',label=f'{case}, {method}')
            axes[1].plot(mm,[np.mean([x['top_one_pct_k0'] for x in rows if x['M']==m]) for m in mm],
                         'o-',label=f'{case}, {method}')
    axes[0].set(xlabel='Imaginary-time links M',ylabel='Independent importance weight ESS / draws',
                title='Weight degeneracy (larger is better)')
    axes[1].set(xlabel='Imaginary-time links M',ylabel='Top 1% absolute contribution share',
                title='Contribution concentration (smaller is better)')
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'importance_tails.png',dpi=160);plt.close(fig)
    print('Saved Patch 25 study to',out)
    return replica_rows,summary_rows,iid_rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--quick',action='store_true')
    p.add_argument('--output',default='outputs/bosons/guided_worm')
    p.add_argument('--steps',type=int)
    p.add_argument('--replicas',type=int)
    p.add_argument('--links',type=int,nargs='+')
    p.add_argument('--iid-draws',type=int)
    args=p.parse_args()
    run_experiment(args.output,quick=args.quick,steps=args.steps,
                   replicas=args.replicas,links=args.links,iid_draws=args.iid_draws)


if __name__=='__main__':
    main()
