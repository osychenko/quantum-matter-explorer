"""Patch 24: detailed-balance-correct rare-configuration sampling comparison.

    python3 -m examples.bosons.run_rare_worm --quick
    python3 -m examples.bosons.run_rare_worm

Compare Patch 22's independent G refresh with exact multiple-try MH at
fixed beta, same numbers of MC steps, and *measured process CPU time*.
Diagnostics are screens, not certification of convergence or physical BEC.
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
from qmc.rare_worm import (RareEventContinuumWorm, RareWormConfig,
                           contribution_concentration, independent_reference_green)
from qmc.worm_audit import audit_worm_chain, replica_summary, split_rhat


def _csv(filename, rows):
    if not rows:
        raise ValueError("cannot write empty CSV")
    with Path(filename).open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _thread_limit():
    try:
        from threadpoolctl import threadpool_limits
        return threadpool_limits(limits=1)
    except ImportError:
        return nullcontext()


def run_experiment(output_dir="outputs/bosons/rare_worm",*,quick=False,
                   steps=None, replicas=None):
    out=Path(output_dir)
    out.mkdir(parents=True,exist_ok=True)
    M_values=(4,) if quick else (4,6)
    beta=2.4
    nsteps=(360 if quick else 1300) if steps is None else steps
    nrep=(2 if quick else 3) if replicas is None else replicas
    if (type(nsteps) is not int or nsteps<160 or type(nrep) is not int or nrep<2):
        raise ValueError("steps must be integer >=160; replicas must be integer >=2")
    cell=CubicCell(2.8)
    height_cases=(('ideal',0.0),('repulsive',3.0))
    modes=((0,0,0),(1,0,0))
    methods=(('baseline',0.,1),('multiple_try_4',1.,4))
    base=RareWormConfig(n_links=4,time_step=beta/4,n_steps=nsteps,
                         burn_in=min(nsteps//4,300),thin=2,seed=2026,
                         sector_fugacity=.90,toggle_probability=.42,
                         global_refresh_probability=.30,ghost_refresh_probability=.60,
                         time_move_probability=.45,endpoint_move_probability=.20,
                         exchange_bridge_probability=.25)
    reference=[]
    replicas_rows=[]
    diagnostics=[]
    summary=[]
    independent_rows=[]
    for M in M_values:
        times=beta/M*np.arange(1,M)
        for case,height in height_cases:
            if height==0:
                exact=exact_free_addition_green(cell,1,beta,times,modes)
                K1=K2=exact
            else:
                # Independently truncated interacting N=1 -> 2 reference.
                with _thread_limit():
                    K1=TwoBosonAdditionReference(cell_length=2.8,height=height,
                         cutoff=1.2,k_max=1,n_grid=16).measure(beta,times,modes).green
                    K2=TwoBosonAdditionReference(cell_length=2.8,height=height,
                         cutoff=1.2,k_max=2,n_grid=16).measure(beta,times,modes).green
                exact=K2
            for it,t in enumerate(times):
                for ik, mode in enumerate(modes):
                    reference.append(dict(case=case,height=height,M=M,beta=beta,tau=beta/M,
                         separation=float(t),mode_x=mode[0],mode_y=mode[1],mode_z=mode[2],
                         spectral_K1=float(K1[it,ik]),spectral_K2=float(K2[it,ik]),
                         basis_difference=float(K2[it,ik]-K1[it,ik])))
            mid=M//2-1
            # Independent q-reference cross-check: exact target identity,
            # stratified by interval. It is not an MCMC convergence proof;
            # heavy-tailed importance weights can make its SEM optimistic.
            for ir in range(nrep):
                seed=990_000+M*10000+int(height)*1000+ir*47
                icfg=replace(base,n_links=M,time_step=beta/M,seed=seed)
                independent_sampler=RareEventContinuumWorm(
                    cell, SoftCorePairPotential(cell,height=height,cutoff=1.2),
                    1,icfg,indices=modes)
                t_start=time.process_time()
                ires=independent_reference_green(independent_sampler,
                        draws_per_interval=400 if quick else 1200)
                cost=time.process_time()-t_start
                for it,t in enumerate(times):
                    for ik, mode in enumerate(modes):
                        independent_rows.append(dict(case=case,height=height,M=M,
                            beta=beta,separation=float(t),
                            mode_x=mode[0],mode_y=mode[1],mode_z=mode[2],
                            replica=ir,seed=seed,green=float(ires.green[it,ik]),
                            iid_stderr=float(ires.stderr[it,ik]),
                            top_one_pct_contribution=float(ires.top_one_pct[it,ik]),
                            draws=ires.samples_per_interval,cpu_seconds=cost,
                            spectral_K2=float(exact[it,ik])))
            independent_center = [r['green'] for r in independent_rows
                if r['case']==case and r['M']==M and r['mode_x']==0
                and r['mode_y']==0 and np.isclose(r['separation'],beta/2)]
            i_mean,i_sem=replica_summary(independent_center)
            print(f'{case:9s} M={M} iid_q: G0(beta/2)={i_mean:.3f} ± {i_sem:.3f}'
                  f' (spectral {exact[mid,0]:.3f}; heavy-tail risk)')
            for method,prob,candidates in methods:
                runs=[]
                for ir in range(nrep):
                    # Same seed across strategies is not an independent paired
                    # estimate: algorithms consume different RNG streams.
                    seed=2400+M*100000+int(height)*10000+ir*127+int(prob)*3000
                    cfg=replace(base,n_links=M,time_step=beta/M,seed=seed,
                                multiple_try_probability=prob,
                                multiple_try_candidates=candidates)
                    potential=SoftCorePairPotential(cell,height=height,cutoff=1.2)
                    sampler=RareEventContinuumWorm(cell,potential,1,cfg,indices=modes)
                    start=time.process_time()
                    audit=audit_worm_chain(sampler,n_blocks=4 if quick else 8)
                    seconds=max(time.process_time()-start,np.finfo(float).tiny)
                    tail=contribution_concentration(audit.numerator_trace)
                    ess=float(audit.ratio_ess[mid,0]);est=float(audit.green[mid,0])
                    diag=dict(case=case,height=height,M=M,tau=beta/M,
                              method=method,replica=ir,seed=seed,
                              cpu_seconds=seconds,n_measurements=audit.n_measurements,
                              green_mid=est,jackknife_sem_mid=float(audit.jackknife_stderr[mid,0]),
                              half_block_sem_mid=float(audit.half_block_stderr[mid,0]),
                              exact_mid=float(exact[mid,0]),ratio_ess_mid=ess,
                              ess_per_cpu=ess/seconds,top_one_pct_contribution=float(tail[mid,0]),
                              closed_count=audit.closed_count,
                              global_acceptance=audit.acceptance.get('global_refresh',float('nan')),
                              multiple_try_acceptance=audit.acceptance.get('multiple_try',float('nan')),
                              attempted_multiple_try=sampler._counts['multiple_try'][1])
                    diagnostics.append(diag)
                    runs.append(audit)
                    for it,t in enumerate(times):
                        for ik,mode in enumerate(modes):
                            replicas_rows.append(dict(case=case,M=M,method=method,
                                replica=ir,seed=seed,separation=float(t),
                                mode_x=mode[0],mode_y=mode[1],mode_z=mode[2],
                                green=float(audit.green[it,ik]),
                                block_stderr=float(audit.jackknife_stderr[it,ik]),
                                green_raw=float(audit.green_raw[it,ik]),
                                ratio_ess=float(audit.ratio_ess[it,ik]),
                                reference=float(exact[it,ik])))
                for it,t in enumerate(times):
                    for ik, mode in enumerate(modes):
                        mean,sem=replica_summary([r.green[it,ik] for r in runs])
                        rh=split_rhat([r.influence[:,it,ik] for r in runs])
                        # No run is declared 'converged' from two or three replicas.
                        screen='reference_discrepancy' if (sem>0 and abs(mean-exact[it,ik])>2*sem) else 'inconclusive'
                        summary.append(dict(case=case,height=height,M=M,beta=beta,tau=beta/M,
                            method=method,replicas=nrep,separation=float(t),
                            mode_x=mode[0],mode_y=mode[1],mode_z=mode[2],
                            green_mean=mean,replica_sem=sem,spectral_K2=float(exact[it,ik]),
                            spectral_K1=float(K1[it,ik]),
                            difference=mean-float(exact[it,ik]),
                            split_rhat=rh,mean_ratio_ess=float(np.mean([r.ratio_ess[it,ik] for r in runs])),
                            mean_ess_per_cpu=float(np.mean([r.ratio_ess[it,ik]/d['cpu_seconds']
                                for r,d in zip(runs,diagnostics[-nrep:])])),
                            screening=screen))
                selected=summary[-len(times)*len(modes):]
                center=next(x for x in selected if x['mode_x']==0 and x['mode_y']==0
                            and np.isclose(x['separation'], beta/2))
                print(f'{case:9s} M={M} {method:14s} '
                      f'G0(beta/2)={center["green_mean"]:.3f} ± {center["replica_sem"]:.3f} '
                      f'(spectral {exact[mid,0]:.3f}), '
                      f'ESS/CPU={np.mean([x["ess_per_cpu"] for x in diagnostics[-nrep:]]):.2f}')
    _csv(out/'spectral_reference.csv',reference)
    _csv(out/'replicas.csv',replicas_rows)
    _csv(out/'diagnostics.csv',diagnostics)
    _csv(out/'independent_reference.csv',independent_rows)
    _csv(out/'summary.csv',summary)
    with (out/'protocol.json').open('w',encoding='utf-8') as f:
        json.dump(dict(benchmark='rare-event independence multiple-try vs baseline plus independent q stratification',
                       beta=beta,links=M_values,methods=methods,cases=height_cases,
                       replicas=nrep,base_config=asdict(base),
                       notes=['N_background=1, insertion N+1=2; canonical finite-T Green function',
                              'Multiple-try uses exact reversibility correction',
                              'CPU is process_time for sampled chains, machine-specific',
                              'Ratios are correlated; ESS is an autocorrelation diagnostic',
                              'Independent q importance sampling is unbiased at fixed action but can be heavy-tailed',
                              'No screening label certifies convergence or continuum physics']),f,indent=2)
    fig,ax=plt.subplots(figsize=(7.5,4.3))
    for case,height in height_cases:
        for method,_,_ in methods:
            x=[r for r in summary if r['case']==case and r['method']==method
               and r['mode_x']==0 and r['mode_y']==0 and np.isclose(r['separation'],beta/2)]
            ax.errorbar([r['tau'] for r in x],[r['green_mean'] for r in x],
                        yerr=[r['replica_sem'] for r in x],marker='o',capsize=3,
                        label=f'{case}, {method}')
        exact=next(r['spectral_K2'] for r in reference if r['case']==case and
                    np.isclose(r['separation'],beta/2) and r['mode_x']==0)
        ax.axhline(exact,ls='--',alpha=.45,label=f'{case} spectral')
    ax.set(xlabel='Imaginary-time step',ylabel=r'$G_{k=0}(\beta/2)$',
           title='Independent-replica means vs spectral reference')
    ax.grid(alpha=.25);ax.legend(fontsize=7);fig.tight_layout()
    fig.savefig(out/'green_reference.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10.4,4.3))
    for case,height in height_cases:
        for method,_,_ in methods:
            x=[r for r in diagnostics if r['case']==case and r['method']==method]
            # Mean across independent replicas for each fixed M.
            mvals=sorted(set(r['M'] for r in x))
            axes[0].plot(mvals,[np.mean([r['ess_per_cpu'] for r in x if r['M']==M]) for M in mvals],
                    'o-',label=f'{case}, {method}')
            axes[1].plot(mvals,[np.mean([r['top_one_pct_contribution'] for r in x if r['M']==M]) for M in mvals],
                    'o-',label=f'{case}, {method}')
    axes[0].set(xlabel='Imaginary-time links',ylabel='Ratio ESS per process CPU-second',
                title='Accuracy-cost diagnostic')
    axes[1].set(xlabel='Imaginary-time links',ylabel='Top 1% absolute contribution share',
                title='Rare-event concentration')
    for ax in axes:ax.grid(alpha=.25);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'rare_efficiency.png',dpi=150);plt.close(fig)
    print('Saved Patch 24 study to',out)
    return diagnostics,summary


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--quick',action='store_true')
    p.add_argument('--output',default='outputs/bosons/rare_worm')
    p.add_argument('--steps',type=int)
    p.add_argument('--replicas',type=int)
    args=p.parse_args()
    run_experiment(args.output,quick=args.quick,steps=args.steps,replicas=args.replicas)


if __name__=='__main__':
    main()
