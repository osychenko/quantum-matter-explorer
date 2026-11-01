"""Patch 23: fixed-beta continuum worm statistical/systematic error audit.

    python3 -m examples.bosons.run_worm_audit --quick
    python3 -m examples.bosons.run_worm_audit

Compares independent chains, run lengths and *predeclared* move mixtures.
Finite-temperature N=1 -> 2 addition Green function, NOT condensate fraction.
The quick run is a smoke test and is never evidence of convergence.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qmc.bosons import SoftCorePairPotential
from qmc.cells import CubicCell
from qmc.green_spectral import TwoBosonAdditionReference
from qmc.reliable_spacetime_worm import ReliableContinuumSpacetimeWorm, ReliableSpacetimeConfig
from qmc.worm_audit import audit_worm_chain, replica_summary, split_rhat


@dataclass(frozen=True)
class WormAuditConfig:
    betas: tuple[float, ...] = (2.4, 3.2)
    n_links: tuple[int, ...] = (4, 6)
    heights: tuple[float, ...] = (0.0, 3.0)
    chain_steps: tuple[int, ...] = (2400, 4800)
    mixtures: tuple[str, ...] = ("baseline", "refresh_heavy")
    replicas: int = 3
    burn_fraction: float = 0.24
    thin: int = 3
    n_blocks: int = 8
    seed: int = 2026
    cell_length: float = 2.8
    cutoff: float = 1.2
    kinetic_coefficient: float = 0.5
    modes: tuple[tuple[int, int, int], ...] = ((0,0,0),(1,0,0))


def _check(cfg):
    if not isinstance(cfg,WormAuditConfig):
        raise TypeError("cfg must be WormAuditConfig")
    for name in ("betas","heights","chain_steps","n_links","mixtures"):
        if not getattr(cfg,name):
            raise ValueError(f"{name} cannot be empty")
    for name in ("betas", "heights"):
        values=np.asarray(getattr(cfg,name),dtype=float)
        if values.ndim!=1 or not np.all(np.isfinite(values)) or np.any(values<(0 if name=="heights" else 1e-15)):
            raise ValueError(f"invalid {name}")
        if len(set(values.tolist()))!=len(values):
            raise ValueError(f"duplicate {name}")
    for name in ("n_links","chain_steps"):
        values=getattr(cfg,name)
        if any(isinstance(x,(bool,np.bool_)) or not isinstance(x,(int,np.integer)) or x<4 for x in values):
            raise ValueError(f"invalid {name}")
        if len(set(values))!=len(values):
            raise ValueError(f"duplicate {name}")
    if any(M%2 for M in cfg.n_links):
        raise ValueError("n_links must be even so beta/2 is on all grids")
    if any(x not in ("baseline","refresh_heavy") for x in cfg.mixtures) or len(set(cfg.mixtures))!=len(cfg.mixtures):
        raise ValueError("mixtures must be distinct baseline/refresh_heavy")
    for key, minimum in (("replicas",2),("thin",1),("n_blocks",4),("seed",0)):
        v=getattr(cfg,key)
        if isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,np.integer)) or v<minimum:
            raise ValueError(f"{key} must be integer >= {minimum}")
    if not np.isfinite(cfg.burn_fraction) or not 0<cfg.burn_fraction<0.7:
        raise ValueError("burn_fraction must be in (0,0.7)")
    if any(((s-int(s*cfg.burn_fraction)+cfg.thin-1)//cfg.thin)<8*cfg.n_blocks
           for s in cfg.chain_steps):
        raise ValueError("each chain needs at least eight measurements per jackknife block")
    cell=CubicCell(cfg.cell_length)
    SoftCorePairPotential(cell,height=1.,cutoff=cfg.cutoff)
    if not np.isfinite(cfg.kinetic_coefficient) or cfg.kinetic_coefficient<=0:
        raise ValueError("kinetic_coefficient must be positive")
    modes=np.asarray(cfg.modes)
    if (modes.ndim!=2 or modes.shape[1]!=3 or not np.issubdtype(modes.dtype,np.number)
            or not np.all(np.isfinite(modes)) or not np.all(modes==np.rint(modes))
            or not np.any(np.all(modes==0,axis=1)) or np.max(abs(modes))>1):
        raise ValueError("modes must include zero and lie within Fourier basis K=1")
    return cell


def _csv(path,rows, *, empty_header=()):
    if not rows and not empty_header:
        raise ValueError("empty CSV rows")
    with Path(path).open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else list(empty_header))
        writer.writeheader();writer.writerows(rows)


def _mixture(name):
    if name=="baseline":
        return dict(toggle_probability=.42,global_refresh_probability=.30,
                    ghost_refresh_probability=.60,time_move_probability=.45,
                    endpoint_move_probability=.20,exchange_bridge_probability=.25)
    if name=="refresh_heavy":
        return dict(toggle_probability=.35,global_refresh_probability=.49,
                    ghost_refresh_probability=.85,time_move_probability=.35,
                    endpoint_move_probability=.15,exchange_bridge_probability=.40)
    raise ValueError("unknown mixture")


def _thread_limit_context():
    try:
        from threadpoolctl import threadpool_limits
        return threadpool_limits(limits=1)
    except ImportError:
        return nullcontext()


def run_experiment(output_dir="outputs/bosons/worm_audit", *, config=None,quick=False):
    if config is None:
        config = (WormAuditConfig(betas=(2.4,),chain_steps=(360,720),
                                  mixtures=("baseline",),replicas=2)
                  if quick else WormAuditConfig())
    cell=_check(config)
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    base=ReliableSpacetimeConfig(n_steps=360,burn_in=70,thin=config.thin)
    spec={} ; refs=[]
    with _thread_limit_context():
        for height in config.heights:
            for K in (1,2):
                spec[height,K]=TwoBosonAdditionReference(
                    cell_length=config.cell_length,kinetic_coefficient=config.kinetic_coefficient,
                    cutoff=config.cutoff,height=height,k_max=K,n_grid=16)
        for beta in config.betas:
            for M in config.n_links:
                times=np.arange(1,M)*beta/M
                for height in config.heights:
                    for K in (1,2):
                        result=spec[height,K].measure(beta,times,config.modes)
                        for it,t in enumerate(times):
                            for ik,mode in enumerate(config.modes):
                                refs.append(dict(beta=beta,n_links=M,tau=beta/M,height=height,
                                                 time_index=it,separation=float(t),mode_x=mode[0],mode_y=mode[1],mode_z=mode[2],
                                                 k_max=K,reference=float(result.green[it,ik])))
    ref_lookup={(r['beta'],r['n_links'],r['height'],r['time_index'],
                 r['mode_x'],r['mode_y'],r['mode_z'],r['k_max']):r['reference'] for r in refs}
    replica_rows=[];summary_rows=[]
    ident=0
    for beta in config.betas:
        for M in config.n_links:
            tau=beta/M
            for height in config.heights:
                for mixture in config.mixtures:
                    for steps in config.chain_steps:
                        runs=[];cpu_times=[]
                        for ir in range(config.replicas):
                            seed=config.seed+ident*10007+ir*173
                            overrides=_mixture(mixture)
                            burn=int(steps*config.burn_fraction)
                            settings=replace(base,time_step=tau,n_links=M,n_steps=steps,
                                             burn_in=burn,seed=seed,**overrides)
                            potential=SoftCorePairPotential(cell,height=height,cutoff=config.cutoff)
                            sampler=ReliableContinuumSpacetimeWorm(cell,potential,1,settings,
                                kinetic_coefficient=config.kinetic_coefficient,indices=config.modes)
                            started=time.process_time()
                            result=audit_worm_chain(sampler,n_blocks=config.n_blocks)
                            cpu=time.process_time()-started
                            runs.append(result)
                            cpu_times.append(cpu)
                            for jt,t in enumerate(result.imaginary_times):
                                for ik,mode in enumerate(config.modes):
                                    replica_rows.append(dict(beta=beta,n_links=M,tau=tau,height=height,
                                        mixture=mixture,n_steps=steps,replica=ir,seed=seed,
                                        separation=float(t),mode_x=mode[0],mode_y=mode[1],mode_z=mode[2],
                                        green=float(result.green[jt,ik]),raw_green=float(result.green_raw[jt,ik]),
                                        block_jackknife_sem=float(result.jackknife_stderr[jt,ik]),
                                        coarse_block_sem=float(result.half_block_stderr[jt,ik]),
                                        ratio_ess=float(result.ratio_ess[jt,ik]),
                                        n_measurements=result.n_measurements,n_closed=result.closed_count,
                                        n_open_at_time=int(result.open_counts[jt]),
                                        acceptance_global_refresh=float(result.acceptance.get('global_refresh',float('nan'))),
                                        acceptance_time=float(result.acceptance.get('time',float('nan'))),
                                        acceptance_exchange_bridge=float(result.acceptance.get('exchange_bridge',float('nan'))),
                                        cpu_seconds=cpu,ess_per_cpu_second=float(result.ratio_ess[jt,ik]/cpu)
                                        if cpu>0 else float('nan')))
                        ident+=1
                        matrix=np.asarray([r.green for r in runs])
                        # Estimate Rhat for the *common pooled ratio influence*;
                        # centering each chain by its own ratio would mask between-chain drift.
                        common_den=sum(float(r.denominator_trace.sum()) for r in runs)
                        pooled=(M-1)/settings.sector_fugacity * sum(r.numerator_trace.sum(axis=0) for r in runs)/common_den
                        for jt,t in enumerate(runs[0].imaginary_times):
                            for ik,mode in enumerate(config.modes):
                                means=matrix[:,jt,ik]
                                mean,sem=replica_summary(means)
                                rhat=split_rhat([((M-1)/settings.sector_fugacity*r.numerator_trace[:,jt,ik]
                                                   -pooled[jt,ik]*r.denominator_trace)
                                                  for r in runs])
                                ref1=ref_lookup[beta,M,height,jt,*mode,1]
                                ref2=ref_lookup[beta,M,height,jt,*mode,2]
                                median_ess=float(np.median([r.ratio_ess[jt,ik] for r in runs]))
                                median_open_visits=float(np.median([r.open_counts[jt] for r in runs]))
                                median_ess_cpu=float(np.median([r.ratio_ess[jt,ik]/s for r,s in zip(runs,cpu_times) if s>0]))
                                relative_mc=abs(mean-ref2)/sem if sem>0 else float('inf')
                                reblock=np.median([abs(r.half_block_stderr[jt,ik]/r.jackknife_stderr[jt,ik]-1)
                                    if r.jackknife_stderr[jt,ik]>0 else float('inf') for r in runs])
                                # Gate is intentionally conservative, never report "converged"
                                # based solely on small SEM or a matching reference.
                                flags=[]
                                if config.replicas<4 or min(config.chain_steps)<2000:
                                    flags.append('short_protocol')
                                if median_ess<30:flags.append('low_ess')
                                if median_open_visits<5:flags.append('few_open_visits')
                                if not np.isfinite(rhat) or rhat>1.10: flags.append('rhat')
                                if not np.isfinite(reblock) or reblock>.6:flags.append('block_sensitivity')
                                if np.isfinite(relative_mc) and relative_mc>2:flags.append('reference_tension')
                                if not np.isfinite(relative_mc):flags.append('zero_replica_sem')
                                summary_rows.append(dict(beta=beta,n_links=M,tau=tau,height=height,
                                    mixture=mixture,n_steps=steps,replicas=config.replicas,
                                    separation=float(t),mode_x=mode[0],mode_y=mode[1],mode_z=mode[2],
                                    green=mean,replica_sem=sem,
                                    ratio_ess_median=median_ess,median_ess_per_cpu_second=median_ess_cpu,
                                    median_open_visits=median_open_visits,split_rhat=rhat,
                                    reblock_median_relative_change=float(reblock),
                                    reference_K1=ref1,reference_K2=ref2,basis_difference=ref2-ref1,
                                    difference_from_K2=mean-ref2,
                                    discrepancy_in_replica_sem=float(relative_mc),
                                    status='inconclusive' if flags else 'passes_screening_only',
                                    flags=';'.join(flags) if flags else 'none'))
                        mid=[x for x in summary_rows if x['beta']==beta and x['n_links']==M
                             and x['height']==height and x['mixture']==mixture and x['n_steps']==steps
                             and (x['mode_x'],x['mode_y'],x['mode_z'])==(0,0,0)
                             and np.isclose(x['separation'],beta/2)]
                        if mid:
                            x=mid[0]
                            print(f"beta={beta:g} M={M} V0={height:g} {mixture} steps={steps} "
                                  f"G0(beta/2)={x['green']:.3f} +/- {x['replica_sem']:.3f} "
                                  f"K2={x['reference_K2']:.3f} ESS={x['ratio_ess_median']:.1f} "
                                  f"Rhat={x['split_rhat']:.2f} {x['status']}")
    contrasts=_contrasts(summary_rows)
    _csv(out/'contrasts.csv',contrasts,empty_header=(
        'kind','beta','height','mixture','separation','mode_x','mode_y','mode_z',
        'from_n_links','to_n_links','from_n_steps','to_n_steps','from_green','to_green',
        'difference','independent_replica_difference_sem','difference_in_sem','interpretation'))
    _csv(out/'spectral_reference.csv',refs)
    _csv(out/'replicas.csv',replica_rows)
    _csv(out/'summary.csv',summary_rows)
    with (out/'protocol.json').open('w',encoding='utf-8') as f:
        json.dump(dict(config=asdict(config),reference='truncated symmetric two-boson Fourier Galerkin, K=1,2',
                       statistical_warning=('block jackknife is correlation-sensitive; replica SEM and split-Rhat are independent checks; '
                           'passing screening does not prove convergence or bound systematic errors'),
                       physics='canonical N=1 -> 2 finite-temperature particle-addition Green function'),f,indent=2)
    _plots(out,summary_rows,config)
    print('Saved Patch 23 audit:',out)
    return replica_rows,summary_rows,refs


def _contrasts(rows):
    """Predeclared independent-chain duration/timestep contrasts at matched physics.

    We do NOT use a continuum extrapolation or combine spectral basis
    truncation with Monte Carlo uncertainty. The two grid estimates have
    independent seeds, so their replica SEMs combine in quadrature.
    """
    comparisons=[]
    for left_index,first in enumerate(rows):
        for second in rows[left_index+1:]:
            common=("beta","height","mixture","separation","mode_x","mode_y","mode_z")
            if any(first[k]!=second[k] for k in common):
                continue
            if first["n_links"]==second["n_links"] and first["n_steps"]!=second["n_steps"]:
                # Exactly one direction (short->long).
                if first["n_steps"]>second["n_steps"]:continue
                kind='run_length'
            elif first["n_steps"]==second["n_steps"] and first["n_links"]!=second["n_links"]:
                if first["n_links"]>second["n_links"]:continue
                kind='time_step'
            else:
                continue
            stderr=float(np.hypot(first['replica_sem'],second['replica_sem']))
            delta=float(second['green']-first['green'])
            comparisons.append(dict(kind=kind,beta=first['beta'],height=first['height'],
                mixture=first['mixture'],separation=first['separation'],
                mode_x=first['mode_x'],mode_y=first['mode_y'],mode_z=first['mode_z'],
                from_n_links=first['n_links'],to_n_links=second['n_links'],
                from_n_steps=first['n_steps'],to_n_steps=second['n_steps'],
                from_green=first['green'],to_green=second['green'],
                difference=delta,independent_replica_difference_sem=stderr,
                difference_in_sem=abs(delta)/stderr if stderr>0 else float('nan'),
                interpretation='independent-chain comparison, NOT a zero-time-step extrapolation'))
    # In the quick protocol the beta/2 time on even grids ensures at least
    # one time-step contrast and one run-length contrast.
    return comparisons


def _plots(output,rows,config):
    mid=[r for r in rows if (r['mode_x'],r['mode_y'],r['mode_z'])==(0,0,0)
         and np.isclose(r['separation'],r['beta']/2)]
    beta=max(config.betas);height=max(config.heights);steps=max(config.chain_steps)
    fig,ax=plt.subplots(figsize=(7.3,4.6))
    pts=[r for r in mid if r['beta']==beta and r['height']==height and r['n_steps']==steps]
    for mixture in config.mixtures:
        selected=sorted((r for r in pts if r['mixture']==mixture),key=lambda r:r['tau'])
        ax.errorbar([r['tau'] for r in selected],[r['green'] for r in selected],
                    yerr=[r['replica_sem'] for r in selected],fmt='o-',capsize=3,label=mixture)
    if pts: ax.axhline(pts[0]['reference_K2'],color='gray',ls='--',label='spectral K=2')
    ax.set(xlabel='Primitive time step τ, fixed β',ylabel=r'$G_{k=0}^{(+)}(\beta/2)$',
           title='Interacting timestep audit — error bars across independent chains')
    ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(output/'timestep_audit.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.3,4.6))
    mixture=config.mixtures[0]
    for M in config.n_links:
        pts=sorted((r for r in mid if r['beta']==beta and r['height']==height and r['n_links']==M
                    and r['mixture']==mixture),key=lambda r:r['n_steps'])
        ax.errorbar([r['n_steps'] for r in pts],[r['green'] for r in pts],
                    yerr=[r['replica_sem'] for r in pts],marker='o',capsize=3,label=f'M={M}')
    if pts:ax.axhline(pts[0]['reference_K2'],color='gray',ls='--',label='spectral K=2')
    ax.set(xlabel='Markov steps per chain',ylabel=r'$G_{k=0}^{(+)}(\beta/2)$',
           title='Run-length sensitivity, separate independent seeds')
    ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(output/'length_audit.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.3,4.6))
    selected=[r for r in mid if r['beta']==beta and r['height']==height and r['n_steps']==steps]
    for mixture in config.mixtures:
        pts=sorted((r for r in selected if r['mixture']==mixture),key=lambda r:r['tau'])
        ax.plot([r['tau'] for r in pts],[r['ratio_ess_median'] for r in pts],'o-',label=mixture)
    ax.axhline(30,color='gray',ls=':',label='30 ESS screening threshold')
    ax.set(xlabel='Primitive time step τ',ylabel='Median ratio-influence ESS / chain',
           title='Autocorrelation of the actual Green-function ratio')
    ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(output/'ratio_ess.png',dpi=150);plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick',action='store_true',help='Short numerical integration check, NOT convergence evidence')
    parser.add_argument('--output',default='outputs/bosons/worm_audit')
    parser.add_argument('--steps',type=int,nargs='+',help='Independent chain durations, e.g. --steps 4000 12000')
    parser.add_argument('--replicas',type=int,help='Independent chains per condition, ideally >=4')
    parser.add_argument('--links',type=int,nargs='+',help='Even link counts, fixed beta')
    parser.add_argument('--betas',type=float,nargs='+',help='Inverse temperatures')
    args=parser.parse_args()
    cfg = (WormAuditConfig(betas=(2.4,),chain_steps=(360,720),
                           mixtures=('baseline',),replicas=2)
           if args.quick else WormAuditConfig())
    overrides={}
    if args.steps is not None:overrides['chain_steps']=tuple(args.steps)
    if args.replicas is not None:overrides['replicas']=args.replicas
    if args.links is not None:overrides['n_links']=tuple(args.links)
    if args.betas is not None:overrides['betas']=tuple(args.betas)
    run_experiment(args.output,config=replace(cfg,**overrides))

if __name__=='__main__':
    main()
