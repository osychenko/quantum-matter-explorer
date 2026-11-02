"""Patch 28: CPU-calibrated held-out comparison for N=1 -> 2 continuum worms.

python3 -m examples.bosons.run_equal_cpu_worm --quick
python3 -m examples.bosons.run_equal_cpu_worm --cpu-seconds 12 --replicas 4
"""
from __future__ import annotations
import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from qmc.cells import CubicCell
from qmc.bosons import SoftCorePairPotential
from qmc.replica_exchange_worm import ReplicaExchangeConfig
from qmc.equal_cpu_worm import plan_budget,run_validation,summarize_validation
from qmc.green_spectral import TwoBosonAdditionReference


def csv_write(path, rows):
    if not rows: raise ValueError('empty output')
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader();w.writerows(rows)


def run_experiment(output='outputs/bosons/equal_cpu_worm',*,quick=False,
                   cpu_seconds=None,replicas=None,links=None):
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    cell=CubicCell(2.8); height=3.;cutoff=1.2;beta=2.4
    pot=SoftCorePairPotential(cell,height=height,cutoff=cutoff)
    target=(.30 if quick else 4.0) if cpu_seconds is None else float(cpu_seconds)
    count=(2 if quick else 4) if replicas is None else replicas
    grids=(4,) if quick else (4,6)
    if links is not None: grids=tuple(links)
    if not np.isfinite(target) or target<=0 or type(count) is not int or count<2:
        raise ValueError('cpu-seconds >0 and replicas >=2 are required')
    if not grids or any(type(m) is not int or m<4 or m%2 for m in grids):
        raise ValueError('links must be even integers >=4')
    pilot=[]; runs=[]; summaries=[];refs=[]
    for M in grids:
        # Full reference costs substantially more; K=1 is sufficient for the
        # smoke benchmark. The reference basis error is not the MC SEM.
        ref=TwoBosonAdditionReference(cell_length=cell.length,height=height,cutoff=cutoff,k_max=1,n_grid=16)
        expected=float(ref.measure(beta,[beta/2],((0,0,0),)).green[0,0]);refs.append({'M':M,'beta':beta,'k_max':1,'reference':expected})
        base=ReplicaExchangeConfig(n_links=M,time_step=beta/M,
             n_steps=200,burn_in=40,thin=2,sector_fugacity=.9,
             hot_interaction_fraction=.25,swap_probability=.3)
        for method in ('baseline','exchange'):
            pilot_seed=100000+M*1000+(0 if method=='baseline' else 100)
            plan=plan_budget(method,cell,pot,base,target_seconds=target,
                pilot_steps=12 if quick else 40,seed=pilot_seed,
                min_validation_steps=160 if quick else 800,
                max_validation_steps=600 if quick else 100000)
            pilot.append({'M':M,**asdict(plan)})
            group=[]
            for i in range(count):
                seed=300000+M*10000+(0 if method=='baseline' else 1000)+i
                r=run_validation(method,cell,pot,base,plan,seed=seed)
                r['M']=M;r['beta']=beta;r['spectral_reference']=expected
                runs.append(r);group.append(r)
            s=summarize_validation(group,expected)
            s['M']=M;s['beta']=beta
            summaries.append(s)
            print(f'M={M} {method:8s} G={s["mean_green"]:.3f} +/- {s["replica_sem"]:.3f} '
                  f'exact_K1={expected:.3f} CPU={s["mean_cpu_seconds"]:.2f}s '
                  f'ESS/CPU={s["mean_ess_per_cpu"]:.1f} [{s["status"]}]')
    csv_write(out/'pilots.csv',pilot);csv_write(out/'replicas.csv',runs)
    csv_write(out/'summary.csv',summaries);csv_write(out/'spectral_reference.csv',refs)
    with (out/'protocol.json').open('w',encoding='utf-8') as f:
        json.dump({'model':'N=1->2 3D continuum canonical addition worm',
          'beta':beta,'cutoff':cutoff,'height':height,'links':grids,'cpu_target_per_chain':target,
          'replicas':count,'baseline_and_exchange':'both evaluated independently, all cold samples only',
          'limitations':'CPU budget approximate; K=1 finite basis; no claim of convergence; autocorrelation may be underestimated'},f,indent=2)
    fig,ax=plt.subplots(figsize=(7,4))
    for method in ('baseline','exchange'):
        ss=sorted((s for s in summaries if s['method']==method),key=lambda x:x['M'])
        ax.errorbar([s['M'] for s in ss],[s['mean_green'] for s in ss],
                    yerr=[s['replica_sem'] for s in ss],marker='o',capsize=3,label=method)
    ax.axhline(refs[0]['reference'],linestyle='--',color='gray',label='spectral K=1')
    ax.set(xlabel='Number of imaginary-time links (beta fixed)',ylabel='G0(beta/2)',title='Held-out, pilot-calibrated CPU budgets')
    ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(out/'accuracy.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    for method in ('baseline','exchange'):
        ss=sorted((s for s in summaries if s['method']==method),key=lambda x:x['M'])
        ax.plot([s['M'] for s in ss],[s['mean_ess_per_cpu'] for s in ss],marker='o',label=method)
    ax.set(xlabel='Number of imaginary-time links',ylabel='Ratio ESS / process CPU second',title='Actual measured CPU efficiency')
    ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(out/'efficiency.png',dpi=140);plt.close(fig)
    return pilot,runs,summaries

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--quick',action='store_true');p.add_argument('--output',default='outputs/bosons/equal_cpu_worm')
    p.add_argument('--cpu-seconds',type=float);p.add_argument('--replicas',type=int)
    p.add_argument('--links',nargs='+',type=int)
    a=p.parse_args();run_experiment(a.output,quick=a.quick,cpu_seconds=a.cpu_seconds,replicas=a.replicas,links=a.links)
if __name__=='__main__':main()
