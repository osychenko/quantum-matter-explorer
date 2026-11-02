"""Patch 28: CPU-calibrated held-out comparison for N=1 -> 2 continuum worms.

python3 -m examples.bosons.run_equal_cpu_worm --quick
python3 -m examples.bosons.run_equal_cpu_worm --cpu-seconds 12 --replicas 4
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from multiprocessing import get_context
import time
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



def _validation_task(task):
    """Picklable independent worker; all task inputs and the RNG are local."""
    method, cell, potential, base, plan, seed, M, beta, reference = task
    result = run_validation(method, cell, potential, base, plan, seed=seed)
    result.update(M=M, beta=beta, spectral_reference=reference)
    return result


def _validation_tasks(method, cell, potential, base, plan, M, beta, reference, replicas):
    """Use Patch 28's unchanged, disjoint deterministic validation seeds."""
    return [
        (method, cell, potential, base, plan,
         300000 + M*10000 + (0 if method == 'baseline' else 1000) + i,
         M, beta, reference)
        for i in range(replicas)
    ]


def _map_validations(tasks, workers, pool=None):
    """Ordered results in both serial and multiprocessing execution."""
    if workers == 1:
        return list(map(_validation_task, tasks))
    if pool is not None:
        return list(pool.map(_validation_task, tasks))
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as executor:
        return list(executor.map(_validation_task, tasks))


def run_experiment(output='outputs/bosons/equal_cpu_worm',*,quick=False,
                   cpu_seconds=None,replicas=None,links=None,workers=1,pilot_steps=None):
    if type(workers) is not int or not 1 <= workers <= 64:
        raise ValueError('workers must be an integer in [1,64]')
    if pilot_steps is not None and (type(pilot_steps) is not int or pilot_steps < 8):
        raise ValueError('pilot_steps must be an integer >=8')
    pilot_size = (12 if quick else 400) if pilot_steps is None else pilot_steps
    wall_start = time.perf_counter()
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
    # A single process pool is reused across every method/grid to avoid paying
    # worker startup costs repeatedly. Pilot CPU calibration stays serial.
    context = nullcontext(None) if workers == 1 else ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn'))
    with context as pool:
        for M in grids:
            # K=1 is a finite-basis reference, NOT an exact continuum result.
            ref=TwoBosonAdditionReference(cell_length=cell.length,height=height,cutoff=cutoff,k_max=1,n_grid=16)
            expected=float(ref.measure(beta,[beta/2],((0,0,0),)).green[0,0])
            refs.append({'M':M,'beta':beta,'k_max':1,'reference':expected})
            base=ReplicaExchangeConfig(n_links=M,time_step=beta/M,
                 n_steps=200,burn_in=40,thin=2,sector_fugacity=.9,
                 hot_interaction_fraction=.25,swap_probability=.3)
            for method in ('baseline','exchange'):
                pilot_seed=100000+M*1000+(0 if method=='baseline' else 100)
                plan=plan_budget(method,cell,pot,base,target_seconds=target,
                    pilot_steps=pilot_size,seed=pilot_seed,
                    min_validation_steps=160 if quick else 800,
                    max_validation_steps=600 if quick else 100000)
                pilot.append({'M':M,**asdict(plan)})
                tasks=_validation_tasks(method,cell,pot,base,plan,M,beta,expected,count)
                group=_map_validations(tasks,workers,pool)
                runs.extend(group)
                s=summarize_validation(group,expected)
                s['M']=M;s['beta']=beta
                summaries.append(s)
                print(f'M={M} {method:8s} G={s["mean_green"]:.3f} +/- {s["replica_sem"]:.3f} '
                      f'exact_K1={expected:.3f} CPU={s["mean_cpu_seconds"]:.2f}s '
                      f'ESS/CPU={s["mean_ess_per_cpu"]:.1f} [{s["status"]}]', flush=True)
    csv_write(out/'pilots.csv',pilot);csv_write(out/'replicas.csv',runs)
    csv_write(out/'summary.csv',summaries);csv_write(out/'spectral_reference.csv',refs)
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
    total_wall_seconds=time.perf_counter()-wall_start
    total_validation_cpu_seconds=float(sum(r['cpu_seconds'] for r in runs))
    with (out/'protocol.json').open('w',encoding='utf-8') as f:
        json.dump({'model':'N=1->2 3D continuum canonical addition worm',
          'beta':beta,'cutoff':cutoff,'height':height,'links':grids,'cpu_target_per_chain':target,
          'replicas':count,'workers':workers,'pilot_steps':pilot_size,
          'total_wall_seconds':total_wall_seconds,
          'total_validation_cpu_seconds':total_validation_cpu_seconds,
          'total_pilot_cpu_seconds':float(sum(p['pilot_seconds'] for p in pilot)),
          'baseline_and_exchange':'both evaluated independently, all cold samples only',
          'limitations':'CPU budget approximate; K=1 finite basis; no claim of convergence; autocorrelation may be underestimated'},f,indent=2)
    print(f'Workers={workers} | total wall time={total_wall_seconds:.2f}s | '
          f'validation CPU sum={total_validation_cpu_seconds:.2f}s',flush=True)
    return pilot,runs,summaries

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--quick',action='store_true');p.add_argument('--output',default='outputs/bosons/equal_cpu_worm')
    p.add_argument('--cpu-seconds',type=float);p.add_argument('--replicas',type=int)
    p.add_argument('--links',nargs='+',type=int)
    p.add_argument('--workers',type=int,default=1,
                   help='Independent validation processes (default: 1)')
    p.add_argument('--pilot-steps',type=int,
                   help='Independent CPU calibration steps (default: 400, or 12 in --quick)')
    a=p.parse_args();run_experiment(a.output,quick=a.quick,cpu_seconds=a.cpu_seconds,replicas=a.replicas,links=a.links,workers=a.workers,pilot_steps=a.pilot_steps)
if __name__=='__main__':main()
