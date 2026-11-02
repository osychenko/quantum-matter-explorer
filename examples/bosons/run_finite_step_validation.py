"""Patch 29: projected primitive transfer reference and conservative CSV audit.

python3 -m examples.bosons.run_finite_step_validation --summary outputs/bosons/equal_cpu_convergence_282/summary.csv --replicas outputs/bosons/equal_cpu_convergence_282/replicas.csv
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from qmc.finite_step_validation import ProjectedPrimitiveReference, compare_replicas, continuum_fit


def read_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        return
    with open(path,'w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def run_analysis(summary_path, replicas_path, *, output='outputs/bosons/finite_step_validation',
                 k_max=(1,2), n_grid=16, fit=True):
    summaries=read_csv(summary_path)
    replicas=read_csv(replicas_path)
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    grids=sorted({int(r['M']) for r in summaries})
    betas={float(r['beta']) for r in summaries}
    if len(betas)!=1:
        raise ValueError('all grids must share a single beta')
    beta=betas.pop()
    refs=[]; audit=[]
    for k in k_max:
        r=ProjectedPrimitiveReference(k_max=k,n_grid=n_grid)
        for M in grids:
            value=r.green_midpoint(beta=beta,links=M)
            refs.append(dict(M=M,beta=beta,k_max=k,projected_primitive=value,
                             reference_kind='P exp(-tau*K/2) P exp(-tau*V) P exp(-tau*K/2) P'))
    refmap={(int(r['M']),int(r['k_max'])):float(r['projected_primitive']) for r in refs}
    for row in summaries:
        M=int(row['M']); k=max(k_max)
        result=compare_replicas(replicas, reference=refmap[M,k],method=row['method'],links=M)
        result['projected_basis_k']=k
        result['basis_gap']=refmap[M,2]-refmap[M,1] if set((1,2)).issubset(k_max) else float('nan')
        result['projected_primitive_difference']=result['mean_green']-refmap[M,k]
        audit.append(result)
    fits=[]
    if fit:
        for method in ('baseline','exchange'):
            for power in (1,2):
                if len(grids)>=3:
                    fits.append(continuum_fit(summaries,method=method,power=power))
                if len(grids)>=4:
                    fits.append(continuum_fit(summaries,method=method,power=power,exclude=(min(grids),)))
    write_csv(out/'projected_primitive.csv',refs)
    write_csv(out/'replica_diagnostics.csv',audit)
    write_csv(out/'continuum_fits.csv',fits)
    (out/'protocol.json').write_text(json.dumps(dict(beta=beta,grids=grids,k_max=k_max,n_grid=n_grid,
        limitations=['Finite k projects at every transfer step; NOT exact coordinate-space primitive worm target',
                     'No full trajectory traces: cannot compute Rhat or block-length sweeps retroactively',
                     'Continuum fits are exploratory and depend on model power and grid independence'],
        outputs=['projected_primitive.csv','replica_diagnostics.csv','continuum_fits.csv']),indent=2))
    for a in audit:
        print(f"M={a['M']} {a['method']}: G={a['mean_green']:.4f} +/- {a['replica_sem']:.4f} "
              f"projected_K{a['projected_basis_k']}={a['reference']:.4f} "
              f"z={a['discrepancy_sigma']:.2f} 16/8={a['block_16_over_8_median']:.2f}")
    return refs,audit,fits


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary',required=True)
    p.add_argument('--replicas',required=True)
    p.add_argument('--output',default='outputs/bosons/finite_step_validation')
    p.add_argument('--k-max',nargs='+',type=int,default=[1,2])
    p.add_argument('--n-grid',type=int,default=16)
    p.add_argument('--no-fit',action='store_true')
    a=p.parse_args()
    run_analysis(a.summary,a.replicas,output=a.output,k_max=tuple(a.k_max),n_grid=a.n_grid,fit=not a.no_fit)
if __name__=='__main__':
    main()
