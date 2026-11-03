"""Patch 30: analyze compressed Patch 28/30 worm traces (no new MCMC runs).

python3 -m examples.bosons.run_worm_trace_diagnostics \
    --traces outputs/bosons/equal_cpu_trace_30/traces \
    --output outputs/bosons/worm_trace_audit_30
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from qmc.worm_trace_diagnostics import (load_trace, group_diagnostics,
    blocking_curve, autocorrelation_curve, running_ratio)


def _csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer=csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_analysis(traces, *, output='outputs/bosons/worm_trace_audit_30'):
    directory=Path(traces)
    paths=sorted(directory.glob('*.npz'))
    if not paths:
        raise ValueError('no .npz trace files found')
    out=Path(output)
    out.mkdir(parents=True,exist_ok=True)
    groups=defaultdict(list)
    for path in paths:
        t=load_trace(path)
        groups[(t['M'],t['method'])].append(t)
    summaries=[]; blocks=[]; acf_rows=[]
    fig_running,ax_running=plt.subplots(figsize=(8,4.5))
    fig_blocks,ax_blocks=plt.subplots(figsize=(8,4.5))
    fig_acf,ax_acf=plt.subplots(figsize=(8,4.5))
    for (M,method),chains in sorted(groups.items()):
        result=group_diagnostics(chains)
        summaries.append(result)
        label=f'M={M} {method}'
        for t in chains:
            for b in blocking_curve(t):
                blocks.append(dict(method=method,M=M,seed=t['seed'],**b))
            # Common pooled estimate for comparable cross-chain influence.
            theta=result['pooled_green']
            inf=t['scale']*t['numerator']-theta*t['denominator']
            ac=autocorrelation_curve(inf)
            acf_rows.append(dict(method=method,M=M,seed=t['seed'],
                n=len(inf),tau_int=ac['tau_int'],acf_ess=ac['ess'],
                window=ac['window'],warning=ac['warning']))
        # Display only a few chains for legibility; ALL chains contribute statistics.
        for t in sorted(chains,key=lambda r:r['seed'])[:3]:
            run=running_ratio(t)
            step=max(1,len(run)//350)
            ax_running.plot(np.arange(step-1,len(run),step),run[step-1::step],
                alpha=.5,label=f'{label} seed={t["seed"]}')
        ss=[r for r in blocks if r['method']==method and r['M']==M]
        by_length=defaultdict(list)
        for row in ss:
            by_length[int(row['blocks'])].append(row['sem'])
        grid=sorted(by_length)
        if grid:
            ax_blocks.plot([len(chains[0]['numerator'])/b for b in grid],
                [np.median(by_length[b]) for b in grid],marker='o',label=label)
        representative=sorted(chains,key=lambda r:r['seed'])[0]
        pooled=result['pooled_green']
        inf=representative['scale']*representative['numerator']-pooled*representative['denominator']
        ac=autocorrelation_curve(inf)
        ax_acf.plot(ac['lags'][:min(160,len(ac['lags']))],
                    ac['rho'][:min(160,len(ac['rho']))],label=label)
        print(f'{label}: G={result["mean_green"]:.5f} +/- {result["replica_sem"]:.5f} '
              f'Rhat(influence)={result["split_rhat_influence"]:.3f} '
              f'median_tau_int={result["median_tau_int"]:.1f}',flush=True)
    ax_running.set(xlabel='Production measurement index',ylabel='Running G0(beta/2)',
                   title='Running ratio estimates (up to 3 displayed chains per group)')
    ax_running.legend(fontsize=7);ax_running.grid(alpha=.2)
    fig_running.tight_layout();fig_running.savefig(out/'running_green.png',dpi=150)
    ax_blocks.set(xlabel='Observations per contiguous block',ylabel='Median ratio jackknife SEM',
                  title='Block-length sensitivity (not a convergence certificate)')
    ax_blocks.set_xscale('log');ax_blocks.legend(fontsize=8);ax_blocks.grid(alpha=.2)
    fig_blocks.tight_layout();fig_blocks.savefig(out/'blocking_curves.png',dpi=150)
    ax_acf.set(xlabel='Lag in recorded measurements',ylabel='Autocorrelation of pooled-ratio influence',
               title='Representative autocorrelation curves')
    ax_acf.legend(fontsize=8);ax_acf.grid(alpha=.2)
    fig_acf.tight_layout();fig_acf.savefig(out/'autocorrelation.png',dpi=150)
    plt.close(fig_running);plt.close(fig_blocks);plt.close(fig_acf)
    _csv(out/'trace_summary.csv',summaries)
    _csv(out/'blocking_curves.csv',blocks)
    _csv(out/'autocorrelation.csv',acf_rows)
    (out/'protocol.json').write_text(json.dumps({
        'trace_source':str(directory),'files':len(paths),'groups':len(groups),
        'observable':'midpoint zero-momentum Rao-Blackwellized Green ratio',
        'rhat':'rank-free split Rhat of common-pooled-ratio influence; independent, equal target chains',
        'caveats':['Rhat near 1 is necessary but not sufficient',
                   'ACF windows may miss unresolved long tails',
                   'Longer chains may be needed to see a blocking plateau',
                   'Existing Patch 28 CSV files cannot reconstruct these traces']
    },indent=2),encoding='utf-8')
    return summaries,blocks,acf_rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--traces',required=True)
    parser.add_argument('--output',default='outputs/bosons/worm_trace_audit_30')
    args=parser.parse_args()
    run_analysis(args.traces,output=args.output)

if __name__=='__main__':
    main()
