"""Patch 31 diagnostic gate: inspect Patch 30 CSV and optional per-chain traces.

python3 -m examples.bosons.run_engine_readiness \\
  --replicas outputs/bosons/equal_cpu_trace_30/replicas.csv \\
  --traces outputs/bosons/equal_cpu_trace_30/traces \\
  --output outputs/bosons/engine_readiness_31
"""
from __future__ import annotations
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from qmc.engine_readiness import (_records,assess_preset,compare_methods,apply_cross_checks,chain_influence_report)
from qmc.worm_trace_diagnostics import load_trace


def _write_csv(path, rows):
    if not rows:return
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def run_readiness(replicas, *, traces=None,output='outputs/bosons/engine_readiness_31',
                  reference=1.6910255956087066):
    with Path(replicas).open(newline='',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    groups=_records(rows)
    trace_groups=defaultdict(list)
    if traces is not None:
        files=sorted(Path(traces).glob('*.npz'))
        if not files:
            raise ValueError('requested trace directory is empty; cannot certify trace diagnostics')
        for p in files:
            t=load_trace(p);trace_groups[(t['M'],t['method'])].append(t)
        if set(trace_groups)!=set(groups):
            raise ValueError('CSV and traces cover different grid/method groups')
    reports=[];chains=[]
    for key,group in sorted(groups.items()):
        reports.append(assess_preset(group,reference=reference,traces=trace_groups.get(key)))
        chains.extend(chain_influence_report(group,traces=trace_groups.get(key)))
    comparison=compare_methods(reports)
    apply_cross_checks(reports,comparison)
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    _write_csv(out/'preset_readiness.csv',reports)
    _write_csv(out/'method_consistency.csv',comparison)
    _write_csv(out/'chain_influence.csv',chains)
    blocked=[{'M':r['M'],'method':r['method'],'reasons':r['blocking_flags']}
             for r in reports if r['status']=='blocked']
    decision={'version':'patch31', 'scope':'N=1->2 midpoint G0, beta 2.4 only',
              'ready_for_engine_freeze':False,
              'reason':'This diagnostic gate never auto-certifies the engine; Patch 32 requires independent physics and product gates',
              'blocked_presets':blocked,
              'candidate_presets':[{'M':r['M'],'method':r['method']} for r in reports if r['status']=='candidate'],
              'needs_evidence':[{'M':r['M'],'method':r['method']} for r in reports if r['status']=='needs_evidence'],
              'limits':{'max_reference_z':2,'max_split_rhat':1.05,'max_relative_sem':0.05,
                        'min_reported_ess':100,'min_replicas':8},
              'notes':['Spectral K=2 is a finite-basis reference, not a rigorous continuum value',
                       'Rhat is rank-free and not sufficient to certify convergence',
                       'Comparisons of methods assume independent seeds',
                       'Candidate means cleared these screens, not proven scientifically correct']}
    (out/'decision.json').write_text(json.dumps(decision,indent=2),encoding='utf-8')
    lines=['# Patch 31 — Engineering readiness evidence','',
           'This is a screening report, not a proof or full engine certification.','',
           '| Grid | Method | Estimate ± replica SEM | z(reference) | Status | Reasons |',
           '|---:|---|---:|---:|---|---|']
    for r in reports:
        lines.append(f"| {r['M']} | {r['method']} | {r['mean_green']:.4f} ± {r['replica_sem']:.4f} | {r['z_reference']:.2f} | {r['status']} | {r['blocking_flags']}; {r['evidence_warnings']} |")
    lines.extend(['','## Cross-method consistency','',
                  '| Grid | Difference in independent-replica SEM units | Decision |',
                  '|---:|---:|---|'])
    for c in comparison:
        lines.append(f"| {c['M']} | {c['disagreement_z']:.2f} | {c['flag']} |")
    lines.extend(['','## Next decision','',
                  'Do not enable blocked presets as certified educational calculations.',
                  'Review full trace analysis and targeted independent validation before Patch 32.',
                  'MVP interface prototyping can begin with clearly labeled controls and precomputed demonstrations.'])
    (out/'readiness_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    for r in reports:
        print(f"M={r['M']:2} {r['method']:8} G={r['mean_green']:.4f}±{r['replica_sem']:.4f} "
              f"z={r['z_reference']:.2f} status={r['status']} "
              f"[{r['blocking_flags']};{r['evidence_warnings']}]")
    return reports,comparison,decision


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--replicas',required=True,help='replicas.csv from a single independent benchmark')
    p.add_argument('--traces',help='optional compressed traces directory; required for trace-based readiness')
    p.add_argument('--output',default='outputs/bosons/engine_readiness_31')
    a=p.parse_args()
    run_readiness(a.replicas,traces=a.traces,output=a.output)

if __name__=='__main__':main()
