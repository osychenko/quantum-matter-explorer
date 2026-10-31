"""Patch 17: finite-temperature bosonic exchange plus CPU-normalized PIGS tuning.

    python3 -m examples.bosons.run_boson_permutations --quick
    python3 -m examples.bosons.run_boson_permutations

The permutation sampler is finite-temperature, closed-worldline PIMC; the
bridge-efficiency study is separate zero-temperature-projection (open PIGS).
Neither is a pure interacting OBDM or an interacting condensate estimator.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qmc.bosons import SoftCoreJastrowTrial, SoftCorePairPotential
from qmc.cells import CubicCell
from qmc.boson_permutations import (
    BosonPermutationPaths, PermutationPathConfig, free_two_boson_exchange_probability,
)
from qmc.path_efficiency import EfficiencyConfig, compare_bridge_probabilities


def _csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mean_sem(values):
    vals = np.asarray(values, dtype=float)
    return float(vals.mean()), float(vals.std(ddof=1)/np.sqrt(len(vals)))


def run_experiment(output_dir="outputs/bosons/boson_permutations", *, quick=False):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cell = CubicCell(2.4)
    beta, tau = 1.6, 0.2
    n_beads = round(beta/tau)
    cutoff, height, amp = 1.0, 2.0, 0.3
    base = PermutationPathConfig(
        time_step=tau, n_beads=n_beads,
        n_steps=1600 if quick else 8000,
        burn_in=300 if quick else 1600,
        thin=4 if quick else 8,
        seed=2026,
    )
    replica_count = 2 if quick else 4
    exact_free = free_two_boson_exchange_probability(cell,beta)
    replica_rows, summary_rows = [], []
    for label, h in (("free",0.0),("soft_core",height)):
        pot = SoftCorePairPotential(cell,height=h,cutoff=cutoff)
        probs=[]
        potentials=[]
        for replica in range(replica_count):
            cfg=replace(base,seed=base.seed+100_000*(label=="soft_core")+97*replica)
            sampler=BosonPermutationPaths(cell,pot,2,cfg)
            result=sampler.run()
            p=float(result.exchange_indicator.mean())
            v=float(result.midpoint_potential.mean())
            probs.append(p)
            potentials.append(v)
            replica_rows.append(dict(
                ensemble="finite_temperature_closed",case=label,replica=replica,
                seed=cfg.seed,beta=result.beta,tau=tau,n_beads=n_beads,
                exchange_probability=p,midpoint_potential=v,
                longest_cycle_fraction=float(result.longest_cycle_fraction.mean()),
                accepted_swap=result.acceptance["swap"],
                accepted_bridge=result.acceptance["bridge"],
                accepted_refresh=result.acceptance["refresh"],
                n_measurements=len(result.exchange_indicator),
            ))
        pmean,psem=_mean_sem(probs)
        vmean,vsem=_mean_sem(potentials)
        summary_rows.append(dict(
            ensemble="finite_temperature_closed",case=label,beta=beta,
            probability=pmean,probability_sem=psem,
            exact_free_probability=exact_free if label=="free" else "",
            midpoint_potential=vmean,midpoint_potential_sem=vsem,
            replicas=replica_count))
        print(f"{label:9s} P(exchange)={pmean:.4f} +/- {psem:.4f}"+
              (f" (analytic free {exact_free:.4f})" if label=="free" else ""))

    # Separate open-path (PIGS) performance experiment. *Never* interpret
    # its statistics as the canonical finite-temperature exchange ensemble.
    efficiency_cfg=EfficiencyConfig(
        pilot_steps=220 if quick else 1000,
        validation_steps=320 if quick else 1800,
        burn_in=40 if quick else 180,
        thin=3 if quick else 4,
        pilot_replicas=2 if quick else 3,
        validation_replicas=2 if quick else 3,
        n_links=8, time_step=0.2, seed=8080,
    )
    perf_cell=CubicCell(3.5)
    perf_pot=SoftCorePairPotential(perf_cell,height=4.,cutoff=1.5)
    perf_trial=SoftCoreJastrowTrial(perf_cell,amplitude=.5,cutoff=1.5)
    pilot,validation,chosen=compare_bridge_probabilities(
        perf_cell,perf_pot,perf_trial,n_particles=2,cfg=efficiency_cfg)
    print(f"Pilot selected bridge probability {chosen:g} for held-out validation")
    _csv(output/"permutation_replicas.csv",replica_rows)
    _csv(output/"permutation_summary.csv",summary_rows)
    _csv(output/"efficiency_pilot.csv",pilot)
    _csv(output/"efficiency_validation.csv",validation)
    with (output/"protocol.json").open("w",encoding="utf-8") as f:
        json.dump(dict(
            permutation_config=asdict(base),permutation_replicas=replica_count,
            bosonic_ensemble="finite-temperature canonical PIMC; N=2; closed worldlines",
            efficiency_config=asdict(efficiency_cfg),
            efficiency_ensemble="separate fixed-beta primitive PIGS open-path experiment",
            pilot_selected_probability=chosen,
            limitations="sampling bias/autocorrelation, finite action timestep and basis; no pure OBDM or worm sector",
        ),f,indent=2)
    fig,ax=plt.subplots(figsize=(7,4))
    xs=np.arange(len(summary_rows))
    ax.errorbar(xs,[r["probability"] for r in summary_rows],
                yerr=[r["probability_sem"] for r in summary_rows],fmt="o",capsize=4)
    ax.axhline(exact_free,color="0.45",ls="--",label="Exact free two-boson reference")
    ax.set_xticks(xs,[r["case"] for r in summary_rows])
    ax.set(ylabel="Exchange-cycle probability",title=f"Canonical finite-T PIMC (beta={beta:g})")
    ax.grid(alpha=.2);ax.legend();fig.tight_layout()
    fig.savefig(output/"exchange_probability.png",dpi=150);plt.close(fig)

    fig,ax=plt.subplots(figsize=(7,4))
    ps=sorted({r["bridge_probability"] for r in pilot})
    y=[np.mean([r["effective_samples_per_cpu_second"] for r in pilot
                if r["bridge_probability"]==p]) for p in ps]
    ax.plot(ps,y,"o-",label="Pilot mean ESS/CPU second")
    for row in validation:
        ax.scatter(row["bridge_probability"],row["effective_samples_per_cpu_second"],
                   marker="x",color="tab:orange",label="Held-out chain" if row is validation[0] else None)
    ax.axvline(chosen,ls="--",color="0.5",label="Pilot-selected mixture")
    ax.set(xlabel="Bridge move probability",ylabel="Effective samples / CPU second",
           title="Open-path PIGS midpoint-potential mixing")
    ax.grid(alpha=.2);ax.legend();fig.tight_layout()
    fig.savefig(output/"cpu_efficiency.png",dpi=150);plt.close(fig)
    print("Saved Patch 17 outputs:",output)
    return replica_rows,summary_rows,pilot,validation,chosen


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick",action="store_true")
    parser.add_argument("--output",default="outputs/bosons/boson_permutations")
    args=parser.parse_args()
    run_experiment(args.output,quick=args.quick)


if __name__=="__main__":
    main()
