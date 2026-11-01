"""Patch 21: 3D interacting N=1->2 spacetime Green-function worm benchmark.

    python3 -m examples.bosons.run_continuum_spacetime_worm --quick
    python3 -m examples.bosons.run_continuum_spacetime_worm

This is canonical particle-ADDITION G_k(tau), not a condensate occupation,
zero-temperature ground-state Green function, or superfluid estimator.
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

from qmc.bosons import SoftCorePairPotential
from qmc.cells import CubicCell
from qmc.continuum_spacetime_worm import (
    ContinuumSpacetimeConfig, ContinuumSpacetimeWorm,
    exact_free_addition_green,
)


def _write_csv(path, rows):
    if not rows:
        raise ValueError("cannot export empty rows")
    with Path(path).open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _replica_summary(values):
    data=np.asarray(values,dtype=float)
    if data.ndim!=1 or len(data)<2 or not np.all(np.isfinite(data)):
        raise ValueError("need at least two finite independent replica estimates")
    return float(data.mean()),float(data.std(ddof=1)/np.sqrt(len(data)))


def run_experiment(output_dir="outputs/bosons/continuum_spacetime_worm", *, quick=False,
                   steps_override=None, replicas_override=None):
    out=Path(output_dir)
    out.mkdir(parents=True,exist_ok=True)
    cell=CubicCell(2.8)
    D,cutoff,beta=0.5,1.2,2.4
    n_links_grid=(4,) if quick else (4,6)
    replicas=2 if quick else 3
    if replicas_override is not None:
        if isinstance(replicas_override,bool) or not isinstance(replicas_override,int) or replicas_override<2:
            raise ValueError("replicas_override must be an integer >= 2")
        replicas=replicas_override
    modes=((0,0,0),(1,0,0),(0,1,0))
    cases=(("vacuum_free",0,0.0),("one_boson_free",1,0.0),
           ("one_boson_repulsive",1,3.0))
    base=ContinuumSpacetimeConfig(
        n_steps=2600 if quick else 5800,
        burn_in=600 if quick else 1200,
        thin=3 if quick else 3,
        sector_fugacity=0.9,
        toggle_probability=0.42,
        global_refresh_probability=0.30,
        time_move_probability=0.45,
        endpoint_move_probability=0.20,
        exchange_bridge_probability=0.25,
    )
    if steps_override is not None:
        if isinstance(steps_override,bool) or not isinstance(steps_override,int) or steps_override<100:
            raise ValueError("steps_override must be an integer >= 100")
        base=replace(base,n_steps=steps_override,burn_in=min(base.burn_in,steps_override//4))
    replica_rows=[]
    summary_rows=[]
    for ix,M in enumerate(n_links_grid):
        timestep=beta/M
        for ic,(name,n_background,height) in enumerate(cases):
            potential=SoftCorePairPotential(cell,height=height,cutoff=cutoff)
            measurements=[]
            closed=[]
            sector_ess=[]
            length_counts=[]
            exact=exact_free_addition_green(cell,n_background,beta,
                                             timestep*np.arange(1,M),modes,D) if height==0 else None
            for ir in range(replicas):
                seed=2026+ix*100000+ic*10000+ir*100
                cfg=replace(base,n_links=M,time_step=timestep,seed=seed)
                sim=ContinuumSpacetimeWorm(cell,potential,n_background,cfg,
                                          kinetic_coefficient=D,indices=modes)
                r=sim.run()
                measurements.append(r.green)
                closed.append(r.n_closed)
                sector_ess.append(r.sector_effective_samples)
                length_counts.append(r.n_open_by_length)
                for j,time in enumerate(r.imaginary_times):
                    for ik,mode in enumerate(modes):
                        replica_rows.append(dict(
                            case=name,n_background=n_background,potential_height=height,
                            beta=beta,step=timestep,n_links=M,replica=ir,seed=seed,
                            separation=float(time),mode_x=mode[0],mode_y=mode[1],mode_z=mode[2],
                            green=float(r.green[j,ik]),
                            exact_free=float(exact[j,ik]) if exact is not None else "",
                            n_closed=r.n_closed,n_open_at_length=int(r.n_open_by_length[j]),
                            sector_ess=r.sector_effective_samples,
                            open_acceptance=r.acceptances["open"],
                            close_acceptance=r.acceptances["close"],
                            time_acceptance=r.acceptances["time"],
                            global_refresh_acceptance=r.acceptances["global_refresh"],
                        ))
            matrix=np.asarray(measurements)
            for j in range(M-1):
                for ik,mode in enumerate(modes):
                    mean,sem=_replica_summary(matrix[:,j,ik])
                    summary_rows.append(dict(
                        case=name,n_background=n_background,potential_height=height,
                        beta=beta,step=timestep,n_links=M,replicas=replicas,
                        separation=(j+1)*timestep,mode_x=mode[0],mode_y=mode[1],mode_z=mode[2],
                        green=mean,green_replica_sem=sem,
                        exact_free=float(exact[j,ik]) if exact is not None else "",
                        reference_discrepancy_in_sem=(float((mean-exact[j,ik])/sem)
                                                     if exact is not None and sem>0 else ""),
                        mean_closed=float(np.mean(closed)),
                        mean_open_at_length=float(np.mean(np.asarray(length_counts)[:,j])),
                        mean_sector_ess=float(np.mean(sector_ess)),
                    ))
            indices0=[i for i,m in enumerate(modes) if m==(0,0,0)]
            estimate=float(np.mean(matrix[:,0,indices0[0]]))
            print(f"{name:22s} M={M:2d} tau={timestep:.3f} "
                  f"G0(tau)={estimate:.4f} "
                  + (f"(free exact {exact[0,indices0[0]]:.4f})" if exact is not None else "(interacting)"))
    for row in summary_rows:
        z=row["reference_discrepancy_in_sem"]
        if z!="" and abs(z)>2:
            print(f"WARNING free-reference deviation: {row['case']} M={row['n_links']} "
                  f"t={row['separation']:.3f} k=({row['mode_x']},{row['mode_y']},{row['mode_z']}) "
                  f"difference/replica-SEM={z:.1f}. Increase independent sampling; "
                  "replica SEM can underestimate slow-mixing uncertainty.")
    _write_csv(out/"replicas.csv",replica_rows)
    _write_csv(out/"summary.csv",summary_rows)
    with (out/"protocol.json").open("w",encoding="utf-8") as f:
        json.dump(dict(
            model="3D canonical particle-addition G_k(t), N=0/1 background, N+1=1/2 inserted",
            cell_length=cell.length,kinetic_coefficient=D,cutoff=cutoff,beta=beta,
            n_links_grid=n_links_grid,replicas=replicas,cases=cases,modes=modes,
            base_config=asdict(base),
            normalization="G_k(t)=G-sector phase sum / [eta*(1/(M-1))*Z-sector count]",
            limitations=("finite tau, finite beta, N<=1 background, correlated paths, "
                         "free-background only; interacting N=2 sector; "
                         "G(k,t) is particle-addition and not n_k; "
                         "symmetrized per-link kernel increases estimator variance")),f,indent=2)
    fig,axes=plt.subplots(1,2,figsize=(11,4.3))
    for case,_,height in cases:
        for M in n_links_grid:
            points=sorted((x for x in summary_rows if x["case"]==case and x["n_links"]==M
                           and x["mode_x"]==0 and x["mode_y"]==0 and x["mode_z"]==0),
                           key=lambda x:x["separation"])
            ax=axes[0] if height==0 else axes[1]
            ax.errorbar([r["separation"] for r in points],[r["green"] for r in points],
                        yerr=[r["green_replica_sem"] for r in points],
                        marker="o",capsize=2,label=f"{case}, M={M}")
            if height==0:
                ax.plot([r["separation"] for r in points],
                        [r["exact_free"] for r in points],ls="--",alpha=.6)
    axes[0].set_title("Exact free references (dashed)")
    axes[1].set_title("Interacting N=1 to 2; no exact curve")
    for ax in axes:
        ax.set_xlabel("Imaginary-time separation")
        ax.set_ylabel(r"Canonical addition $G_{k=0}(\tau)$")
        ax.grid(alpha=.22)
        ax.legend(fontsize=7)
    fig.suptitle("Continuum spacetime worm: finite-temperature particle addition")
    fig.tight_layout()
    fig.savefig(out/"green_vs_time.png",dpi=160)
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(7.5,4.5))
    for case,_,height in cases:
        if height:
            continue
        for M in n_links_grid:
            points=sorted((x for x in summary_rows if x["case"]==case and x["n_links"]==M
                           and (x["mode_x"],x["mode_y"],x["mode_z"])==(1,0,0)),
                           key=lambda x:x["separation"])
            ax.errorbar([r["separation"] for r in points],[r["green"] for r in points],
                        yerr=[r["green_replica_sem"] for r in points],
                        marker="o",capsize=2,label=f"{case}, M={M}")
            ax.plot([r["separation"] for r in points],
                    [r["exact_free"] for r in points],ls="--",alpha=.55)
    ax.set(xlabel="Imaginary-time separation",ylabel=r"Particle-addition $G_{(1,0,0)}(\tau)$",
           title="Nonzero momentum: stochastic and exact ideal-boson G")
    ax.grid(alpha=.22);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/"nonzero_momentum.png",dpi=160);plt.close(fig)
    print("Saved Patch 21 diagnostics to",out)
    return replica_rows,summary_rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick",action="store_true")
    parser.add_argument("--output",default="outputs/bosons/continuum_spacetime_worm")
    parser.add_argument("--steps",type=int,help="Override number of MCMC steps per replica")
    parser.add_argument("--replicas",type=int,help="Override number of independent replicas (>=2)")
    args=parser.parse_args()
    run_experiment(args.output,quick=args.quick,
                   steps_override=args.steps,replicas_override=args.replicas)


if __name__=="__main__":
    main()
