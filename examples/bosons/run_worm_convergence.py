"""Patch 22: independent interacting spectral reference and Gibbs-sector checks.

    python3 -m examples.bosons.run_worm_convergence --quick
    python3 -m examples.bosons.run_worm_convergence

N_background=1, inserted N+1=2 soft-core bosons on a 3D periodic torus.
These are finite-temperature canonical particle-ADDITION Green functions,
not a pure ground-state OBDM or an estimate of condensate fraction.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qmc.bosons import SoftCorePairPotential
from qmc.cells import CubicCell
from qmc.green_spectral import TwoBosonAdditionReference
from qmc.reliable_spacetime_worm import ReliableContinuumSpacetimeWorm, ReliableSpacetimeConfig


def _csv(path, data):
    if not data:
        raise ValueError("cannot write an empty CSV")
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)


def _mean_sem(values):
    data = np.asarray(values, dtype=float)
    if data.ndim != 1 or len(data) < 2 or not np.all(np.isfinite(data)):
        raise ValueError("at least two finite independent replicas are required")
    return float(data.mean()), float(data.std(ddof=1)/np.sqrt(len(data)))


def _check_grid(betas, links, replicas, n_steps, burn_in):
    if (len(betas) == 0 or any(not np.isfinite(x) or x <= 0 for x in betas)
            or len(set(betas)) != len(betas)):
        raise ValueError("betas must be distinct finite positive values")
    if (not links or any(isinstance(x, bool) or not isinstance(x, int) or x < 4
                          for x in links) or len(set(links)) != len(links)):
        raise ValueError("links must be distinct integer values >= 4")
    if (isinstance(replicas, bool) or replicas < 2
            or isinstance(n_steps, bool) or n_steps < 60
            or isinstance(burn_in, bool) or not 0 <= burn_in < n_steps
            or (n_steps-burn_in)//3 < 4):
        raise ValueError("invalid replicas or production sampling counts")


def run_experiment(output_dir="outputs/bosons/worm_convergence", *, quick=False,
                   betas=None, links=None, replicas=None, n_steps=None):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    beta_grid = tuple(float(x) for x in (betas if betas is not None else
                                        ((2.4,) if quick else (1.6, 2.4))))
    link_grid = tuple(links if links is not None else ((4,) if quick else (4, 6)))
    n_rep = replicas if replicas is not None else (2 if quick else 3)
    steps = n_steps if n_steps is not None else (1500 if quick else 4200)
    burn_in = min(500 if quick else 900, steps//4)
    _check_grid(beta_grid, link_grid, n_rep, steps, burn_in)
    cell = CubicCell(2.8)
    D, cutoff = 0.5, 1.2
    modes = ((0, 0, 0), (1, 0, 0))
    heights = (0.0, 3.0)
    # Reuse the separately diagonalized Galerkin Hamiltonians across tau/beta.
    spectral = {height: {
        k: TwoBosonAdditionReference(cell_length=cell.length,
                                     kinetic_coefficient=D, cutoff=cutoff,
                                     height=height, k_max=k, n_grid=16)
        for k in (1, 2)} for height in heights}
    base = ReliableSpacetimeConfig(
        n_steps=steps, burn_in=burn_in, thin=3,
        sector_fugacity=0.9, toggle_probability=0.42,
        global_refresh_probability=0.30, ghost_refresh_probability=0.6,
        time_move_probability=0.45, endpoint_move_probability=0.20,
        exchange_bridge_probability=0.25,
    )
    reference_rows, replica_rows, summary_rows = [], [], []
    for ib, beta in enumerate(beta_grid):
        for im, M in enumerate(link_grid):
            tau = beta/M
            times = tau*np.arange(1, M)
            for ih, height in enumerate(heights):
                model = "ideal" if height == 0 else "repulsive"
                refs = {k: spectral[height][k].measure(beta, times, modes) for k in (1, 2)}
                for basis_k, ref in refs.items():
                    for jt, t in enumerate(times):
                        for ik, mode in enumerate(modes):
                            reference_rows.append(dict(beta=beta, tau=tau, n_links=M,
                                                       case=model, height=height,
                                                       separation=float(t), k_max=basis_k,
                                                       mode_x=mode[0], mode_y=mode[1], mode_z=mode[2],
                                                       green=float(ref.green[jt, ik])))
                curves, raw_curves = [], []
                effective = []
                length_effective = []
                for ir in range(n_rep):
                    seed = 2200 + ib*100000+im*10000+ih*1000+ir*137
                    config = replace(base, time_step=tau, n_links=M, seed=seed)
                    potential = SoftCorePairPotential(cell, height=height, cutoff=cutoff)
                    sampler = ReliableContinuumSpacetimeWorm(
                        cell, potential, 1, config, kinetic_coefficient=D, indices=modes)
                    t0 = time.process_time()
                    result = sampler.run()
                    seconds = time.process_time()-t0
                    curves.append(result.green)
                    raw_curves.append(result.green_raw)
                    effective.append(result.conditional_effective_samples)
                    length_effective.append(result.effective_samples_by_length)
                    for jt, t in enumerate(times):
                        for ik, mode in enumerate(modes):
                            replica_rows.append(dict(beta=beta, tau=tau, n_links=M,
                                case=model, height=height, replica=ir, seed=seed,
                                separation=float(t), mode_x=mode[0], mode_y=mode[1], mode_z=mode[2],
                                green_gibbs=float(result.green[jt, ik]),
                                green_raw=float(result.green_raw[jt, ik]),
                                spectral_k1=float(refs[1].green[jt, ik]),
                                spectral_k2=float(refs[2].green[jt, ik]),
                                n_closed=result.n_closed,
                                n_open_at_length=int(result.n_open_by_length[jt]),
                                mean_closed_probability=result.mean_closed_probability,
                                effective_closed=result.closed_effective_samples,
                                effective_conditional=result.conditional_effective_samples,
                                effective_length=float(result.effective_samples_by_length[jt]),
                                cpu_seconds=seconds,
                                accept_global_refresh=result.acceptance["global_refresh"],
                                ))
                curves=np.asarray(curves)
                raw_curves=np.asarray(raw_curves)
                for jt, t in enumerate(times):
                    for ik, mode in enumerate(modes):
                        mean, sem = _mean_sem(curves[:,jt,ik])
                        raw_mean, raw_sem = _mean_sem(raw_curves[:,jt,ik])
                        ref1 = float(refs[1].green[jt,ik])
                        ref2 = float(refs[2].green[jt,ik])
                        summary_rows.append(dict(beta=beta, tau=tau, n_links=M,
                            case=model, height=height, replicas=n_rep, separation=float(t),
                            mode_x=mode[0], mode_y=mode[1], mode_z=mode[2],
                            green_gibbs=mean, green_gibbs_sem=sem,
                            green_raw=raw_mean, green_raw_sem=raw_sem,
                            spectral_k1=ref1, spectral_k2=ref2,
                            basis_difference=ref2-ref1,
                            mc_minus_spectral=mean-ref2,
                            discrepancy_in_replica_sem=(float((mean-ref2)/sem)
                                                        if sem>0 else float("nan")),
                            mean_conditional_ess=float(np.mean(effective)),
                            mean_length_ess=float(np.mean(np.asarray(length_effective)[:,jt]))))
                chosen = next(x for x in summary_rows if x["case"] == model
                              and x["beta"]==beta and x["n_links"]==M
                              and np.isclose(x["separation"], tau*round(M/2))
                              and x["mode_x"]==0)
                if model == "ideal":
                    for point in (r for r in summary_rows if r["case"] == model
                                  and r["beta"]==beta and r["n_links"]==M):
                        ratio=point["discrepancy_in_replica_sem"]
                        if np.isfinite(ratio) and abs(ratio)>2:
                            print(f"WARNING: ideal reference differs by {ratio:.1f} replica SEM "
                                  f"(beta={beta:g}, M={M}, t={point['separation']:.3f}, "
                                  f"k={point['mode_x'],point['mode_y'],point['mode_z']}); "
                                  "sampling or replica SEM may be unreliable")
                if chosen["mean_length_ess"] < 20:
                    print(f"WARNING: short-chain length-conditioned ESS "
                          f"at beta={beta:g}, M={M}, t=beta/2: "
                          f"{chosen['mean_length_ess']:.1f}; do not infer convergence")
                print(f"{model:9s} beta={beta:.3f} M={M} tau={tau:.3f} "
                      f"G0(beta/2)={chosen['green_gibbs']:.4f} ± "
                      f"{chosen['green_gibbs_sem']:.4f}; "
                      f"spectral(K=2)={chosen['spectral_k2']:.4f}")
    _csv(out/"spectral_reference.csv", reference_rows)
    _csv(out/"replicas.csv", replica_rows)
    _csv(out/"summary.csv", summary_rows)
    with (out/"protocol.json").open("w", encoding="utf-8") as handle:
        json.dump({"model":"Canonical 1->2 boson addition G_k at finite beta",
          "cell_length":cell.length,"D":D,"cutoff":cutoff,"heights":heights,
          "betas":beta_grid,"n_links_grid":link_grid,"replicas":n_rep,
          "config":asdict(base),"spectral_k_max":[1,2],"spectral_n_grid":16,
          "limitations":"Finite spectral cutoff, primitive-action tau bias, correlated Markov chains, finite beta. Not ground-state OBDM or n0/N."},
          handle,indent=2)
    fig,axes = plt.subplots(1,2,figsize=(11,4.3))
    for ax,(model,height) in zip(axes, (("ideal",0.0),("repulsive",3.0))):
        for M in link_grid:
            pts=sorted((x for x in summary_rows if x["case"]==model
                        and x["beta"]==beta_grid[-1] and x["n_links"]==M
                        and x["mode_x"]==0 and x["mode_y"]==0 and x["mode_z"]==0),
                       key=lambda x:x["separation"])
            ax.errorbar([x["separation"] for x in pts],[x["green_gibbs"] for x in pts],
                        yerr=[x["green_gibbs_sem"] for x in pts],
                        marker="o",capsize=3,label=f"M={M} Gibbs")
            ax.plot([x["separation"] for x in pts],[x["spectral_k2"] for x in pts],
                    ls="--", alpha=.7, label=f"M={M} Galerkin K=2")
        ax.set(xlabel="Imaginary-time separation", ylabel=r"$G_{k=0}^{(+)}(t)$",
               title=f"{model.capitalize()}, beta={beta_grid[-1]:g}")
        ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle("Patch 22: sampled particle-addition G and independent spectral reference")
    fig.tight_layout();fig.savefig(out/"green_spectral.png",dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.3,4.5))
    for model in ("ideal","repulsive"):
        pts=sorted((x for x in summary_rows if x["case"]==model
                    and x["mode_x"]==0 and x["mode_y"]==0 and x["mode_z"]==0
                    and np.isclose(x["separation"],x["beta"]/2)),
                   key=lambda x:(x["beta"],x["tau"]))
        ax.errorbar([x["tau"] for x in pts],[x["green_gibbs"]-x["spectral_k2"] for x in pts],
                    yerr=[x["green_gibbs_sem"] for x in pts],marker="o",capsize=3,
                    label=model)
    ax.axhline(0,color="0.5",ls="--")
    ax.set(xlabel="Primitive-action timestep",ylabel="Monte Carlo minus spectral G0(beta/2)",
           title="Finite-beta and timestep diagnostic (not a controlled extrapolation)")
    ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(out/"bias_diagnostic.png",dpi=160);plt.close(fig)
    print("Saved Patch 22 results to",out)
    return replica_rows, summary_rows, reference_rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick",action="store_true")
    parser.add_argument("--output",default="outputs/bosons/worm_convergence")
    parser.add_argument("--steps",type=int)
    parser.add_argument("--replicas",type=int)
    args=parser.parse_args()
    run_experiment(args.output,quick=args.quick,n_steps=args.steps,replicas=args.replicas)


if __name__=="__main__":
    main()
