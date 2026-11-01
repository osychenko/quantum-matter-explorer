"""Patch 20: normalized Bose-Hubbard spacetime worm with independent endpoints.

    python3 -m examples.bosons.run_spacetime_worm --quick
    python3 -m examples.bosons.run_spacetime_worm

This small *lattice* benchmark uses exact dense transfer matrices. It is not
3D continuum PIMC, pure ground-state coherence, or scalable worm sampling.
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

from qmc.pure_projector import BoseHubbardRing
from qmc.spacetime_worm import ExactSpacetimeWorm, SpacetimeWormConfig


def _csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mean_sem(values):
    x = np.asarray(values, dtype=float)
    if len(x) < 2:
        raise ValueError("at least two replicas are required")
    return float(x.mean()), float(x.std(ddof=1)/np.sqrt(len(x)))


def lattice_free_occupation(n_bosons, n_sites, hopping, beta):
    """Independent ideal Bose gas canonical recursion for lattice k=0 occupation."""
    if n_bosons < 1 or n_sites < 3 or hopping <= 0 or beta <= 0:
        raise ValueError("invalid free lattice model parameters")
    eps = -2.0*hopping*np.cos(2*np.pi*np.arange(n_sites)/n_sites)
    z = np.zeros(n_bosons+1)
    z[0] = 1.
    for n in range(1, n_bosons+1):
        z[n] = sum(z[n-l]*np.exp(-l*beta*eps).sum() for l in range(1,n+1))/n
    occ = sum(z[n_bosons-l]*np.exp(-l*beta*eps[0]) for l in range(1,n_bosons+1))
    return float(occ/(n_bosons*z[n_bosons]))


def run_experiment(output_dir="outputs/bosons/spacetime_worm", *, quick=False):
    """Replicated finite-temperature G(k,t) with exact spectral references."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    beta = 1.2
    m_grid = (6,) if quick else (6, 8)
    replicas = 2 if quick else 3
    base = SpacetimeWormConfig(
        n_steps=4500 if quick else 30000,
        burn_in=600 if quick else 5000,
        thin=3 if quick else 4,
        toggle_probability=.35, time_move_probability=.4,
        space_move_probability=.25, fugacity=1.0,
    )
    models = (("free", 0.), ("interacting", 3.))
    n_sites, n_particles = 3, 2
    summary_rows, replica_rows, green_rows = [], [], []
    for model_no, (label, onsite_u) in enumerate(models):
        model = BoseHubbardRing(n_sites=n_sites, n_bosons=n_particles,
                                hopping=1., onsite_u=onsite_u)
        for m_no, m in enumerate(m_grid):
            cfg = replace(base, n_slices=m, time_step=beta/m)
            reference = ExactSpacetimeWorm(model, replace(cfg, seed=777))
            exact_by_sep = reference.exact_green()
            exact_gk0 = exact_by_sep.sum(axis=2)[:,0]  # translation symmetry
            exact_gk1 = np.sum(exact_by_sep[:,0,:]*np.cos(2*np.pi*np.arange(n_sites)/n_sites),axis=1)
            exact_occ = reference.exact_zero_momentum_fraction()
            if onsite_u == 0:
                analytical = lattice_free_occupation(n_particles,n_sites,1.,beta)
                if not np.isclose(exact_occ, analytical, atol=1e-11):
                    raise AssertionError("independent ideal-boson occupation mismatch")
            per_sep = []
            fractions = []
            acc = []
            for ir in range(replicas):
                seed = 2026000 + model_no*100000 + m_no*10000 + ir*100
                sampler = ExactSpacetimeWorm(model, replace(cfg, seed=seed))
                run = sampler.run()
                k0 = run.green.sum(axis=1)
                k1 = np.sum(run.green*np.cos(2*np.pi*np.arange(n_sites)/n_sites)[None,:], axis=1)
                fractions.append(run.observed_open_fraction)
                per_sep.append((k0,k1))
                acc.append(run.acceptance["open"])
                replica_rows.append(dict(model=label, n_sites=n_sites, n_particles=n_particles,
                                         onsite_u=onsite_u, beta=beta, n_slices=m, dt=beta/m,
                                         replica=ir, seed=seed,
                                         n_open=run.n_open, n_closed=run.n_closed,
                                         observed_open_fraction=run.observed_open_fraction,
                                         exact_open_fraction=run.exact_open_fraction,
                                         gk0_first=float(k0[0]), exact_gk0_first=float(exact_gk0[1]),
                                         gk1_first=float(k1[0]), exact_gk1_first=float(exact_gk1[1]),
                                         open_acceptance=run.acceptance["open"],
                                         close_acceptance=run.acceptance["close"],
                                         head_time_acceptance=run.acceptance["head_time"],
                                         tail_time_acceptance=run.acceptance["tail_time"]))
            p_mean,p_sem=_mean_sem(fractions)
            g0 = np.stack([x[0] for x in per_sep])
            g1 = np.stack([x[1] for x in per_sep])
            summary_rows.append(dict(model=label, n_slices=m, beta=beta, dt=beta/m,
                                     replicas=replicas, observed_open_fraction=p_mean,
                                     open_fraction_sem=p_sem, exact_open_fraction=reference.exact_open_fraction(),
                                     gk0_first=float(g0[:,0].mean()), gk0_first_sem=float(g0[:,0].std(ddof=1)/np.sqrt(replicas)),
                                     exact_gk0_first=float(exact_gk0[1]),
                                     exact_thermal_n0_fraction=exact_occ,
                                     mean_open_acceptance=float(np.mean(acc))))
            for s in range(1,m):
                mu0,sem0=_mean_sem(g0[:,s-1]);mu1,sem1=_mean_sem(g1[:,s-1])
                green_rows.append(dict(model=label, n_slices=m, beta=beta, dt=beta/m,
                                       time=s*beta/m, separation=s,
                                       gk0=mu0, gk0_replica_sem=sem0,
                                       exact_gk0=float(exact_gk0[s]),
                                       gk1=mu1, gk1_replica_sem=sem1,
                                       exact_gk1=float(exact_gk1[s]),
                                       exact_equal_time_gk0=1+n_particles*exact_occ))
            print(f"{label:12s} M={m:2d}  open={p_mean:.4f}±{p_sem:.4f} "
                  f"(exact {reference.exact_open_fraction():.4f}) "
                  f"G(k=0,dt)={g0[:,0].mean():.3f} "
                  f"(exact {exact_gk0[1]:.3f})")
    _csv(output/"replicas.csv",replica_rows)
    _csv(output/"summary.csv",summary_rows)
    _csv(output/"green.csv",green_rows)
    with (output/"protocol.json").open("w",encoding="utf-8") as f:
        json.dump(dict(model="canonical Bose-Hubbard ring with N vs N+1 spacetime worm sector",
                       n_sites=n_sites,n_particles=n_particles,beta=beta,
                       onsite_u_values={label:u for label,u in models},
                       n_slices=m_grid,replicas=replicas,base_config=asdict(base),
                       measurement="time-ordered particle-addition Green G(k,tau), not equal-time occupation",
                       limitations="finite temperature; exact dense lattice propagator; not continuum; endpoint-autocorrelation and finite-tau grids"),
                  f,indent=2)
    fig,axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for ax,(label,_) in zip(axes,models):
        for m in m_grid:
            pts=[r for r in green_rows if r["model"]==label and r["n_slices"]==m]
            ax.errorbar([r["time"] for r in pts],[r["gk0"] for r in pts],
                        yerr=[r["gk0_replica_sem"] for r in pts],
                        marker="o",capsize=3,label=f"MC M={m}")
            ax.plot([r["time"] for r in pts],[r["exact_gk0"] for r in pts],
                    "--",label=f"Exact M={m}")
        ax.set(xlabel="Imaginary-time separation",title=label)
        ax.grid(alpha=.25); ax.legend(fontsize=8)
    axes[0].set_ylabel(r"$G(k=0,\tau)$, particle-addition")
    fig.tight_layout();fig.savefig(output/"green_vs_time.png",dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.6,4.5))
    tick=np.arange(len(summary_rows))
    ax.errorbar(tick,[r["observed_open_fraction"] for r in summary_rows],
                yerr=[r["open_fraction_sem"] for r in summary_rows],
                fmt="o",capsize=4,label="Worm sector frequency")
    ax.scatter(tick,[r["exact_open_fraction"] for r in summary_rows],
               marker="x",s=70,label="Exact sector weight")
    ax.set_xticks(tick,[f'{r["model"]}\nM={r["n_slices"]}' for r in summary_rows])
    ax.set(ylabel="P(open sector)",title="Extended-sector normalization check")
    ax.grid(alpha=.22);ax.legend()
    fig.tight_layout();fig.savefig(output/"sector_normalization.png",dpi=150);plt.close(fig)
    print("Saved Patch 20 results:",output)
    return replica_rows,summary_rows,green_rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick",action="store_true",help="Short two-state smoke test")
    parser.add_argument("--output",default="outputs/bosons/spacetime_worm")
    args=parser.parse_args()
    run_experiment(args.output,quick=args.quick)


if __name__=="__main__":
    main()
