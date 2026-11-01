"""Patch 18: normalized canonical Bose worm-sector g1 and momentum occupations.

    python3 -m examples.bosons.run_worm --quick
    python3 -m examples.bosons.run_worm

A finite-temperature primitive-action calculation, NOT a pure projector and
NOT a ground-state condensate-fraction or superfluid-fraction measurement.
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
from qmc.worm import CanonicalWorm, WormConfig, canonical_free_occupations


MODES = ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1))


def _write_csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mean_sem(values):
    arr = np.asarray(values, dtype=float)
    if len(arr) < 2 or not np.all(np.isfinite(arr)):
        raise ValueError("at least two finite independent replica estimates required")
    return float(arr.mean()), float(arr.std(ddof=1)/np.sqrt(len(arr)))


def run_experiment(output_dir="outputs/bosons/worm", *, quick=False):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cell = CubicCell(2.8)
    n_particles, cutoff, D, beta = 2, 1.1, 0.5, 1.8
    heights = (0., 2.)
    taus = (0.3,) if quick else (0.3, 0.225)
    replicas = 2 if quick else 3
    base = WormConfig(
        n_beads=6, time_step=0.3, n_steps=4500 if quick else 8500,
        burn_in=700 if quick else 1400, thin=2,
        seed=2026, bridge_lengths=(2,3), swap_probability=.25,
        bridge_probability=.2, refresh_probability=.25,
        translate_probability=.05, toggle_probability=.45,
        gibbs_probability=.85, local_shift_probability=.15,
        sector_fugacity=1., radial_bins=6,
    )
    rep_rows, sum_rows, momenta_rows, radial_rows = [], [], [], []
    for it, tau in enumerate(taus):
        beads = int(round(beta/tau))
        if not np.isclose(beads*tau, beta) or beads<4:
            raise ValueError("time-step values must preserve beta")
        exact = canonical_free_occupations(cell, n_particles, beta, MODES, D)
        for ih, height in enumerate(heights):
            occs, closed, g1s, ratios, effects, exchange = [], [], [], [], [], []
            for ir in range(replicas):
                seed = base.seed + it*1_000_000 + ih*100_000 + ir*1000
                cfg = replace(base, n_beads=beads, time_step=float(tau), seed=seed)
                potential = SoftCorePairPotential(cell, height=height, cutoff=cutoff)
                sampler = CanonicalWorm(cell, potential, n_particles, cfg,
                                        kinetic_coefficient=D, indices=MODES)
                res = sampler.run()
                occs.append(res.occupation_fractions)
                closed.append(res.closed_occupation_fractions)
                g1s.append(res.g1_radial)
                ratios.append(res.n0_fraction)
                effects.append(res.sector_effective_samples)
                exchange.append(res.closed_mean_exchange)
                rep_rows.append(dict(
                    height=height, tau=tau, beta=beta, replica=ir, seed=seed,
                    n_open=res.n_open, n_closed=res.n_closed,
                    n0_worm=res.n0_fraction,
                    n0_closed=res.closed_occupation_fractions[0],
                    free_exact_n0=exact[0] if height==0 else "",
                    sector_effective_samples=res.sector_effective_samples,
                    mean_exchange=res.closed_mean_exchange,
                    accept_open=res.acceptance["open"],
                    accept_close=res.acceptance["close"],
                    accept_gibbs=res.acceptance["sector_gibbs"],
                    accept_refresh=res.acceptance["refresh"],
                    accept_swap=res.acceptance["swap"],
                ))
            a = np.asarray(occs)
            b = np.asarray(closed)
            g = np.asarray(g1s)
            mu, se = _mean_sem(ratios)
            cmu, cse = _mean_sem(b[:,0])
            sum_rows.append(dict(height=height,tau=tau,beta=beta,replicas=replicas,
                                 n0_worm=mu,n0_worm_sem=se,
                                 n0_closed=cmu,n0_closed_sem=cse,
                                 free_exact_n0=exact[0] if height==0 else "",
                                 mean_sector_ess=float(np.mean(effects)),
                                 mean_exchange=float(np.mean(exchange))))
            for im, (index, val, cval) in enumerate(zip(MODES, a.T, b.T)):
                m, err = _mean_sem(val)
                cm, cerr = _mean_sem(cval)
                momenta_rows.append(dict(height=height,tau=tau,
                    kx=index[0],ky=index[1],kz=index[2],
                    worm=m,worm_sem=err,closed=cm,closed_sem=cerr,
                    free_exact=float(exact[im]) if height==0 else ""))
            radii = res.radius
            for j, r in enumerate(radii):
                m, err = _mean_sem(g[:,j])
                radial_rows.append(dict(height=height,tau=tau,radius=float(r),
                                        g1=m,g1_sem=err))
            print(f"height={height:g} tau={tau:g}: n0/N worm={mu:.4f} ± {se:.4f}; "
                  f"closed={cmu:.4f} ± {cse:.4f}; "
                  f"free exact={exact[0]:.4f}" if height==0 else
                  f"height={height:g} tau={tau:g}: n0/N worm={mu:.4f} ± {se:.4f}; "
                  f"closed={cmu:.4f} ± {cse:.4f} (interacting, no exact reference)")
    _write_csv(output/"replicas.csv", rep_rows)
    _write_csv(output/"summary.csv", sum_rows)
    _write_csv(output/"momenta.csv", momenta_rows)
    _write_csv(output/"g1_radial.csv", radial_rows)
    with (output/"protocol.json").open("w",encoding="utf-8") as f:
        json.dump(dict(model="N=2 canonical finite-T 3D bosons, primitive action",
                       cell_length=cell.length,cutoff=cutoff,D=D,beta=beta,
                       heights=heights,taus=taus,replicas=replicas,modes=MODES,
                       config=asdict(base),
                       note="Z/G sector ratio normalized by known eta; finite-T not pure PIGS; "
                            "radial histogram has strong variance at small r; "
                            "free benchmark exact at any tau; interacting primitive tau bias"),f,indent=2)
    fig,ax=plt.subplots(figsize=(7.4,4.5))
    for height in heights:
        rows=sorted((r for r in sum_rows if r["height"]==height),key=lambda r:r["tau"])
        ax.errorbar([x["tau"] for x in rows],[x["n0_worm"] for x in rows],
                    yerr=[x["n0_worm_sem"] for x in rows],fmt="o-",capsize=3,
                    label=f"worm height={height:g}")
        ax.errorbar([x["tau"] for x in rows],[x["n0_closed"] for x in rows],
                    yerr=[x["n0_closed_sem"] for x in rows],fmt="s--",capsize=3,
                    label=f"closed height={height:g}")
    ax.axhline(canonical_free_occupations(cell,n_particles,beta,MODES,D)[0],
               color="gray",ls=":",label="exact ideal Bose gas")
    ax.set(xlabel="Primitive timestep τ",ylabel="Finite-T k=0 occupation n0/N",
           title="Normalized canonical worm and closed-link estimates")
    ax.grid(alpha=.25);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(output/"occupation.png",dpi=160);plt.close(fig)
    fig, ax=plt.subplots(figsize=(7.2,4.5))
    for height in heights:
        rows=[r for r in momenta_rows if r["height"]==height and r["tau"]==taus[0]]
        x=np.arange(len(rows))
        ax.errorbar(x+.08*(height/2-.5),[r["worm"] for r in rows],
                    yerr=[r["worm_sem"] for r in rows],fmt="o",capsize=3,
                    label=f"height={height:g}")
    ref=[r["free_exact"] for r in momenta_rows if r["height"]==0 and r["tau"]==taus[0]]
    ax.plot(range(len(ref)),ref,"k_",markersize=18,label="exact free")
    ax.set_xticks(range(len(MODES)),[str(x) for x in MODES])
    ax.set(xlabel="Reciprocal index",ylabel="n(k)/N",
           title="Finite-T bosonic momentum occupations")
    ax.grid(alpha=.22);ax.legend()
    fig.tight_layout();fig.savefig(output/"momentum.png",dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.2,4.5))
    for height in heights:
        rr=[r for r in radial_rows if r["height"]==height and r["tau"]==taus[0]]
        ax.errorbar([r["radius"] for r in rr],[r["g1"] for r in rr],
                    yerr=[r["g1_sem"] for r in rr],fmt="o-",capsize=3,
                    label=f"height={height:g}")
    ax.axhline(1,color="gray",ls=":")
    ax.set(xlabel="Periodic radial separation r",ylabel="Finite-T g1(r)",
           title="Worm-sector radial coherence (noisy shell histogram)")
    ax.grid(alpha=.25);ax.legend();fig.tight_layout()
    fig.savefig(output/"g1_radial.png",dpi=160);plt.close(fig)
    print("Saved Patch 18 results to",output)
    return rep_rows,sum_rows,momenta_rows,radial_rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick",action="store_true")
    parser.add_argument("--output",default="outputs/bosons/worm")
    args=parser.parse_args()
    run_experiment(args.output,quick=args.quick)


if __name__=="__main__":
    main()
