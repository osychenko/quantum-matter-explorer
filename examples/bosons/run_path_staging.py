"""Patch 16: staging/bisection mixing and interacting two-boson reference.

    python3 -m examples.bosons.run_path_staging --quick
    python3 -m examples.bosons.run_path_staging

Fixed-beta primitive open paths, without Bose permutation cycles or OBDM.
Quick mode only checks integration; it is not a convergence demonstration.
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
from qmc.interacting_path import PrimitiveInteractingPath
from qmc.path_staging import StagedPathConfig, StagedInteractingPath, autocorrelation_estimate
from qmc.two_boson_reference import TwoBosonRelativeReference


def _csv(path, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mean_sem(values):
    data = np.asarray(values, dtype=float)
    if len(data) < 2:
        raise ValueError("independent replica errors need >=2 replicas")
    return float(np.mean(data)), float(np.std(data, ddof=1) / np.sqrt(len(data)))


def run_experiment(output_dir="outputs/bosons/path_staging", *, quick=False):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cell = CubicCell(3.5)
    cutoff, height, D, amplitude, beta = 1.5, 4.0, 0.5, 0.5, 0.8
    steps = (0.2, 0.1) if quick else (0.2, 0.1, 0.05)
    replicas = 2 if quick else 4
    base = StagedPathConfig(n_steps=450 if quick else 3000,
                            burn_in=100 if quick else 600,
                            thin=3 if quick else 5,
                            refresh_probability=0.12, translate_probability=0.12,
                            bridge_probability=0.7,
                            bridge_lengths=(2, 4), move_sigma=0.40)
    potential = SoftCorePairPotential(cell, height=height, cutoff=cutoff)
    trial = SoftCoreJastrowTrial(cell, amplitude=amplitude, cutoff=cutoff)
    references = []
    for k_max in (1, 2):
        ref = TwoBosonRelativeReference(
            cell_length=cell.length, kinetic_coefficient=D, height=height,
            cutoff=cutoff, jastrow_amplitude=amplitude, k_max=k_max,
            n_grid=24 if quick else 32)
        value = ref.measure(beta)
        references.append(dict(k_max=k_max, n_modes=value.n_modes,
                               n_grid=value.grid_points_per_axis, beta=beta,
                               midpoint_potential=value.midpoint_potential,
                               projected_energy=value.energy))
    reference_mid = references[-1]["midpoint_potential"]
    replica_rows = []
    summary_rows = []
    for it, tau in enumerate(steps):
        links = round(2 * beta / tau)
        if not np.isclose(links*tau/2, beta) or links % 2:
            raise RuntimeError("time-step grid does not preserve symmetric projection beta")
        for kind in ("baseline", "staged"):
            series = []
            eff = []
            accept = []
            for ir in range(replicas):
                seed = 2026 + it * 100_000 + (0 if kind == "baseline" else 10_000) + 100*ir
                cfg = replace(base, time_step=tau, n_links=links, seed=seed)
                sampler = (PrimitiveInteractingPath(cell, potential, trial, 2, cfg,
                                                     kinetic_coefficient=D)
                           if kind == "baseline" else
                           StagedInteractingPath(cell, potential, trial, 2, cfg,
                                                 kinetic_coefficient=D))
                result = sampler.run()
                diag = autocorrelation_estimate(result.midpoint_potential)
                series.append(diag.mean)
                eff.append(diag.effective_samples)
                accept.append(result.acceptances)
                replica_rows.append(dict(
                    sampler=kind, tau=tau, n_links=links, beta=beta, replica=ir,
                    seed=seed, n_measurements=diag.n_samples,
                    midpoint_v=diag.mean, midpoint_v_chain_stderr=diag.stderr,
                    tau_int_measurements=diag.tau_integrated,
                    effective_measurements=diag.effective_samples,
                    autocorrelation_window=diag.window,
                    endpoint_energy=float(result.endpoint_local_energy.mean()),
                    thermodynamic_energy=float(result.thermodynamic_energy.mean()),
                    accept_bead=result.acceptances["bead"],
                    accept_staging=result.acceptances.get("staging", float("nan")),
                    accept_bisection=result.acceptances.get("bisection", float("nan")),
                    accept_translate=result.acceptances["translate"],
                    accept_refresh=result.acceptances["refresh"],
                ))
            mean, sem = _mean_sem(series)
            row = dict(sampler=kind, tau=tau, n_links=links, beta=beta,
                       replicas=replicas, midpoint_v=mean, midpoint_v_sem=sem,
                       mean_effective_measurements=float(np.mean(eff)),
                       reference_midpoint_v=reference_mid,
                       reference_difference=mean-reference_mid,
                       mean_staging_acceptance=float(np.nanmean([a.get("staging", np.nan) for a in accept]))
                       if kind == "staged" else "",
                       mean_bisection_acceptance=float(np.nanmean([a.get("bisection", np.nan) for a in accept]))
                       if kind == "staged" else "")
            summary_rows.append(row)
            print(f"{kind:8s} tau={tau:.3f}  V_mid={mean:.5f} ± {sem:.5f}  "
                  f"ESS/chain={row['mean_effective_measurements']:.1f} "
                  f"(Fourier K=2: {reference_mid:.5f})")
    _csv(output / "replicas.csv", replica_rows)
    _csv(output / "summary.csv", summary_rows)
    _csv(output / "fourier_reference.csv", references)
    with (output / "protocol.json").open("w", encoding="utf-8") as f:
        json.dump({"model": "N=2 soft-core bosons in periodic 3D cell; primitive fixed-beta PIGS",
                   "L": cell.length, "height": height, "cutoff": cutoff,
                   "D": D, "jastrow_amplitude": amplitude, "beta": beta,
                   "time_steps": steps, "replicas": replicas, "base_config": asdict(base),
                   "reference": "truncated relative 3D Fourier-Galerkin basis with grid quadrature",
                   "limitations": "finite tau, beta, K, grid, correlated MCMC; labelled paths; no exchanges or OBDM"},
                  f, indent=2)
    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    for kind, name in (("baseline", "Local/refresh"), ("staged", "Staging/bisection")):
        pts = sorted((r for r in summary_rows if r["sampler"] == kind), key=lambda r:r["tau"])
        ax.errorbar([x["tau"] for x in pts], [x["midpoint_v"] for x in pts],
                    yerr=[x["midpoint_v_sem"] for x in pts], marker="o", capsize=3, label=name)
    ax.axhline(reference_mid, color="gray", ls="--", label="Fourier K=2 reference")
    ax.set(xlabel="Primitive imaginary-time step τ", ylabel="Pure projected midpoint V",
           title="Two-boson interacting fixed-β benchmark")
    ax.grid(alpha=.22);ax.legend()
    fig.tight_layout();fig.savefig(output / "midpoint_convergence.png", dpi=160);plt.close(fig)
    fig, ax = plt.subplots(figsize=(7.4,4.5))
    for kind, name in (("baseline", "Local/refresh"), ("staged", "Staging/bisection")):
        pts = sorted((r for r in summary_rows if r["sampler"] == kind),key=lambda r:r["tau"])
        ax.plot([x["tau"] for x in pts],[x["mean_effective_measurements"] for x in pts],
                "o-",label=name)
    ax.set(xlabel="Primitive imaginary-time step τ", ylabel="Effective samples per replica",
           title="Autocorrelation-adjusted sample count (same MC steps)")
    ax.grid(alpha=.22);ax.legend()
    fig.tight_layout();fig.savefig(output / "effective_samples.png",dpi=160);plt.close(fig)
    print("Saved Patch 16 results to", output)
    return replica_rows, summary_rows, references


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output", default="outputs/bosons/path_staging")
    args = parser.parse_args()
    run_experiment(args.output, quick=args.quick)


if __name__ == "__main__":
    main()
