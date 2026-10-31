"""Exact *finite-lattice* pure OBDM reference: imaginary-time projector.

    python3 -m examples.bosons.run_pure_projector --quick
    python3 -m examples.bosons.run_pure_projector

This is small-system Bose-Hubbard exact diagonalization, NOT 3D continuum
PIGS/reptation QMC and NOT a pure estimator for the soft-core QMC model.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qmc.pure_projector import BoseHubbardRing, ExactBosonProjector


def _csv(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def run_experiment(
    *, output_dir="outputs/bosons/pure_projector", quick=False,
    n_sites=None, n_bosons=None, hopping=1.0, onsite_u=2.0,
    nearest_v=0.4, gutzwiller_penalty=1.0,
):
    """Compare exact beta->infinity OBDM against finite-beta symmetric projections."""
    L = (4 if quick else 6) if n_sites is None else n_sites
    N = (3 if quick else 5) if n_bosons is None else n_bosons
    model = BoseHubbardRing(
        n_sites=L, n_bosons=N, hopping=hopping,
        onsite_u=onsite_u, nearest_v=nearest_v,
    )
    projector = ExactBosonProjector(model)
    if projector.gap < 1e-10:
        raise ValueError("benchmark requires a unique ground state")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    betas = (0., .15, .4, 1., 2., 4., float("inf")) if quick else (
        0., .05, .15, .4, .8, 1.5, 3., 6., float("inf")
    )
    seeds = ("coherent", "gutzwiller", "localized")
    ground = projector.measure(float("inf"))
    results = {}
    rows, obdm_rows, spectrum_rows = [], [], []
    for name in seeds:
        seed = projector.trial(name, penalty=gutzwiller_penalty)
        series = []
        for beta in betas:
            data = projector.measure(beta, seed)
            series.append(data)
            rows.append((
                name, "infinity" if np.isinf(beta) else beta,
                data.energy, data.energy - ground.energy,
                data.n0_fraction, data.condensate_eigenvalue_fraction,
                data.n0_fraction - ground.n0_fraction, data.gap,
            ))
            for shift, g1 in enumerate(data.g1):
                obdm_rows.append((name, "infinity" if np.isinf(beta) else beta, shift, g1))
            for orbital_rank, population in enumerate(data.occupations):
                spectrum_rows.append((name, "infinity" if np.isinf(beta) else beta,
                                      orbital_rank, population))
        results[name] = series
        print(f"{name:10s} beta=0 occupation={series[0].n0_fraction:.6f} "
              f"beta=inf occupation={series[-1].n0_fraction:.6f}")

    _csv(output / "projection.csv",
         ("trial", "beta", "energy", "energy_above_ground", "n0_fraction",
          "largest_natural_occupation_fraction", "n0_minus_exact", "spectral_gap"), rows)
    _csv(output / "obdm.csv", ("trial", "beta", "displacement", "g1"), obdm_rows)
    _csv(output / "natural_occupations.csv",
         ("trial", "beta", "orbital_rank", "occupation"), spectrum_rows)
    (output / "protocol.json").write_text(json.dumps({
        "model": {"n_sites": L, "n_bosons": N, "hopping": hopping,
                  "onsite_u": onsite_u, "nearest_v": nearest_v},
        "hilbert_dimension": model.dimension,
        "gutzwiller_penalty": gutzwiller_penalty,
        "projection_beta_per_side": ["infinity" if np.isinf(b) else b for b in betas],
        "exact_ground_energy": ground.energy,
        "exact_gap": projector.gap,
        "exact_ground_n0_fraction": ground.n0_fraction,
        "interpretation": "Dense finite-lattice symmetric imaginary-time projector; "
                          "not a continuum or stochastic QMC estimator.",
    }, indent=2) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    for name in seeds:
        series = results[name][:-1]
        xs = [p.beta for p in series]
        axes[0].plot(xs, [p.n0_fraction for p in series], "o-", label=name)
        axes[1].plot(xs, [p.energy - ground.energy for p in series], "o-", label=name)
    axes[0].axhline(ground.n0_fraction, linestyle="--", color="0.5", label="exact ground state")
    axes[1].axhline(0, linestyle="--", color="0.5")
    axes[0].set(ylabel=r"Pure projected $n_{k=0}/N$",
                title="Symmetric finite-basis imaginary-time projection")
    axes[1].set(xlabel=r"Imaginary time per side $\beta$",
                ylabel=r"$E(\beta)-E_0$")
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output / "projection_convergence.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for name in seeds:
        ax.plot(range(L), results[name][0].g1, "o--", alpha=.65, label=f"{name}: beta=0")
    ax.plot(range(L), ground.g1, "ko-", linewidth=2, label="exact pure ground state")
    ax.set(xlabel="Lattice displacement (sites)", ylabel=r"$g_1(d)$",
           title="Equal-time one-body coherence on a finite 1D ring")
    ax.grid(alpha=.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "obdm_comparison.png", dpi=160)
    plt.close(fig)
    print("Ground energy:", ground.energy, "gap:", ground.gap,
          "n0/N:", ground.n0_fraction)
    print("Saved exact projector benchmarks to", output)
    return results, ground


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Use a 4-site, 3-boson example")
    parser.add_argument("--output", default="outputs/bosons/pure_projector")
    parser.add_argument("--onsite-u", type=float, default=2.0)
    parser.add_argument("--nearest-v", type=float, default=0.4)
    args = parser.parse_args()
    run_experiment(output_dir=args.output, quick=args.quick,
                   onsite_u=args.onsite_u, nearest_v=args.nearest_v)


if __name__ == "__main__":
    main()
