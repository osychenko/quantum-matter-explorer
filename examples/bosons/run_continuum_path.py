"""Stochastic free-continuum open-path projector versus analytic coherence.

Run from the repository root:
  python3 -m examples.bosons.run_continuum_path --quick
  python3 -m examples.bosons.run_continuum_path

This is a noninteracting 3D periodic reference with an initially nonuniform
positive trial orbital. It does not measure interacting continuum Bose phases.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qmc.cells import CubicCell
from qmc.continuum_path import (
    ContinuumPathConfig, FreeOpenPathSampler, PeriodicCosineOrbital,
    PeriodicFreePropagator, measure_free_coherence,
)


@dataclass(frozen=True)
class ExperimentConfig:
    length: float = 4.0
    kinetic_coefficient: float = 0.5
    amplitudes: tuple[float, ...] = (0.0, 0.8)
    half_projection_times: tuple[float, ...] = (0.45, 0.9, 1.6)
    n_links: tuple[int, ...] = (12, 24)
    n_steps: int = 10000
    burn_in: int = 2500
    thin: int = 5
    refresh_probability: float = 0.60
    n_radii: int = 9
    direction_draws: int = 8
    replicas: int = 3
    seed: int = 2026


def _write(path, rows):
    rows = list(rows)
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _sem(values):
    data = np.asarray(values, dtype=float)
    if len(data) < 2:
        raise ValueError("at least two replicas required")
    return float(np.mean(data)), float(np.std(data, ddof=1) / np.sqrt(len(data)))


def _validate(cfg: ExperimentConfig):
    from qmc.continuum_path import _positive
    cell = CubicCell(_positive(cfg.length, "length"))
    _positive(cfg.kinetic_coefficient, "kinetic_coefficient")
    if (not cfg.amplitudes or not cfg.half_projection_times or not cfg.n_links
            or any(not np.isfinite(a) or abs(a) >= 1 for a in cfg.amplitudes)
            or any(not np.isfinite(t) or t <= 0 for t in cfg.half_projection_times)):
        raise ValueError("amplitudes must be finite with |a|<1 and beta positive")
    for n in cfg.n_links:
        if isinstance(n, bool) or not isinstance(n, int) or n < 2 or n % 2:
            raise ValueError("n_links must contain positive even integers >= 2")
    for key, lo in (("n_steps", 2), ("burn_in", 0), ("thin", 1),
                    ("n_radii", 2), ("direction_draws", 1), ("replicas", 2), ("seed", 0)):
        v = getattr(cfg, key)
        if isinstance(v, bool) or not isinstance(v, int) or v < lo:
            raise ValueError(f"{key} must be an integer >= {lo}")
    if cfg.burn_in >= cfg.n_steps:
        raise ValueError("burn_in must be less than n_steps")
    if not np.isfinite(cfg.refresh_probability) or not 0 <= cfg.refresh_probability <= 1:
        raise ValueError("refresh_probability must be in [0,1]")
    return cell


def run_experiment(cfg: ExperimentConfig, output_dir="outputs/bosons/continuum_path"):
    cell = _validate(cfg)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    kernel = PeriodicFreePropagator(cell, cfg.kinetic_coefficient)
    radii = np.linspace(0, cell.length / 2, cfg.n_radii)
    replica_rows, summary_rows, g1_rows = [], [], []
    for ia, amplitude in enumerate(cfg.amplitudes):
        orbital = PeriodicCosineOrbital(cell, amplitude)
        for ib, beta in enumerate(cfg.half_projection_times):
            exact_g1, exact_n0 = orbital.exact_radial(beta, radii, cfg.kinetic_coefficient)
            for il, links in enumerate(cfg.n_links):
                occupancies, curves, accept_refresh, accept_reptation = [], [], [], []
                for ir in range(cfg.replicas):
                    seed = cfg.seed + ia*1_000_000 + ib*100_000 + il*10_000 + ir*100
                    pcfg = ContinuumPathConfig(
                        time_step=2*beta/links, n_links=links, n_steps=cfg.n_steps,
                        burn_in=cfg.burn_in, thin=cfg.thin,
                        refresh_probability=cfg.refresh_probability, seed=seed,
                    )
                    run = FreeOpenPathSampler(kernel, orbital, pcfg).run()
                    measured = measure_free_coherence(
                        kernel, run, radii, draws=cfg.direction_draws,
                        seed=seed+1, estimator="endpoint",
                    )
                    occupancies.append(measured.n0_fraction)
                    curves.append(measured.g1)
                    accept_refresh.append(run.refresh_acceptance)
                    accept_reptation.append(run.reptation_acceptance)
                    replica_rows.append(dict(amplitude=amplitude, beta=beta, n_links=links,
                        time_step=pcfg.time_step, replica=ir, seed=seed,
                        n_paths=measured.n_paths, n0_fraction=measured.n0_fraction,
                        exact_n0_fraction=exact_n0, refresh_acceptance=run.refresh_acceptance,
                        reptation_acceptance=run.reptation_acceptance))
                n0, n0_sem = _sem(occupancies)
                curves = np.asarray(curves)
                curve_mean = curves.mean(axis=0)
                curve_sem = curves.std(axis=0, ddof=1) / np.sqrt(cfg.replicas)
                summary_rows.append(dict(amplitude=amplitude, beta=beta,
                    n_links=links, time_step=2*beta/links, replicas=cfg.replicas,
                    n0_fraction=n0, n0_replica_sem=n0_sem,
                    exact_n0_fraction=exact_n0,
                    mean_refresh_acceptance=float(np.mean(accept_refresh)),
                    mean_reptation_acceptance=float(np.mean(accept_reptation))))
                for r, g, se, exact in zip(radii, curve_mean, curve_sem, exact_g1):
                    g1_rows.append(dict(amplitude=amplitude, beta=beta,
                        n_links=links, radius=float(r), g1=float(g),
                        g1_replica_sem=float(se), exact_g1=float(exact)))
                print(f"a={amplitude:+.2f} beta={beta:.2f} links={links:2d} "
                      f"n0/N={n0:.4f} +/- {n0_sem:.3f} exact={exact_n0:.4f}")
    _write(output / "replicas.csv", replica_rows)
    _write(output / "summary.csv", summary_rows)
    _write(output / "g1.csv", g1_rows)
    with (output / "protocol.json").open("w", encoding="utf-8") as f:
        json.dump({"config": asdict(cfg), "model": "3D periodic free particle, factorized bosonic orbital",
                   "estimator": "exact free-kernel blocked endpoint ratio",
                   "caution": "finite-time pure reference; not interacting continuum PIGS"}, f, indent=2)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for amplitude in cfg.amplitudes:
        points = sorted((p for p in summary_rows if p["amplitude"] == amplitude and
                         p["n_links"] == cfg.n_links[-1]), key=lambda x:x["beta"])
        x = [p["beta"] for p in points]
        ax.errorbar(x, [p["n0_fraction"] for p in points],
                    yerr=[p["n0_replica_sem"] for p in points], fmt="o-", capsize=3,
                    label=f"stochastic a={amplitude:g}")
        ax.plot(x, [p["exact_n0_fraction"] for p in points], "--", alpha=.8,
                label=f"exact a={amplitude:g}")
    ax.set(xlabel="Projection time beta (each side)", ylabel="Finite-cell n0/N",
           title="Free 3D continuum: pure projector vs exact reference")
    ax.legend();ax.grid(alpha=.2);fig.tight_layout()
    fig.savefig(output / "occupation_convergence.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    beta = cfg.half_projection_times[0]
    links = cfg.n_links[-1]
    for amplitude in cfg.amplitudes:
        rows = [p for p in g1_rows if p["amplitude"] == amplitude and p["beta"] == beta
                and p["n_links"] == links]
        x = np.array([p["radius"] for p in rows])
        y = np.array([p["g1"] for p in rows]); err = np.array([p["g1_replica_sem"] for p in rows])
        ax.errorbar(x, y, yerr=err, marker="o", capsize=2, label=f"MC a={amplitude:g}")
        ax.plot(x, [p["exact_g1"] for p in rows], "--", alpha=.8,
                label=f"exact a={amplitude:g}")
    ax.set(xlabel="Radial separation r", ylabel="Normalized pure g1(r)",
           title=f"Free continuum midpoint coherence (beta={beta:g})")
    ax.legend();ax.grid(alpha=.2);fig.tight_layout()
    fig.savefig(output / "g1_comparison.png", dpi=160)
    plt.close(fig)
    print("Saved continuum path benchmark:", output)
    return replica_rows, summary_rows, g1_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Small smoke run")
    parser.add_argument("--output", default="outputs/bosons/continuum_path")
    args = parser.parse_args()
    if args.quick:
        cfg = ExperimentConfig(half_projection_times=(0.6,1.2), n_links=(12,),
             n_steps=4400, burn_in=900, thin=3, n_radii=6,
             direction_draws=4, replicas=2)
    else:
        cfg = ExperimentConfig()
    run_experiment(cfg, args.output)


if __name__ == "__main__":
    main()
