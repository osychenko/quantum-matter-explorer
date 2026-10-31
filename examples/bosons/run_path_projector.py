"""Patch 13: stochastic open paths against the exact pure OBDM benchmark.

    python3 -m examples.bosons.run_path_projector --quick
    python3 -m examples.bosons.run_path_projector

The kernel comes from dense exact finite-basis evolution. This is a controlled
1D lattice path-sampling benchmark, NOT 3D continuum open-path QMC.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from qmc.path_projector import OpenPathProjector, PathConfig
from qmc.pure_projector import BoseHubbardRing, ExactBosonProjector


@dataclass(frozen=True)
class ExperimentConfig:
    n_sites: int = 4
    n_bosons: int = 3
    hopping: float = 1.0
    onsite_u: float = 2.0
    nearest_v: float = 0.25
    dt: float = 0.25
    half_steps: tuple[int, ...] = (2, 5, 10)
    trials: tuple[str, ...] = ("coherent", "gutzwiller")
    n_replicas: int = 3
    burn_in: int = 800
    n_samples: int = 1500
    sample_stride: int = 4
    gibbs_updates: int = 4
    seed: int = 2026


def _mean_sem(numbers):
    values = np.asarray(numbers, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("need at least two finite independent replica estimates")
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))


def _csv(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def run_experiment(cfg: ExperimentConfig, output_dir="outputs/bosons/path_projector"):
    if (not isinstance(cfg.n_replicas, int) or cfg.n_replicas < 2
            or not cfg.half_steps or not cfg.trials
            or len(set(cfg.half_steps)) != len(cfg.half_steps)
            or any(not isinstance(x, int) or x < 1 for x in cfg.half_steps)
            or len(set(cfg.trials)) != len(cfg.trials)
            or any(x not in ("coherent", "gutzwiller", "uniform") for x in cfg.trials)):
        raise ValueError("need distinct positive half_steps, supported distinct trials and >=2 replicas")
    model = BoseHubbardRing(
        n_sites=cfg.n_sites, n_bosons=cfg.n_bosons, hopping=cfg.hopping,
        onsite_u=cfg.onsite_u, nearest_v=cfg.nearest_v,
    )
    if model.hopping <= 0:
        raise ValueError("stochastic sampler requires positive hopping")
    # PathConfig validates remaining numerical run parameters.
    PathConfig(cfg.dt, cfg.half_steps[0], cfg.burn_in, cfg.n_samples,
               cfg.sample_stride, cfg.gibbs_updates, cfg.seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    exact = ExactBosonProjector(model)
    replica_rows, summary_rows, g1_rows = [], [], []
    for it, kind in enumerate(cfg.trials):
        phi = exact.trial(kind)
        for ib, half_steps in enumerate(cfg.half_steps):
            beta = half_steps * cfg.dt
            expected = exact.measure(beta, phi)
            runs = []
            for ir in range(cfg.n_replicas):
                seed = cfg.seed + it * 1_000_000 + ib * 10_000 + ir * 100
                sampler = OpenPathProjector(model, PathConfig(
                    dt=cfg.dt, half_steps=half_steps, burn_in=cfg.burn_in,
                    n_samples=cfg.n_samples, sample_stride=cfg.sample_stride,
                    gibbs_updates=cfg.gibbs_updates, seed=seed,
                ), phi=phi)
                result = sampler.run()
                runs.append(result)
                replica_rows.append(dict(
                    trial=kind, beta=beta, half_steps=half_steps,
                    replica=ir, seed=seed,
                    occupation=result.n0_fraction, energy=result.energy,
                    reptation_acceptance=result.reptation_acceptance,
                    exact_occupation=expected.n0_fraction,
                    exact_energy=expected.energy,
                    n_samples=result.n_samples,
                ))
            occupancy, occupancy_sem = _mean_sem([r.n0_fraction for r in runs])
            energy, energy_sem = _mean_sem([r.energy for r in runs])
            acceptance = float(np.mean([r.reptation_acceptance for r in runs]))
            summary_rows.append(dict(
                trial=kind, beta=beta, half_steps=half_steps,
                occupation=occupancy, occupation_stderr=occupancy_sem,
                exact_occupation=expected.n0_fraction,
                energy=energy, energy_stderr=energy_sem,
                exact_energy=expected.energy,
                acceptance=acceptance, n_replicas=cfg.n_replicas,
            ))
            gstack = np.asarray([r.g1 for r in runs])
            gmean = np.mean(gstack, axis=0)
            gsem = np.std(gstack, axis=0, ddof=1)/np.sqrt(cfg.n_replicas)
            for r, (val, err, target) in enumerate(zip(gmean, gsem, expected.g1)):
                g1_rows.append(dict(trial=kind, beta=beta, separation=r,
                                    g1=val, g1_stderr=err, exact_g1=target))
            print(f"{kind:11s} beta={beta:.3f} "
                  f"n0/N={occupancy:.4f} +/- {occupancy_sem:.4f} "
                  f"(exact={expected.n0_fraction:.4f}), "
                  f"E={energy:.4f} (exact={expected.energy:.4f}) "
                  f"accept={acceptance:.3f}")
    _csv(out / "replicas.csv", replica_rows, replica_rows[0].keys())
    _csv(out / "summary.csv", summary_rows, summary_rows[0].keys())
    _csv(out / "g1.csv", g1_rows, g1_rows[0].keys())
    with (out / "protocol.json").open("w", encoding="utf-8") as f:
        json.dump({"config": asdict(cfg), "dimension": model.dimension,
                   "method": "dense exact finite-basis transfer kernel; sampled open paths",
                   "pure": "two-sided projected finite-beta OBDM; not infinite-beta extrapolation",
                   "uncertainty": "independent Markov-chain replica SEM; no claim of pure continuum DMC"},
                  f, indent=2)
    _plot(summary_rows, g1_rows, cfg, out)
    return replica_rows, summary_rows, g1_rows


def _plot(summary, g1, cfg, output):
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    for name in cfg.trials:
        rows = sorted((r for r in summary if r["trial"] == name), key=lambda x: x["beta"])
        x = [r["beta"] for r in rows]
        axes[0].errorbar(x, [r["occupation"] for r in rows],
                         yerr=[r["occupation_stderr"] for r in rows],
                         marker="o", capsize=3, label=name + " MC")
        axes[0].plot(x, [r["exact_occupation"] for r in rows], "--", label=name + " exact")
        axes[1].errorbar(x, [r["energy"] for r in rows],
                         yerr=[r["energy_stderr"] for r in rows],
                         marker="o", capsize=3, label=name + " MC")
        axes[1].plot(x, [r["exact_energy"] for r in rows], "--", label=name + " exact")
    axes[0].set_ylabel("Zero-momentum occupation n₀/N")
    axes[1].set(xlabel="Projection beta on each side", ylabel="Energy")
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.suptitle("Pure midpoint open-path estimator vs exact diagonalization")
    fig.tight_layout()
    fig.savefig(output / "projection_convergence.png", dpi=170)
    plt.close(fig)

    max_beta = max(cfg.half_steps) * cfg.dt
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for name in cfg.trials:
        rows = sorted((r for r in g1 if r["trial"] == name and r["beta"] == max_beta),
                      key=lambda x: x["separation"])
        x = [r["separation"] for r in rows]
        ax.errorbar(x, [r["g1"] for r in rows],
                    yerr=[r["g1_stderr"] for r in rows], capsize=3,
                    marker="o", label=name + " MC")
        ax.plot(x, [r["exact_g1"] for r in rows], "--", label=name + " exact")
    ax.set(xlabel="Lattice-site separation", ylabel="g₁(d)",
           title=f"Projected one-body coherence, beta={max_beta:g}")
    ax.grid(alpha=.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "g1_comparison.png", dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="short laptop smoke run")
    parser.add_argument("--output", default="outputs/bosons/path_projector")
    args = parser.parse_args()
    cfg = ExperimentConfig()
    if args.quick:
        cfg = replace(cfg, half_steps=(2, 4), n_replicas=2, burn_in=160,
                      n_samples=350, sample_stride=3, gibbs_updates=3)
    run_experiment(cfg, args.output)


if __name__ == "__main__":
    main()
