"""Variational g1(r) and finite-cell k=0 occupancy for periodic soft-core bosons.

Run from repository root:
    python3 -m examples.bosons.run_coherence --quick
    python3 -m examples.bosons.run_coherence

Not a DMC pure estimator or a proof of thermodynamic BEC.  The ideal-gas
case uses a different Hamiltonian and is an exact-method reference, not an
energetically competing trial for the interacting model.
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

from qmc import CubicCell, SoftCorePairPotential, VMCConfig, run_vmc
from qmc.coherence import OneBodyDensityMatrix, VariationalCondensateOccupation
from qmc.order import DensityWaveJastrowTrial


@dataclass(frozen=True)
class CoherenceExperimentConfig:
    n_particles: int = 8
    cell_length: float = 4.0
    cutoff: float = 1.8
    n_walkers: int = 64
    n_steps: int = 200
    burn_in: int = 50
    sample_every: int = 4
    proposal_sigma: float = 0.35
    n_radii: int = 13
    draws_per_configuration: int = 2
    replicas: int = 3
    seed: int = 2026


def _write_csv(path, rows, fieldnames):
    with Path(path).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _mean_sem(data):
    values = np.asarray(data, dtype=float)
    if len(values) < 2:
        raise ValueError("at least two independent runs are required")
    return float(np.mean(values)), float(np.std(values, ddof=1) / np.sqrt(len(values)))


def _validate(cfg):
    if not isinstance(cfg.n_particles, int) or cfg.n_particles < 2:
        raise ValueError("n_particles must be >= 2")
    if not isinstance(cfg.n_walkers, int) or cfg.n_walkers < 2:
        raise ValueError("n_walkers must be >= 2")
    if not isinstance(cfg.replicas, int) or cfg.replicas < 2:
        raise ValueError("replicas must be >= 2")
    if not isinstance(cfg.n_radii, int) or cfg.n_radii < 2:
        raise ValueError("n_radii must be >= 2")
    if not isinstance(cfg.draws_per_configuration, int) or cfg.draws_per_configuration < 1:
        raise ValueError("draws_per_configuration must be positive")
    if (not isinstance(cfg.n_steps, int) or not isinstance(cfg.burn_in, int)
            or not 0 <= cfg.burn_in < cfg.n_steps):
        raise ValueError("require 0 <= burn_in < n_steps")
    if not isinstance(cfg.sample_every, int) or cfg.sample_every < 1:
        raise ValueError("sample_every must be positive")
    if not isinstance(cfg.seed, int) or cfg.seed < 0:
        raise ValueError("seed must be nonnegative")
    if not np.isfinite(cfg.proposal_sigma) or cfg.proposal_sigma <= 0:
        raise ValueError("proposal_sigma must be finite and positive")
    if (cfg.n_steps - cfg.burn_in + cfg.sample_every - 1) // cfg.sample_every < 2:
        raise ValueError("need at least two production measurements")
    cell = CubicCell(cfg.cell_length)
    if not 0.0 < cfg.cutoff <= cell.length / 2:
        raise ValueError("cutoff must be <= half the cubic cell length")
    return cell


def run_coherence_experiment(cfg, output_dir="outputs/bosons/coherence"):
    """Save replica-level variational coherence and finite k=0 occupation data."""
    cell = _validate(cfg)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    radii = np.linspace(0.0, cell.length / 2, cfg.n_radii)
    modes = ((2, 0, 0), (0, 2, 0), (0, 0, 2))
    # The ideal case is an analytic estimator check, not a competing variational
    # state for the interacting Hamiltonian used in the other two cases.
    cases = (
        ("ideal_reference", 0.0, 0.0, 0.0),
        ("interacting_jastrow", 3.0, 0.8, 0.0),
        ("density_modulated", 3.0, 0.8, 0.10),
    )
    replica_rows, summary_rows, obdm_rows = [], [], []
    for case_id, (label, height, amplitude, order) in enumerate(cases):
        e_values, n0_values, g1_values, acceptance_values = [], [], [], []
        potential = SoftCorePairPotential(cell, height=height, cutoff=cfg.cutoff)
        trial = DensityWaveJastrowTrial(
            cell, jastrow_amplitude=amplitude, order_amplitude=order,
            cutoff=cfg.cutoff, indices=modes,
        )
        for i in range(cfg.replicas):
            seed = cfg.seed + case_id * 100_000 + i * 1_000
            rng = np.random.default_rng(seed + 7)
            initial = rng.uniform(0, cell.length,
                                  size=(cfg.n_walkers, cfg.n_particles, 3))
            obdm = OneBodyDensityMatrix(
                trial, cell, cfg.n_particles, radii,
                draws_per_configuration=cfg.draws_per_configuration,
                seed=seed + 19,
            )
            n0 = VariationalCondensateOccupation(
                trial, cell, cfg.n_particles,
                draws_per_configuration=cfg.draws_per_configuration,
                seed=seed + 23,
            )
            result = run_vmc(
                potential, trial,
                VMCConfig(
                    n_walkers=cfg.n_walkers,
                    configuration_shape=(cfg.n_particles, 3),
                    proposal_mode="particle", n_steps=cfg.n_steps,
                    burn_in=cfg.burn_in, sample_every=cfg.sample_every,
                    proposal_sigma=cfg.proposal_sigma, adaptive_proposal=True,
                    seed=seed,
                ),
                initial_walkers=initial,
                observables=(obdm, n0),
            )
            e = float(np.mean(result.energy_history) / cfg.n_particles)
            occupancy = n0.result().n0_over_n
            g1 = obdm.result().g1
            e_values.append(e)
            n0_values.append(occupancy)
            g1_values.append(g1)
            acceptance_values.append(float(result.production_acceptance))
            replica_rows.append(dict(
                case=label, replica=i, seed=seed, height=height,
                jastrow_amplitude=amplitude, order_amplitude=order,
                energy_per_particle=e, n0_over_n=occupancy,
                acceptance=float(result.production_acceptance),
                n_configurations=obdm.n_configurations,
                n_displacement_samples=n0.n_displacement_samples,
            ))
        energy_mean, energy_sem = _mean_sem(e_values)
        n0_mean, n0_sem = _mean_sem(n0_values)
        summary_rows.append(dict(
            case=label, height=height,
            jastrow_amplitude=amplitude, order_amplitude=order,
            energy_per_particle=energy_mean, energy_replica_stderr=energy_sem,
            n0_over_n=n0_mean, n0_replica_stderr=n0_sem,
            mean_acceptance=float(np.mean(acceptance_values)),
            replicas=cfg.replicas,
        ))
        g1_stack = np.asarray(g1_values, dtype=float)
        means = np.mean(g1_stack, axis=0)
        sems = np.std(g1_stack, axis=0, ddof=1) / np.sqrt(cfg.replicas)
        for r, mean, sem in zip(radii, means, sems):
            obdm_rows.append(dict(case=label, radius=float(r),
                                  g1=float(mean), g1_replica_stderr=float(sem)))
        print(f"{label:22s}  E/N={energy_mean:.5f} +/- {energy_sem:.4f}  "
              f"n0/N={n0_mean:.5f} +/- {n0_sem:.4f}  "
              f"accept={np.mean(acceptance_values):.3f}")
    _write_csv(output / "replicas.csv", replica_rows, replica_rows[0].keys())
    _write_csv(output / "summary.csv", summary_rows, summary_rows[0].keys())
    _write_csv(output / "obdm.csv", obdm_rows, obdm_rows[0].keys())
    with (output / "protocol.json").open("w", encoding="utf-8") as f:
        json.dump({"config": asdict(cfg), "cases": list(cases),
                   "interpretation": "VMC trial-state coherence; no pure DMC estimation"},
                  f, indent=2)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for row in summary_rows:
        branch = [x for x in obdm_rows if x["case"] == row["case"]]
        x = np.array([x["radius"] for x in branch])
        y = np.array([x["g1"] for x in branch])
        err = np.array([x["g1_replica_stderr"] for x in branch])
        ax.plot(x, y, "o-", markersize=3, label=row["case"].replace("_", " "))
        ax.fill_between(x, y - err, y + err, alpha=0.16)
    ax.axhline(1, color="0.5", linestyle="--", alpha=0.6)
    ax.set(xlabel="Radial separation r", ylabel=r"$g_1(r)=\rho_1(r)/\rho$",
           title="Variational one-body coherence (finite periodic cell)")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "obdm.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    xx = np.arange(len(summary_rows))
    ax.errorbar(xx, [r["n0_over_n"] for r in summary_rows],
                yerr=[r["n0_replica_stderr"] for r in summary_rows],
                fmt="o", capsize=4)
    ax.set_xticks(xx, [r["case"].replace("_", "\n") for r in summary_rows])
    ax.set(ylabel=r"Variational finite-cell $n_{k=0}/N$",
           title="Uniform-displacement coherence integral")
    ax.axhline(1, color="0.5", linestyle="--", alpha=0.5)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "occupation.png", dpi=160)
    plt.close(fig)
    print("Saved variational coherence results to", output)
    return replica_rows, summary_rows, obdm_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Small CPU smoke experiment")
    parser.add_argument("--output", default="outputs/bosons/coherence")
    args = parser.parse_args()
    cfg = CoherenceExperimentConfig()
    if args.quick:
        cfg = CoherenceExperimentConfig(
            n_particles=6, cell_length=4.0, cutoff=1.8,
            n_walkers=16, n_steps=34, burn_in=10, sample_every=4,
            n_radii=7, draws_per_configuration=1, replicas=2,
        )
    run_coherence_experiment(cfg, output_dir=args.output)


if __name__ == "__main__":
    main()
