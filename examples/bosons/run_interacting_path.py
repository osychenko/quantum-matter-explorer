"""Patch 15: continuum interacting primitive-action open-path benchmark.

    python3 -m examples.bosons.run_interacting_path --quick
    python3 -m examples.bosons.run_interacting_path

Not a pure OBDM or a thermodynamic Bose-exchange simulation. The quick mode
is a smoke test, not a converged interacting ground-state calculation.
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
from qmc.interacting_path import InteractingPathConfig, PrimitiveInteractingPath


class UniformTrial:
    """Exact free-particle trial and boundary state."""

    def log_psi(self, R):
        return np.zeros(len(R))

    def grad_log_psi(self, R):
        return np.zeros_like(R)

    def laplacian_log_psi(self, R):
        return np.zeros(len(R))


class CosinePotential:
    """Separable, nonconstant 1-particle benchmark V(x)=a(1-cos(2pi x/L))."""

    def __init__(self, cell, amplitude=1.0):
        self.cell = cell
        self.amplitude = float(amplitude)

    def __call__(self, R):
        return self.amplitude * np.sum(
            1 - np.cos(2 * np.pi * R[..., 0] / self.cell.length), axis=1
        )


def exact_cosine_reference(cell_length, D, amplitude, beta, n_modes=9):
    """Independent Fourier reference: exact symmetric two-sided projection.

    The basis comprises wave numbers m=-n_modes..+n_modes in the x direction.
    Y and z remain free, with uniform orbital and zero kinetic energy.
    """
    m = np.arange(-n_modes, n_modes + 1)
    h = np.diag(D * (2*np.pi*m / cell_length)**2 + amplitude)
    for j in range(len(m) - 1):
        h[j, j+1] = h[j+1, j] = -amplitude/2
    evals, evecs = np.linalg.eigh(h)
    w = evecs[n_modes, :]**2
    probs = w * np.exp(-2*beta*(evals-evals[0]))
    projected_energy = float(np.dot(probs, evals) / probs.sum())
    trial = np.zeros(len(m)); trial[n_modes] = 1
    phi = evecs @ (np.exp(-beta*(evals-evals[0])) * (evecs.T @ trial))
    phi /= np.linalg.norm(phi)
    midpoint_potential = float(amplitude - amplitude*np.dot(phi[:-1], phi[1:]))
    return projected_energy, midpoint_potential


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _mean_sem(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))


def run_experiment(output, *, quick=False):
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    box = CubicCell(3.5)
    D, cutoff, beta = 0.5, 1.4, 0.9
    step_sizes = (0.30, 0.15) if quick else (0.30, 0.15, 0.075)
    replicas = 2 if quick else 3
    config = InteractingPathConfig(
        n_steps=380 if quick else 2600,
        burn_in=90 if quick else 500,
        thin=4 if quick else 8,
        refresh_probability=0.15,
        translate_probability=0.12,
        move_sigma=0.48,
    )
    uniform = UniformTrial()
    cases = (
        ("free_reference", 1, lambda: (lambda R: np.zeros(len(R)), uniform)),
        ("cosine_reference", 1, lambda: (CosinePotential(box, 1.0), uniform)),
        ("soft_core_pairs", 3, lambda: (
            SoftCorePairPotential(box, height=2.0, cutoff=cutoff),
            SoftCoreJastrowTrial(box, amplitude=0.6, cutoff=cutoff))),
    )
    replica_rows, summary_rows = [], []
    for ic, (label, n_particles, factory) in enumerate(cases):
        for it, tau in enumerate(step_sizes):
            links = int(round(2*beta/tau))
            if links % 2 or not np.isclose(links*tau/2, beta):
                raise RuntimeError("time-step grid must preserve projection beta")
            vals = {k: [] for k in ("midpoint_v", "endpoint_energy", "thermodynamic_energy")}
            accept = {k: [] for k in ("bead", "translate", "refresh")}
            for ir in range(replicas):
                seed = 2026 + ic*1_000_000 + it*100_000 + ir*1000
                cfg = replace(config, seed=seed, time_step=float(tau), n_links=links)
                potential, trial = factory()
                sampler = PrimitiveInteractingPath(box, potential, trial, n_particles, cfg,
                                                   kinetic_coefficient=D)
                result = sampler.run()
                values = {
                    "midpoint_v": float(np.mean(result.midpoint_potential)),
                    "endpoint_energy": float(np.mean(result.endpoint_local_energy)),
                    "thermodynamic_energy": float(np.mean(result.thermodynamic_energy)),
                }
                for key, value in values.items():
                    vals[key].append(value)
                for key in accept:
                    accept[key].append(result.acceptances[key])
                replica_rows.append(dict(
                    case=label, n_particles=n_particles, beta=beta,
                    tau=tau, n_links=links, replica=ir, seed=seed,
                    n_measurements=len(result.midpoint_potential), **values,
                    **{f"accept_{key}": result.acceptances[key] for key in accept},
                ))
            row = dict(case=label, n_particles=n_particles, beta=beta,
                       tau=tau, n_links=links, replicas=replicas)
            for key, measurements in vals.items():
                mean, sem = _mean_sem(measurements)
                row[key] = mean
                row[key + "_sem"] = sem
            for key, values in accept.items():
                row["accept_" + key] = float(np.nanmean(values))
            if label == "cosine_reference":
                row["exact_endpoint_energy"], row["exact_midpoint_v"] = exact_cosine_reference(
                    box.length, D, 1.0, beta
                )
            else:
                row["exact_endpoint_energy"] = 0.0 if label == "free_reference" else ""
                row["exact_midpoint_v"] = 0.0 if label == "free_reference" else ""
            summary_rows.append(row)
            print(f"{label:19s} tau={tau:.3f} V_mid={row['midpoint_v']:.4f} "
                  f"E_end={row['endpoint_energy']:.4f} +/- "
                  f"{row['endpoint_energy_sem']:.4f} "
                  f"E_thermo={row['thermodynamic_energy']:.3f}")
    _write_csv(out / "replicas.csv", replica_rows)
    _write_csv(out / "summary.csv", summary_rows)
    with (out / "protocol.json").open("w", encoding="utf-8") as f:
        json.dump({
            "model": "primitive interacting continuum open paths (no permutations)",
            "cell_length": box.length, "D": D, "cutoff": cutoff,
            "beta": beta, "time_steps": step_sizes, "replicas": replicas,
            "base_config": asdict(config),
            "estimators": "diagonal midpoint V, endpoint trial local energy, thermodynamic derivative",
            "limitations": "primitive time-step bias, finite projection, MCMC autocorrelation; no pure OBDM",
        }, f, indent=2)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for label, _, _ in cases:
        points = sorted((r for r in summary_rows if r["case"] == label),
                        key=lambda r: r["tau"])
        x = [r["tau"] for r in points]
        for ax, k, sem in ((axes[0], "endpoint_energy", "endpoint_energy_sem"),
                           (axes[1], "midpoint_v", "midpoint_v_sem")):
            ax.errorbar(x, [r[k] for r in points], yerr=[r[sem] for r in points],
                        marker="o", capsize=3, label=label.replace("_", " "))
    for ax, ylabel in zip(axes, ("Endpoint local energy", "Midpoint potential")):
        ax.set(xlabel="Imaginary-time step τ", ylabel=ylabel)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Primitive interacting open-path benchmark (fixed projection β)")
    fig.tight_layout()
    fig.savefig(out / "timestep_convergence.png", dpi=170)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    for case in ("cosine_reference", "soft_core_pairs"):
        points = sorted((r for r in summary_rows if r["case"] == case), key=lambda r: r["tau"])
        x = [r["tau"] for r in points]
        ax.errorbar(x, [r["thermodynamic_energy"] for r in points],
                    yerr=[r["thermodynamic_energy_sem"] for r in points],
                    marker="o", capsize=3, label=case.replace("_", " "))
    exact = exact_cosine_reference(box.length, D, 1.0, beta)[0]
    ax.axhline(exact, linestyle="--", color="0.5", label="Cosine Fourier reference")
    ax.set(xlabel="Imaginary-time step τ", ylabel="Thermodynamic energy",
           title="Noisy derivative-of-Z energy diagnostic")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "thermodynamic_energy.png", dpi=170)
    plt.close(fig)
    print("Saved Patch 15 benchmark to", out)
    return replica_rows, summary_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Small smoke benchmark")
    parser.add_argument("--output", default="outputs/bosons/interacting_path")
    options = parser.parse_args()
    run_experiment(options.output, quick=options.quick)


if __name__ == "__main__":
    main()
