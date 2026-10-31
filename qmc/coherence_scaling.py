"""Finite-size VMC sensitivity study of one-body coherence and density modulation.

Every (N, density, gamma) point independently selects trial parameters using
training chains, then evaluates them on fresh replica chains. This compares
*variational trial-state* coherence, not pure-DMC BEC or a thermodynamic phase.

The finite-cell zero-momentum occupation is estimated by uniform-volume
one-body displacements. The radial g1 data are diagnostics, not an independent
estimate of the zero-mode occupation from their last data point.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .bosons import SoftCorePairPotential
from .cells import CubicCell
from .coherence import OneBodyDensityMatrix, VariationalCondensateOccupation
from .observables import StaticStructureFactor
from .order import DensityWaveJastrowTrial
from .vmc import VMCConfig, run_vmc


@dataclass(frozen=True)
class CoherenceScalingConfig:
    particle_counts: tuple[int, ...] = (8, 16, 32)
    reduced_densities: tuple[float, ...] = (0.65,)
    gammas: tuple[float, ...] = (3.0, 18.0)
    cutoff: float = 1.0
    kinetic_coefficient: float = 0.5
    target_k_cutoff: float = 2.0 * np.pi
    jastrow_amplitudes: tuple[float, ...] = (0.3, 0.8)
    order_amplitudes: tuple[float, ...] = (0.0, 0.08)
    n_walkers: int = 48
    training_steps: int = 100
    validation_steps: int = 180
    burn_in: int = 40
    sample_every: int = 4
    proposal_sigma: float = 0.3
    n_radii: int = 10
    draws_per_configuration: int = 2
    replicas: int = 3
    seed: int = 2026


def _float_grid(values, name, *, integer=False, allow_zero=False):
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a numeric sequence")
    try:
        a = np.asarray(values, dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must be a numeric sequence") from exc
    if (a.ndim != 1 or a.size == 0 or not np.all(np.isfinite(a))
            or np.any(a < 0 if allow_zero else a <= 0)
            or (integer and np.any(a != np.floor(a)))
            or len(np.unique(a)) != len(a)):
        raise ValueError(f"{name} must be finite, unique, and {'nonnegative' if allow_zero else 'positive'}"
                         + (" integers" if integer else ""))
    return tuple(int(v) if integer else float(v) for v in a)


def validate_scaling_config(cfg: CoherenceScalingConfig):
    if not isinstance(cfg, CoherenceScalingConfig):
        raise TypeError("cfg must be CoherenceScalingConfig")
    Ns = _float_grid(cfg.particle_counts, "particle_counts", integer=True)
    if min(Ns) < 2 or len(Ns) < 2:
        raise ValueError("particle_counts must contain at least two distinct N >= 2")
    densities = _float_grid(cfg.reduced_densities, "reduced_densities")
    gammas = _float_grid(cfg.gammas, "gammas")
    amps = _float_grid(cfg.jastrow_amplitudes, "jastrow_amplitudes", allow_zero=True)
    orders = _float_grid(cfg.order_amplitudes, "order_amplitudes", allow_zero=True)
    if 0.0 not in orders or not any(v > 0 for v in orders):
        raise ValueError("order_amplitudes must include 0.0 and a positive value")
    if (not np.isfinite(cfg.cutoff) or cfg.cutoff <= 0 or
            not np.isfinite(cfg.kinetic_coefficient) or cfg.kinetic_coefficient <= 0 or
            not np.isfinite(cfg.target_k_cutoff) or cfg.target_k_cutoff <= 0):
        raise ValueError("cutoff, kinetic_coefficient and target_k_cutoff must be positive and finite")
    if (not np.isfinite(cfg.proposal_sigma) or cfg.proposal_sigma <= 0):
        raise ValueError("proposal_sigma must be positive and finite")
    for name, minimum in (("n_walkers", 2), ("replicas", 2),
                          ("n_radii", 2), ("draws_per_configuration", 1),
                          ("sample_every", 1), ("training_steps", 2),
                          ("validation_steps", 2)):
        value = getattr(cfg, name)
        if not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_)) or value < minimum:
            raise ValueError(f"{name} must be integer >= {minimum}")
    if (not isinstance(cfg.burn_in, (int, np.integer)) or cfg.burn_in < 0
            or cfg.burn_in >= min(cfg.training_steps, cfg.validation_steps)):
        raise ValueError("burn_in must be below training_steps and validation_steps")
    if ((cfg.training_steps - cfg.burn_in - 1) // cfg.sample_every + 1 < 2 or
            (cfg.validation_steps - cfg.burn_in - 1) // cfg.sample_every + 1 < 2):
        raise ValueError("each VMC run needs at least two production measurements")
    if not isinstance(cfg.seed, (int, np.integer)) or isinstance(cfg.seed, (bool, np.bool_)) or cfg.seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    for N in Ns:
        for density in densities:
            length = cfg.cutoff * (N / density) ** (1 / 3)
            if not np.isfinite(length) or length / 2 < cfg.cutoff - 1e-12:
                raise ValueError("cell too small for cutoff: require L >= 2 * cutoff for every N,density")
    return Ns, densities, gammas, amps, orders


def commensurate_modes(cell: CubicCell, cutoff: float, target_k_cutoff: float):
    """Choose three axial reciprocal directions with nearest integer shell index.

    The chosen physical k varies slightly with N: always record actual |k|rc.
    This does not compare rigorously identical k values between distinct cells.
    """
    m = max(1, int(np.floor(target_k_cutoff * cell.length / (2 * np.pi * cutoff) + 0.5)))
    indices = np.array(((m, 0, 0), (0, m, 0), (0, 0, m)), dtype=int)
    actual = float(np.linalg.norm(cell.wavevectors(indices)[0]) * cutoff)
    return indices, m, actual


def _summary(values):
    a = np.asarray(values, dtype=float)
    if a.ndim != 1 or a.size < 2 or not np.all(np.isfinite(a)):
        raise ValueError("need >= 2 independent finite replica estimates")
    return float(a.mean()), float(a.std(ddof=1) / np.sqrt(len(a)))


def _csv(path, rows, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    return path


def _plot_size(summary_rows, output):
    groups = sorted({(r["reduced_density"], r["gamma"], r["branch"])
                     for r in summary_rows})
    fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.8), sharex=True)
    for density, gamma, branch in groups:
        p = sorted((r for r in summary_rows if (r["reduced_density"], r["gamma"], r["branch"])
                    == (density, gamma, branch)), key=lambda r: r["n_particles"])
        x = 1 / np.array([r["n_particles"] for r in p], dtype=float)
        label = f"rho rc^3={density:g}, gamma={gamma:g}, {branch}"
        for ax, value, err in ((axs[0], "n0_over_n", "n0_stderr"),
                               (axs[1], "s_over_n", "s_stderr")):
            ax.errorbar(x, [r[value] for r in p], yerr=[r[err] for r in p],
                        marker="o", capsize=3, label=label)
    axs[0].set_ylabel("Variational finite-cell n0 / N")
    axs[1].set_ylabel("Selected-mode S(k) / N")
    for ax in axs:
        ax.set_xlabel("1 / N (zero is thermodynamic limit)")
        ax.grid(alpha=0.23)
    axs[0].legend(fontsize=7)
    fig.suptitle("Finite-size VMC coherence and order — no extrapolation")
    fig.tight_layout()
    target = Path(output) / "scaling.png"
    fig.savefig(target, dpi=155)
    plt.close(fig)
    return target


def _plot_coherence_structure(summary_rows, output):
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for r in summary_rows:
        ax.errorbar(r["s_over_n"], r["n0_over_n"], xerr=r["s_stderr"],
                    yerr=r["n0_stderr"], fmt="o", capsize=2)
        ax.annotate(f"N={r['n_particles']} {r['branch']} gamma={r['gamma']:g}",
                    (r["s_over_n"], r["n0_over_n"]), xytext=(4, 5),
                    textcoords="offset points", fontsize=7)
    ax.set(xlabel="Selected-mode S(k) / N", ylabel="Variational finite-cell n0 / N",
           title="Coherence versus density modulation — marginal replica SEM")
    ax.grid(alpha=0.23)
    fig.tight_layout()
    target = Path(output) / "coherence_vs_structure.png"
    fig.savefig(target, dpi=155)
    plt.close(fig)
    return target


def _plot_g1(g1_rows, output):
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    groups = sorted({(r["n_particles"], r["reduced_density"], r["gamma"], r["branch"])
                     for r in g1_rows})
    for N, density, gamma, branch in groups:
        pts = [r for r in g1_rows if (r["n_particles"], r["reduced_density"], r["gamma"], r["branch"])
               == (N, density, gamma, branch)]
        ax.plot([r["radius_over_cutoff"] for r in pts], [r["g1"] for r in pts],
                "o-", markersize=2, label=f"N={N}, rho={density:g}, gamma={gamma:g}, {branch}")
    ax.set(xlabel="Separation r / cutoff", ylabel="Variational g1(r)",
           title="Radial coherence (each cell sampled only up to L/2)")
    ax.grid(alpha=0.23)
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    target = Path(output) / "g1_scaling.png"
    fig.savefig(target, dpi=155)
    plt.close(fig)
    return target


def run_coherence_scaling(cfg: CoherenceScalingConfig,
                          output_dir="outputs/bosons/coherence_scaling"):
    """Reoptimize at each size, then collect independent VMC replica diagnostics.

    Returns (training_rows, replica_rows, summary_rows); writes CSV/PNG/JSON.
    All comparisons are *variational*: no DMC mixed/pure extrapolation is made.
    """
    Ns, densities, gammas, amps, orders = validate_scaling_config(cfg)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    training_rows, replica_rows, summary_rows, g1_rows = [], [], [], []
    run_id = 0

    def new_seed():
        nonlocal run_id
        seed = int(cfg.seed) + run_id * 1000
        run_id += 1
        return seed

    def simulate(trial, potential, cell, N, seed, steps, observers=()):
        rng = np.random.default_rng(seed + 1)
        initial = rng.uniform(0, cell.length, size=(cfg.n_walkers, N, 3))
        run = run_vmc(
            potential, trial,
            VMCConfig(
                n_walkers=cfg.n_walkers, configuration_shape=(N, 3),
                kinetic_coefficient=cfg.kinetic_coefficient,
                proposal_mode="particle", n_steps=steps,
                burn_in=cfg.burn_in, sample_every=cfg.sample_every,
                proposal_sigma=cfg.proposal_sigma, adaptive_proposal=True,
                seed=seed,
            ),
            initial_walkers=initial, observables=observers,
        )
        return float(np.mean(run.energy_history) / N), float(run.production_acceptance)

    for N in Ns:
        for density in densities:
            L = float(cfg.cutoff * (N / density) ** (1 / 3))
            cell = CubicCell(L)
            indices, mode_index, k_rc = commensurate_modes(cell, cfg.cutoff, cfg.target_k_cutoff)
            radii = np.linspace(0.0, L / 2.0, cfg.n_radii)
            for gamma in gammas:
                height = float(gamma * cfg.kinetic_coefficient / cfg.cutoff**2)
                potential = SoftCorePairPotential(cell, height=height, cutoff=cfg.cutoff)
                eligible = {"liquid": [], "modulated": []}
                for amplitude in amps:
                    for ordering in orders:
                        branch = "liquid" if ordering == 0 else "modulated"
                        trial = DensityWaveJastrowTrial(
                            cell, jastrow_amplitude=amplitude,
                            order_amplitude=ordering, cutoff=cfg.cutoff,
                            indices=indices,
                        )
                        seed = new_seed()
                        energy, acceptance = simulate(trial, potential, cell, N, seed,
                                                      cfg.training_steps)
                        row = dict(n_particles=N, reduced_density=density, gamma=gamma,
                                   cell_length=L, mode_index=mode_index, k_cutoff=k_rc,
                                   branch=branch, jastrow_amplitude=amplitude,
                                   order_amplitude=ordering, seed=seed,
                                   energy_per_particle=energy, acceptance=acceptance)
                        training_rows.append(row)
                        eligible[branch].append(row)

                for branch in ("liquid", "modulated"):
                    chosen = min(eligible[branch], key=lambda row: row["energy_per_particle"])
                    trial = DensityWaveJastrowTrial(
                        cell, jastrow_amplitude=chosen["jastrow_amplitude"],
                        order_amplitude=chosen["order_amplitude"],
                        cutoff=cfg.cutoff, indices=indices,
                    )
                    energies, occupations, structures, acceptances, g1_values = [], [], [], [], []
                    for replica in range(cfg.replicas):
                        seed = new_seed()
                        n0 = VariationalCondensateOccupation(
                            trial, cell, N,
                            draws_per_configuration=cfg.draws_per_configuration,
                            seed=seed + 2,
                        )
                        g1 = OneBodyDensityMatrix(
                            trial, cell, N, radii,
                            draws_per_configuration=cfg.draws_per_configuration,
                            seed=seed + 3,
                        )
                        sf = StaticStructureFactor(cell, N, indices)
                        energy, acceptance = simulate(
                            trial, potential, cell, N, seed, cfg.validation_steps,
                            observers=(n0, g1, sf),
                        )
                        occupation = float(n0.result().n0_over_n)
                        structure = float(np.mean(sf.result().s_k) / N)
                        energies.append(energy)
                        occupations.append(occupation)
                        structures.append(structure)
                        acceptances.append(acceptance)
                        g1_values.append(g1.result().g1)
                        replica_rows.append(dict(
                            n_particles=N, reduced_density=density, gamma=gamma,
                            branch=branch, replica=replica, seed=seed,
                            cell_length=L, mode_index=mode_index, k_cutoff=k_rc,
                            jastrow_amplitude=chosen["jastrow_amplitude"],
                            order_amplitude=chosen["order_amplitude"],
                            energy_per_particle=energy, n0_over_n=occupation,
                            s_over_n=structure, acceptance=acceptance,
                            n_displacement_samples=n0.n_displacement_samples,
                            n_configurations=n0.n_configurations,
                        ))

                    e_mu, e_se = _summary(energies)
                    n_mu, n_se = _summary(occupations)
                    s_mu, s_se = _summary(structures)
                    summary_rows.append(dict(
                        n_particles=N, reduced_density=density, gamma=gamma,
                        branch=branch, cell_length=L, mode_index=mode_index,
                        k_cutoff=k_rc, target_k_cutoff=cfg.target_k_cutoff,
                        jastrow_amplitude=chosen["jastrow_amplitude"],
                        order_amplitude=chosen["order_amplitude"], replicas=cfg.replicas,
                        energy_per_particle=e_mu, energy_stderr=e_se,
                        n0_over_n=n_mu, n0_stderr=n_se, s_over_n=s_mu,
                        s_stderr=s_se, acceptance=float(np.mean(acceptances)),
                    ))
                    stack = np.asarray(g1_values, dtype=float)
                    g1_mean = stack.mean(axis=0)
                    g1_sem = stack.std(axis=0, ddof=1) / np.sqrt(cfg.replicas)
                    for rad, mu, se in zip(radii, g1_mean, g1_sem):
                        g1_rows.append(dict(
                            n_particles=N, reduced_density=density, gamma=gamma,
                            branch=branch, radius=float(rad),
                            radius_over_cutoff=float(rad / cfg.cutoff),
                            radius_over_L=float(rad / L),
                            g1=float(mu), g1_stderr=float(se),
                        ))
                    print(f"N={N:3d} rho*={density:g} gamma={gamma:g} {branch:9s}: "
                          f"n0/N={n_mu:.4g} ± {n_se:.2g}, "
                          f"S(G)/N={s_mu:.4g} ± {s_se:.2g}, k*={k_rc:.4g}")

    _csv(output / "training.csv", training_rows, training_rows[0].keys())
    _csv(output / "replicas.csv", replica_rows, replica_rows[0].keys())
    _csv(output / "summary.csv", summary_rows, summary_rows[0].keys())
    _csv(output / "g1.csv", g1_rows, g1_rows[0].keys())
    with (output / "protocol.json").open("w", encoding="utf-8") as f:
        json.dump({"config": asdict(cfg),
                   "source": "VMC |Psi_T|^2, independent training and validation seeds",
                   "interpretation": "finite-cell variational sensitivity; not pure DMC BEC or a phase transition",
                   "modes": "nearest axial reciprocal index to target k*; actual k* in CSV",
                   "uncertainty": "SEM over independent replica means; no naive walker error bars"}, f, indent=2)
    _plot_size(summary_rows, output)
    _plot_coherence_structure(summary_rows, output)
    _plot_g1(g1_rows, output)
    print("Saved coherence scaling outputs to", output)
    return training_rows, replica_rows, summary_rows
