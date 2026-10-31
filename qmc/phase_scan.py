"""Independent-validation VMC scans of periodic soft-core boson density waves.

A grid search selects variational trial parameters on training chains. Fresh,
independently seeded Monte Carlo chains validate the best liquid (lambda=0)
and density-modulated (lambda>0) trial for each interaction strength.

Finite-N variational comparisons and structure peaks are exploratory: they do
not identify a thermodynamic transition, condensate fraction, or superfluidity.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .bosons import SoftCorePairPotential
from .cells import CubicCell
from .observables import PairCorrelation, StaticStructureFactor
from .order import DensityWaveJastrowTrial, canonical_modes
from .stats import block_statistics
from .visualization import plot_cell_snapshot
from .vmc import VMCConfig, run_vmc


@dataclass(frozen=True)
class PhaseScanConfig:
    n_particles: int = 8
    cell_length: float = 4.0
    cutoff: float = 1.8
    kinetic_coefficient: float = 0.5
    strengths: tuple[float, ...] = (3.0, 12.0, 30.0)
    jastrow_amplitudes: tuple[float, ...] = (0.3, 0.8)
    order_amplitudes: tuple[float, ...] = (0.0, 0.04, 0.12)
    mode_indices: tuple[tuple[int, int, int], ...] = ((2, 0, 0), (0, 2, 0), (0, 0, 2))
    n_walkers: int = 64
    n_steps: int = 140
    burn_in: int = 40
    sample_every: int = 4
    block_size: int = 5
    proposal_sigma: float = 0.35
    validation_replicas: int = 3
    n_bins: int = 25
    seed: int = 2026


@dataclass(frozen=True)
class Candidate:
    gamma: float
    branch: str
    jastrow_amplitude: float
    order_amplitude: float
    training_energy_per_particle: float
    training_block_stderr_per_particle: float
    acceptance: float


@dataclass(frozen=True)
class Validation:
    gamma: float
    branch: str
    jastrow_amplitude: float
    order_amplitude: float
    energy_per_particle: float
    energy_replica_stderr: float
    s_mode_mean: float
    s_replica_stderr: float
    acceptance: float
    density_cutoff3: float
    n_replicas: int


def _finite_grid(values, label, *, allow_zero=False):
    data = np.asarray(values, dtype=float)
    if data.ndim != 1 or len(data) == 0 or not np.all(np.isfinite(data)):
        raise ValueError(f"{label} must be a nonempty finite numeric sequence")
    if np.any(data < 0 if allow_zero else data <= 0):
        raise ValueError(f"{label} must contain {'nonnegative' if allow_zero else 'positive'} values")
    if len(set(data.tolist())) != len(data):
        raise ValueError(f"{label} contains duplicates")
    return tuple(float(x) for x in data)


def _check_config(cfg: PhaseScanConfig):
    if not isinstance(cfg.n_particles, int) or cfg.n_particles < 2:
        raise ValueError("n_particles must be >=2")
    if not isinstance(cfg.n_walkers, int) or cfg.n_walkers < 2:
        raise ValueError("n_walkers must be >=2")
    if not isinstance(cfg.validation_replicas, int) or cfg.validation_replicas < 2:
        raise ValueError("validation_replicas must be >=2 to estimate across-replica uncertainty")
    if not (0 <= cfg.burn_in < cfg.n_steps) or cfg.sample_every < 1 or cfg.block_size < 1:
        raise ValueError("invalid burn_in/n_steps/sample_every/block_size")
    n_meas = (cfg.n_steps - cfg.burn_in + cfg.sample_every - 1) // cfg.sample_every
    if n_meas // cfg.block_size < 2:
        raise ValueError("need at least two complete production blocks per run")
    if not np.isfinite(cfg.kinetic_coefficient) or cfg.kinetic_coefficient <= 0:
        raise ValueError("kinetic_coefficient must be positive and finite")
    if not np.isfinite(cfg.proposal_sigma) or cfg.proposal_sigma <= 0:
        raise ValueError("proposal_sigma must be positive and finite")
    if not isinstance(cfg.seed, int) or cfg.seed < 0:
        raise ValueError("seed must be nonnegative integer")
    if not isinstance(cfg.n_bins, int) or cfg.n_bins < 1:
        raise ValueError("n_bins must be positive integer")
    cell = CubicCell(cfg.cell_length)
    SoftCorePairPotential(cell, height=1.0, cutoff=cfg.cutoff)
    mode_indices = canonical_modes(cfg.mode_indices)
    gammas = _finite_grid(cfg.strengths, "strengths")
    amplitudes = _finite_grid(cfg.jastrow_amplitudes, "jastrow_amplitudes", allow_zero=True)
    ordering = _finite_grid(cfg.order_amplitudes, "order_amplitudes", allow_zero=True)
    if 0.0 not in ordering or not any(x > 0 for x in ordering):
        raise ValueError("order_amplitudes must include 0 (liquid) and >0 (modulated)")
    return cell, mode_indices, gammas, amplitudes, ordering


def _config(cfg, *, seed):
    return VMCConfig(
        n_walkers=cfg.n_walkers,
        configuration_shape=(cfg.n_particles, 3),
        kinetic_coefficient=cfg.kinetic_coefficient,
        proposal_mode="particle",
        n_steps=cfg.n_steps,
        burn_in=cfg.burn_in,
        sample_every=cfg.sample_every,
        adaptive_proposal=True,
        proposal_sigma=cfg.proposal_sigma,
        seed=int(seed),
    )


def _initial(cfg, cell, seed):
    rng = np.random.default_rng(seed)
    return cell.wrap(rng.uniform(0.0, cell.length, (cfg.n_walkers, cfg.n_particles, 3)))


def _write_csv(path, header, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def _replica_summary(values):
    data = np.asarray(values, dtype=float)
    return float(data.mean()), float(data.std(ddof=1) / np.sqrt(len(data)))


def plot_phase_scan(rows: list[Validation], path):
    """Plot independent-validation energies and preselected directional mode intensities."""
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    for branch, label in (("liquid", "Jastrow, lambda=0"), ("modulated", "Density-wave-biased")):
        points = sorted((p for p in rows if p.branch == branch), key=lambda x: x.gamma)
        if not points:
            continue
        gamma = [p.gamma for p in points]
        axes[0].errorbar(gamma, [p.energy_per_particle for p in points],
                         yerr=[p.energy_replica_stderr for p in points],
                         marker="o", capsize=3, label=label)
        axes[1].errorbar(gamma, [p.s_mode_mean for p in points],
                         yerr=[p.s_replica_stderr for p in points],
                         marker="o", capsize=3, label=label)
    axes[0].set_ylabel("Validation energy / particle")
    axes[1].set_ylabel("Mean S(k) at selected modes")
    axes[1].set_xlabel("Interaction strength gamma = V0 rc^2 / D")
    for axis in axes:
        axis.grid(alpha=0.22)
        axis.legend()
    fig.suptitle("Finite-size VMC scan (not a phase-boundary determination)")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def run_phase_scan(cfg: PhaseScanConfig, output_dir="outputs/bosons/phase_scan"):
    """Run a two-branch scan and save CSV/PNG diagnostics.

    Training optimization uses independent seeds for each candidate. Chosen
    parameters are evaluated on new independent seeds; *replica* means, not
    individual correlated walkers, determine validation standard errors.

    Returns (candidate_rows, validation_rows); all files are also written.
    """
    cell, indices, gammas, amplitudes, orders = _check_config(cfg)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rho_rc3 = cfg.n_particles * cfg.cutoff**3 / cell.volume
    candidate_rows, validation_rows, replica_rows = [], [], []
    for ig, gamma in enumerate(gammas):
        # gamma = (V0/D) rc², all dimensionless for a fixed cutoff.
        height = gamma * cfg.kinetic_coefficient / cfg.cutoff**2
        potential = SoftCorePairPotential(cell, height=height, cutoff=cfg.cutoff)
        for ia, amp in enumerate(amplitudes):
            for io, order in enumerate(orders):
                seed = cfg.seed + ig * 100_000 + ia * 1_000 + io * 10
                trial = DensityWaveJastrowTrial(
                    cell, jastrow_amplitude=amp, order_amplitude=order,
                    cutoff=cfg.cutoff, indices=indices,
                )
                run = run_vmc(potential, trial, _config(cfg, seed=seed),
                              initial_walkers=_initial(cfg, cell, seed + 1))
                est = block_statistics(run.energy_history, block_size=cfg.block_size)
                candidate_rows.append(Candidate(
                    gamma, "liquid" if order == 0 else "modulated", amp, order,
                    est.mean / cfg.n_particles, est.stderr / cfg.n_particles,
                    float(run.production_acceptance),
                ))

        for ib, branch in enumerate(("liquid", "modulated")):
            eligible = [r for r in candidate_rows if r.gamma == gamma and r.branch == branch]
            chosen = min(eligible, key=lambda r: r.training_energy_per_particle)
            print(f"gamma={gamma:g} {branch:9s}: selected a={chosen.jastrow_amplitude:g} "
                  f"lambda={chosen.order_amplitude:g} training E/N="
                  f"{chosen.training_energy_per_particle:.5g}")
            energies, structures, accepts = [], [], []
            gr_results, sf_results = [], []
            for ir in range(cfg.validation_replicas):
                seed = cfg.seed + 10_000_000 + ig * 100_000 + ib * 10_000 + ir * 100
                trial = DensityWaveJastrowTrial(
                    cell, jastrow_amplitude=chosen.jastrow_amplitude,
                    order_amplitude=chosen.order_amplitude, cutoff=cfg.cutoff,
                    indices=indices,
                )
                g = PairCorrelation(cell, cfg.n_particles, n_bins=cfg.n_bins)
                s = StaticStructureFactor(cell, cfg.n_particles, indices)
                run = run_vmc(potential, trial, _config(cfg, seed=seed),
                              initial_walkers=_initial(cfg, cell, seed + 1),
                              observables=(g, s))
                energy = float(np.mean(run.energy_history) / cfg.n_particles)
                structure = float(np.mean(s.result().s_k))
                energies.append(energy)
                structures.append(structure)
                accepts.append(float(run.production_acceptance))
                gr_results.append(g.result().g_r)
                sf_results.append(s.result().s_k)
                replica_rows.append((gamma, branch, ir, seed, energy, structure,
                                     run.production_acceptance))
                if ir == 0:
                    plot_cell_snapshot(
                        run.walkers, cell,
                        output / f"cell_gamma_{ig:02d}_{branch}.png",
                    )
            e_mean, e_sem = _replica_summary(energies)
            s_mean, s_sem = _replica_summary(structures)
            validation_rows.append(Validation(
                gamma, branch, chosen.jastrow_amplitude, chosen.order_amplitude,
                e_mean, e_sem, s_mean, s_sem, float(np.mean(accepts)), rho_rc3,
                cfg.validation_replicas,
            ))
            _write_csv(
                output / f"gr_gamma_{ig:02d}_{branch}.csv",
                ("radius", "g_r_replica_mean", "g_r_replica_stderr"),
                zip(g.edges[:-1] / 2 + g.edges[1:] / 2,
                    np.mean(gr_results, axis=0),
                    np.std(gr_results, axis=0, ddof=1) / np.sqrt(cfg.validation_replicas)),
            )
            _write_csv(
                output / f"sk_gamma_{ig:02d}_{branch}.csv",
                ("kx_index", "ky_index", "kz_index", "k_magnitude", "s_k_replica_mean", "s_k_replica_stderr"),
                ((int(k[0]), int(k[1]), int(k[2]), float(np.linalg.norm(w)), mu, se)
                 for k, w, mu, se in zip(
                     indices, cell.wavevectors(indices),
                     np.mean(sf_results, axis=0),
                     np.std(sf_results, axis=0, ddof=1) / np.sqrt(cfg.validation_replicas))),
            )
            print(f"  validation E/N={e_mean:.6g} +/- {e_sem:.2g} "
                  f"S_modes={s_mean:.4g} +/- {s_sem:.2g}")
    _write_csv(output / "candidates.csv", tuple(Candidate.__dataclass_fields__),
               ([getattr(r, key) for key in Candidate.__dataclass_fields__] for r in candidate_rows))
    _write_csv(output / "validation.csv", tuple(Validation.__dataclass_fields__),
               ([getattr(r, key) for key in Validation.__dataclass_fields__] for r in validation_rows))
    _write_csv(output / "replicas.csv",
               ("gamma", "branch", "replica", "seed", "energy_per_particle", "s_mode_mean", "acceptance"),
               replica_rows)
    plot_phase_scan(validation_rows, output / "scan.png")
    print("Saved phase-scan diagnostics:", output)
    return candidate_rows, validation_rows
