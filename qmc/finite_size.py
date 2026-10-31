"""Finite-size and initialization-robustness VMC protocol for periodic bosons.

This is a *variational sensitivity study*, not a thermodynamic phase detector.
Every N is reoptimized at fixed dimensionless density and coupling; selection
runs are independent of held-out replica chains. Error bars use independent
replica means, never correlated walkers or individual sweeps.
"""

from __future__ import annotations

import csv
import json
import platform
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .bosons import SoftCorePairPotential
from .cells import CubicCell
from .observables import StaticStructureFactor
from .order import DensityWaveJastrowTrial
from .stats import block_statistics
from .vmc import VMCConfig, run_vmc


@dataclass(frozen=True)
class FiniteSizeConfig:
    particle_counts: tuple[int, ...] = (8, 16)
    reduced_densities: tuple[float, ...] = (0.55, 0.85)
    gammas: tuple[float, ...] = (3.0, 18.0)
    cutoff: float = 1.0
    kinetic_coefficient: float = 0.5
    target_kr: float = 2.0 * np.pi
    jastrow_amplitudes: tuple[float, ...] = (0.3, 0.8)
    order_amplitudes: tuple[float, ...] = (0.0, 0.08)
    starts: tuple[str, ...] = ("uniform", "lattice")
    n_walkers: int = 32
    burn_in: int = 25
    training_sweeps: int = 60
    production_sweeps: tuple[int, ...] = (60, 120)
    sample_every: int = 3
    block_size: int = 4
    proposal_sigma: float = 0.30
    replicas: int = 3
    seed: int = 2026


@dataclass(frozen=True)
class TrainingPoint:
    n_particles: int
    reduced_density: float
    gamma: float
    mode_index: int
    k_cutoff: float
    branch: str
    jastrow_amplitude: float
    order_amplitude: float
    seed: int
    energy_per_particle: float
    block_stderr_per_particle: float
    acceptance: float


@dataclass(frozen=True)
class ReplicaPoint:
    n_particles: int
    reduced_density: float
    gamma: float
    cell_length: float
    mode_index: int
    k_cutoff: float
    branch: str
    start: str
    production_sweeps: int
    replica: int
    seed: int
    init_seed: int
    jastrow_amplitude: float
    order_amplitude: float
    energy_per_particle: float
    s_mode: float
    s_per_particle: float
    acceptance: float


@dataclass(frozen=True)
class SummaryPoint:
    n_particles: int
    reduced_density: float
    gamma: float
    mode_index: int
    k_cutoff: float
    branch: str
    start: str
    production_sweeps: int
    replicas: int
    energy_per_particle: float
    energy_stderr: float
    s_mode: float
    s_mode_stderr: float
    s_per_particle: float
    s_per_particle_stderr: float
    acceptance: float


@dataclass(frozen=True)
class LengthContrast:
    n_particles: int
    reduced_density: float
    gamma: float
    branch: str
    start: str
    short_sweeps: int
    long_sweeps: int
    energy_long_minus_short: float
    energy_difference_stderr: float
    s_per_particle_long_minus_short: float
    s_difference_stderr: float


@dataclass(frozen=True)
class StartContrast:
    n_particles: int
    reduced_density: float
    gamma: float
    branch: str
    production_sweeps: int
    energy_lattice_minus_uniform: float
    energy_difference_stderr: float
    s_per_particle_lattice_minus_uniform: float
    s_difference_stderr: float


def _positive_finite_sequence(values, name, *, integers=False, allow_zero=False):
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a numeric sequence")
    arr = np.asarray(values)
    if arr.ndim != 1 or len(arr) == 0 or not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"{name} must be a nonempty numeric sequence")
    try:
        floats = np.asarray(arr, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} must contain finite numbers") from exc
    if not np.all(np.isfinite(floats)) or np.any(floats < (0 if allow_zero else 1e-300)):
        raise ValueError(f"{name} must contain finite {'nonnegative' if allow_zero else 'positive'} numbers")
    if integers and np.any(floats != np.rint(floats)):
        raise ValueError(f"{name} must contain integers")
    if len(set(floats.tolist())) != len(floats):
        raise ValueError(f"{name} contains duplicate entries")
    return tuple(int(x) if integers else float(x) for x in floats)


def _protocol(cfg: FiniteSizeConfig):
    if not isinstance(cfg.seed, (int, np.integer)) or cfg.seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(cfg.replicas, (int, np.integer)) or cfg.replicas < 2:
        raise ValueError("replicas must be at least 2 for between-run stderr")
    if not isinstance(cfg.n_walkers, (int, np.integer)) or cfg.n_walkers < 2:
        raise ValueError("n_walkers must be at least 2")
    for key in ("burn_in", "training_sweeps", "sample_every", "block_size"):
        v = getattr(cfg, key)
        if not isinstance(v, (int, np.integer)) or v < (0 if key == "burn_in" else 1):
            raise ValueError(f"invalid {key}")
    if not np.isfinite(cfg.cutoff) or cfg.cutoff <= 0:
        raise ValueError("cutoff must be finite and positive")
    if not np.isfinite(cfg.kinetic_coefficient) or cfg.kinetic_coefficient <= 0:
        raise ValueError("kinetic_coefficient must be finite and positive")
    if not np.isfinite(cfg.target_kr) or cfg.target_kr <= 0:
        raise ValueError("target_kr must be finite and positive")
    if not np.isfinite(cfg.proposal_sigma) or cfg.proposal_sigma <= 0:
        raise ValueError("proposal_sigma must be finite and positive")
    counts = _positive_finite_sequence(cfg.particle_counts, "particle_counts", integers=True)
    if min(counts) < 2:
        raise ValueError("particle_counts must be >=2")
    densities = _positive_finite_sequence(cfg.reduced_densities, "reduced_densities")
    gammas = _positive_finite_sequence(cfg.gammas, "gammas")
    jastrow = _positive_finite_sequence(cfg.jastrow_amplitudes, "jastrow_amplitudes", allow_zero=True)
    ordering = _positive_finite_sequence(cfg.order_amplitudes, "order_amplitudes", allow_zero=True)
    if 0.0 not in ordering or not any(v > 0 for v in ordering):
        raise ValueError("order_amplitudes must contain 0 and a positive amplitude")
    sweeps = _positive_finite_sequence(cfg.production_sweeps, "production_sweeps", integers=True)
    if len(sweeps) < 2 or any(s2 <= s1 for s1, s2 in zip(sweeps, sweeps[1:])):
        raise ValueError("production_sweeps must be strictly increasing")
    if not isinstance(cfg.starts, (tuple, list)) or tuple(cfg.starts) != ("uniform", "lattice"):
        raise ValueError("starts must be ('uniform', 'lattice') for initialization comparison")
    for steps in (cfg.training_sweeps, *sweeps):
        n_meas = (steps + cfg.sample_every - 1) // cfg.sample_every
        if n_meas < 2 * cfg.block_size:
            raise ValueError("each run needs at least two complete measurement blocks")
    # The minimum-image pair cutoff must not exceed half the cubic box length.
    for n in counts:
        for rho_rc3 in densities:
            length = cfg.cutoff * (n / rho_rc3) ** (1.0 / 3.0)
            if length < 2.0 * cfg.cutoff * (1.0 - 1e-12):
                raise ValueError(f"cutoff exceeds half-box length for N={n}, rho*rc^3={rho_rc3}")
    return counts, densities, gammas, jastrow, ordering, sweeps


def _seed(base: int, *indices: int) -> int:
    """Deterministic independent seed namespaces; no global RNG consumption."""
    seq = np.random.SeedSequence([int(base), *(int(x) for x in indices)])
    a, b = (int(x) for x in seq.generate_state(2, dtype=np.uint32))
    return (a << 32) | b


def _initial(cell, n_walkers, n_particles, kind, seed):
    rng = np.random.default_rng(seed)
    if kind == "uniform":
        return rng.uniform(0.0, cell.length, size=(n_walkers, n_particles, 3))
    if kind != "lattice":
        raise ValueError(f"unknown start: {kind!r}")
    side = int(np.ceil(n_particles ** (1.0 / 3.0)))
    grid = np.stack(np.meshgrid(*([np.arange(side)] * 3), indexing="ij"), axis=-1)
    sites = ((grid.reshape(-1, 3) + 0.5) / side) * cell.length
    result = np.empty((n_walkers, n_particles, 3))
    for w in range(n_walkers):
        selected = rng.choice(len(sites), size=n_particles, replace=False)
        translation = rng.uniform(0.0, cell.length, size=(1, 3))
        jitter = rng.normal(0.0, 0.07 * cell.length / side, size=(n_particles, 3))
        result[w] = cell.wrap(sites[selected] + translation + jitter)
    return result


def _vmc_config(cfg, n_particles, sweeps, seed):
    return VMCConfig(
        n_walkers=cfg.n_walkers,
        n_steps=cfg.burn_in + int(sweeps),
        burn_in=cfg.burn_in,
        sample_every=cfg.sample_every,
        proposal_sigma=cfg.proposal_sigma,
        configuration_shape=(n_particles, 3),
        kinetic_coefficient=cfg.kinetic_coefficient,
        proposal_mode="particle",
        seed=seed,
    )


def _trial(cell, cfg, modes, jastrow, order):
    return DensityWaveJastrowTrial(
        cell, jastrow_amplitude=jastrow, order_amplitude=order,
        cutoff=cfg.cutoff, indices=modes,
    )


def _mean_stderr(data):
    values = np.asarray(data, dtype=float)
    if len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("between-replica uncertainty needs >=2 finite run means")
    return float(values.mean()), float(values.std(ddof=1) / np.sqrt(len(values)))


def _save_csv(path, record_type, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = [field.name for field in fields(record_type)]
        writer.writerow(header)
        writer.writerows([getattr(row, key) for key in header] for row in rows)


def plot_finite_size(rows, density, gamma, destination):
    """Show held-out E/N and S(G)/N versus inverse size, with replica SEM."""
    selected = [p for p in rows if p.reduced_density == density and p.gamma == gamma]
    if not selected:
        raise ValueError("no rows for requested density/gamma")
    longest = max(p.production_sweeps for p in selected)
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 8), sharex=True)
    for branch in ("liquid", "modulated"):
        for start, marker in (("uniform", "o"), ("lattice", "s")):
            pts = sorted(
                (p for p in selected if p.branch == branch and p.start == start
                 and p.production_sweeps == longest),
                key=lambda p: p.n_particles,
            )
            if not pts:
                continue
            x = [1.0 / p.n_particles for p in pts]
            style = dict(marker=marker, capsize=3, label=f"{branch}, {start}")
            axes[0].errorbar(x, [p.energy_per_particle for p in pts],
                             yerr=[p.energy_stderr for p in pts], **style)
            axes[1].errorbar(x, [p.s_per_particle for p in pts],
                             yerr=[p.s_per_particle_stderr for p in pts], **style)
    axes[0].set_ylabel("Validation energy / particle")
    axes[1].set_ylabel("Selected-mode S(G) / N")
    axes[1].set_xlabel("1 / N")
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.suptitle(f"Finite-size sensitivity, rho rc³={density:g}, gamma={gamma:g}\n"
                 "Nonidentical nearby reciprocal modes; not a phase diagram")
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return destination


def run_finite_size_study(cfg: FiniteSizeConfig, output_dir="outputs/bosons/finite_size"):
    """Optimize per (N,rho,gamma), then validate across starts and run lengths.

    Returns (training, replicas, summary, start_contrasts, length_contrasts). Every held-out
    cell has >=2 independently initialized, independently seeded replica runs.
    No thermodynamic extrapolation or phase assignment is performed.
    """
    counts, densities, gammas, amplitudes, ordering, lengths = _protocol(cfg)
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    training: list[TrainingPoint] = []
    replicas: list[ReplicaPoint] = []
    summary: list[SummaryPoint] = []
    contrasts: list[StartContrast] = []
    length_contrasts: list[LengthContrast] = []
    for ni, n in enumerate(counts):
        for di, rho in enumerate(densities):
            box_length = cfg.cutoff * (n / rho) ** (1 / 3)
            cell = CubicCell(box_length)
            # Match physical k*rc as closely as a finite periodic box allows.
            m = max(1, int(np.floor(cfg.target_kr * box_length /
                                    (2 * np.pi * cfg.cutoff) + 0.5)))
            modes = ((m, 0, 0), (0, m, 0), (0, 0, m))
            actual_kr = float(2 * np.pi * m * cfg.cutoff / box_length)
            for gi, gamma in enumerate(gammas):
                height = gamma * cfg.kinetic_coefficient / cfg.cutoff**2
                potential = SoftCorePairPotential(cell, height=height, cutoff=cfg.cutoff)
                local_training: list[TrainingPoint] = []
                for ai, a in enumerate(amplitudes):
                    for li, lam in enumerate(ordering):
                        seed = _seed(cfg.seed, 1, ni, di, gi, ai, li)
                        init_seed = _seed(cfg.seed, 2, ni, di, gi, ai, li)
                        trial = _trial(cell, cfg, modes, a, lam)
                        run = run_vmc(
                            potential, trial, _vmc_config(cfg, n, cfg.training_sweeps, seed),
                            initial_walkers=_initial(cell, cfg.n_walkers, n, "uniform", init_seed),
                        )
                        estimate = block_statistics(run.energy_history, block_size=cfg.block_size)
                        point = TrainingPoint(
                            n, rho, gamma, m, actual_kr,
                            "liquid" if lam == 0 else "modulated", a, lam, seed,
                            float(estimate.mean / n), float(estimate.stderr / n),
                            float(run.production_acceptance),
                        )
                        training.append(point)
                        local_training.append(point)
                for bi, branch in enumerate(("liquid", "modulated")):
                    eligible = [point for point in local_training if point.branch == branch]
                    chosen = min(eligible, key=lambda point: point.energy_per_particle)
                    trial = _trial(cell, cfg, modes, chosen.jastrow_amplitude,
                                   chosen.order_amplitude)
                    for si, start in enumerate(cfg.starts):
                        for ti, sweeps in enumerate(lengths):
                            group = []
                            for ri in range(cfg.replicas):
                                seed = _seed(cfg.seed, 3, ni, di, gi, bi, si, ti, ri)
                                init_seed = _seed(cfg.seed, 4, ni, di, gi, bi, si, ti, ri)
                                s = StaticStructureFactor(cell, n, modes)
                                run = run_vmc(
                                    potential, trial, _vmc_config(cfg, n, sweeps, seed),
                                    initial_walkers=_initial(cell, cfg.n_walkers, n, start, init_seed),
                                    observables=(s,),
                                )
                                energy = float(np.mean(run.energy_history) / n)
                                structure = float(np.mean(s.result().s_k))
                                point = ReplicaPoint(
                                    n, rho, gamma, box_length, m, actual_kr,
                                    branch, start, sweeps, ri, seed, init_seed,
                                    chosen.jastrow_amplitude, chosen.order_amplitude,
                                    energy, structure, structure / n,
                                    float(run.production_acceptance),
                                )
                                replicas.append(point)
                                group.append(point)
                            e, se = _mean_stderr([p.energy_per_particle for p in group])
                            s_mean, s_se = _mean_stderr([p.s_mode for p in group])
                            summary.append(SummaryPoint(
                                n, rho, gamma, m, actual_kr,
                                branch, start, sweeps, cfg.replicas,
                                e, se, s_mean, s_se, s_mean / n, s_se / n,
                                float(np.mean([p.acceptance for p in group])),
                            ))
                    print(f"N={n:3d} rho*rc^3={rho:.3f} gamma={gamma:g} "
                          f"{branch:9s} selected a={chosen.jastrow_amplitude:g} "
                          f"lambda={chosen.order_amplitude:g} k*rc={actual_kr:.3f}")
    for row in summary:
        if row.start != "uniform":
            continue
        other = next(p for p in summary if
                     (p.n_particles, p.reduced_density, p.gamma, p.branch, p.production_sweeps, p.start)
                     == (row.n_particles, row.reduced_density, row.gamma, row.branch,
                         row.production_sweeps, "lattice"))
        contrasts.append(StartContrast(
            row.n_particles, row.reduced_density, row.gamma, row.branch, row.production_sweeps,
            other.energy_per_particle - row.energy_per_particle,
            float(np.hypot(other.energy_stderr, row.energy_stderr)),
            other.s_per_particle - row.s_per_particle,
            float(np.hypot(other.s_per_particle_stderr, row.s_per_particle_stderr)),
        ))
    # Independent short/long chains: sensitivity diagnostic, not paired chains.
    for row in summary:
        if row.production_sweeps != lengths[0]:
            continue
        other = next(p for p in summary if
                     (p.n_particles, p.reduced_density, p.gamma, p.branch, p.start, p.production_sweeps)
                     == (row.n_particles, row.reduced_density, row.gamma, row.branch,
                         row.start, lengths[-1]))
        length_contrasts.append(LengthContrast(
            row.n_particles, row.reduced_density, row.gamma, row.branch, row.start,
            lengths[0], lengths[-1],
            other.energy_per_particle - row.energy_per_particle,
            float(np.hypot(other.energy_stderr, row.energy_stderr)),
            other.s_per_particle - row.s_per_particle,
            float(np.hypot(other.s_per_particle_stderr, row.s_per_particle_stderr)),
        ))
    _save_csv(dest / "training.csv", TrainingPoint, training)
    _save_csv(dest / "replicas.csv", ReplicaPoint, replicas)
    _save_csv(dest / "summary.csv", SummaryPoint, summary)
    _save_csv(dest / "start_contrasts.csv", StartContrast, contrasts)
    _save_csv(dest / "length_contrasts.csv", LengthContrast, length_contrasts)
    with (dest / "protocol.json").open("w", encoding="utf-8") as f:
        json.dump({"configuration": asdict(cfg), "estimator": "variational VMC",
                   "numpy_version": np.__version__, "python_version": platform.python_version(),
                   "error_bars": "independent replica-means SEM",
                   "warning": "finite-N trial comparison, not a phase-boundary estimate"},
                  f, indent=2)
    for di, rho in enumerate(densities):
        for gi, gamma in enumerate(gammas):
            plot_finite_size(summary, rho, gamma, dest / f"scaling_density_{di:02d}_gamma_{gi:02d}.png")
    return training, replicas, summary, contrasts, length_contrasts
