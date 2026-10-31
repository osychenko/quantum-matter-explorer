"""DMC-assisted *approximate* one-body coherence for periodic positive bosonic trials.

The local displaced-wavefunction ratio of Psi_T has expectation
    VMC:   <Psi_T|O|Psi_T>/<Psi_T|Psi_T>
    mixed: <Psi_0|O|Psi_T>/<Psi_0|Psi_T>
when sampled from |Psi_T|^2 and the importance-sampled DMC distribution,
respectively. For a real Hermitian displacement-averaged O, 2*mixed - VMC
cancels first-order trial-wavefunction error, but is NOT a pure estimator.

DMC weights, walker ancestry, and population/time-step bias are not removed.
Keep fresh independent replicas and compare multiple projection times, dt,
walker populations, and trials before interpreting any coherence difference.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .bosons import SoftCorePairPotential
from .cells import CubicCell
from .coherence import OneBodyDensityMatrix, VariationalCondensateOccupation
from .dmc import DMCConfig, run_dmc_importance
from .observables import StaticStructureFactor
from .order import DensityWaveJastrowTrial
from .vmc import VMCConfig, run_vmc


@dataclass(frozen=True)
class ProjectedCoherenceConfig:
    n_particles: int = 6
    cell_length: float = 4.0
    cutoff: float = 1.8
    kinetic_coefficient: float = 0.5
    potential_height: float = 3.0
    jastrow_amplitude: float = 0.8
    order_amplitude: float = 0.08
    indices: tuple[tuple[int, int, int], ...] = ((2, 0, 0), (0, 2, 0), (0, 0, 2))
    branches: tuple[str, ...] = ("liquid", "modulated")
    dt_values: tuple[float, ...] = (0.04, 0.02)
    walker_populations: tuple[int, ...] = (64, 128)
    equilibration_tau_values: tuple[float, ...] = (0.4, 0.8)
    production_tau: float = 0.8
    scheme: str = "metropolis"
    observable_stride: int = 2
    vmc_walkers: int = 64
    vmc_steps: int = 100
    vmc_burn_in: int = 30
    vmc_sample_every: int = 3
    proposal_sigma: float = 0.35
    n_radii: int = 8
    draws_per_configuration: int = 1
    replicas: int = 3
    seed: int = 2026


@dataclass(frozen=True)
class ProjectedReplica:
    branch: str
    dt: float
    n_walkers: int
    equilibration_tau: float
    actual_equilibration_tau: float
    actual_production_tau: float
    replica: int
    vmc_seed: int
    dmc_seed: int
    n_observed_dmc_configurations: int
    vmc_n0_over_n: float
    mixed_n0_over_n: float
    extrapolated_n0_over_n: float
    vmc_s_over_n: float
    mixed_s_over_n: float
    vmc_energy_per_particle: float
    dmc_mixed_energy_per_particle: float
    dmc_acceptance: float


@dataclass(frozen=True)
class ProjectedSummary:
    branch: str
    dt: float
    n_walkers: int
    equilibration_tau: float
    actual_equilibration_tau: float
    actual_production_tau: float
    replicas: int
    vmc_n0_over_n: float
    vmc_n0_stderr: float
    mixed_n0_over_n: float
    mixed_n0_stderr: float
    extrapolated_n0_over_n: float
    extrapolated_n0_stderr: float
    vmc_s_over_n: float
    mixed_s_over_n: float
    dmc_energy_per_particle: float
    dmc_energy_stderr: float
    dmc_acceptance: float


def _finite_positive(value, label, *, allow_zero=False):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be finite") from exc
    if not np.isfinite(number) or (number < 0 if allow_zero else number <= 0):
        raise ValueError(f"{label} must be finite and {'nonnegative' if allow_zero else 'positive'}")
    return number


def _positive_grid(values, label, *, integers=False):
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{label} must be a sequence")
    try:
        arr = np.asarray(values, dtype=float)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{label} must be numeric") from exc
    if arr.ndim != 1 or not len(arr) or not np.all(np.isfinite(arr)) or np.any(arr <= 0):
        raise ValueError(f"{label} must be a nonempty positive finite sequence")
    if integers and (np.any(arr != np.rint(arr)) or np.any(arr < 2)):
        raise ValueError(f"{label} must contain integers >= 2")
    if len(np.unique(arr)) != len(arr):
        raise ValueError(f"{label} contains duplicate values")
    return tuple(int(x) if integers else float(x) for x in arr)


def validate_projected_config(cfg: ProjectedCoherenceConfig):
    if not isinstance(cfg.n_particles, int) or isinstance(cfg.n_particles, bool) or cfg.n_particles < 2:
        raise ValueError("n_particles must be an integer >= 2")
    cell = CubicCell(_finite_positive(cfg.cell_length, "cell_length"))
    cutoff = _finite_positive(cfg.cutoff, "cutoff")
    SoftCorePairPotential(cell, height=_finite_positive(cfg.potential_height, "potential_height", allow_zero=True), cutoff=cutoff)
    _finite_positive(cfg.kinetic_coefficient, "kinetic_coefficient")
    _finite_positive(cfg.jastrow_amplitude, "jastrow_amplitude", allow_zero=True)
    _finite_positive(cfg.order_amplitude, "order_amplitude", allow_zero=True)
    if (not isinstance(cfg.branches, (tuple, list)) or not cfg.branches
            or len(set(cfg.branches)) != len(cfg.branches)
            or set(cfg.branches) - {"liquid", "modulated"}):
        raise ValueError("branches must be unique liquid/modulated names")
    _positive_grid(cfg.dt_values, "dt_values")
    _positive_grid(cfg.walker_populations, "walker_populations", integers=True)
    _positive_grid(cfg.equilibration_tau_values, "equilibration_tau_values")
    _finite_positive(cfg.production_tau, "production_tau")
    if cfg.scheme not in {"metropolis", "second_order"}:
        raise ValueError("scheme must be metropolis or second_order")
    for name in ("observable_stride", "vmc_walkers", "vmc_steps", "vmc_sample_every", "n_radii", "draws_per_configuration", "replicas"):
        value = getattr(cfg, name)
        minimum = 2 if name in {"vmc_walkers", "n_radii", "replicas"} else 1
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    if (not isinstance(cfg.vmc_burn_in, int) or isinstance(cfg.vmc_burn_in, bool)
            or not 0 <= cfg.vmc_burn_in < cfg.vmc_steps):
        raise ValueError("require 0 <= vmc_burn_in < vmc_steps")
    _finite_positive(cfg.proposal_sigma, "proposal_sigma")
    if not isinstance(cfg.seed, int) or isinstance(cfg.seed, bool) or cfg.seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    # Validate modes using the production trial, including +/- redundancy.
    DensityWaveJastrowTrial(
        cell, jastrow_amplitude=cfg.jastrow_amplitude,
        order_amplitude=cfg.order_amplitude, cutoff=cutoff, indices=cfg.indices,
    )
    for dt in cfg.dt_values:
        if np.ceil(cfg.production_tau / dt) < cfg.observable_stride:
            raise ValueError("production_tau/dt must provide at least one DMC observable sample")
        steps = np.ceil(max(cfg.equilibration_tau_values) / dt) + np.ceil(cfg.production_tau / dt)
        if not np.isfinite(steps) or steps > 10_000_000:
            raise ValueError("DMC projection schedule exceeds safe step limit")
    return cell


def _trial(cfg, cell, branch):
    return DensityWaveJastrowTrial(
        cell, jastrow_amplitude=cfg.jastrow_amplitude,
        order_amplitude=cfg.order_amplitude if branch == "modulated" else 0.0,
        cutoff=cfg.cutoff, indices=cfg.indices,
    )


def _observers(cfg, trial, cell, seed):
    radii = np.linspace(0.0, cell.length / 2.0, cfg.n_radii)
    g1 = OneBodyDensityMatrix(
        trial, cell, cfg.n_particles, radii,
        draws_per_configuration=cfg.draws_per_configuration, seed=seed + 1,
    )
    occupation = VariationalCondensateOccupation(
        trial, cell, cfg.n_particles,
        draws_per_configuration=cfg.draws_per_configuration, seed=seed + 2,
    )
    structure = StaticStructureFactor(cell, cfg.n_particles, cfg.indices)
    return g1, occupation, structure


def _mean_sem(vals):
    a = np.asarray(vals, dtype=float)
    if a.ndim != 1 or len(a) < 2 or not np.all(np.isfinite(a)):
        raise ValueError("replicas must give >= 2 finite scalar estimates")
    return float(a.mean()), float(a.std(ddof=1) / np.sqrt(len(a)))


def _save_csv(path: Path, rows, record_type):
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=[f.name for f in fields(record_type)])
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)


def _save_curve(path: Path, rows):
    names = (
        "branch", "dt", "n_walkers", "equilibration_tau", "radius",
        "vmc_g1", "vmc_g1_stderr", "mixed_g1", "mixed_g1_stderr",
        "extrapolated_g1", "extrapolated_g1_stderr",
    )
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def _plots(summary, curve, output):
    for branch in sorted({x.branch for x in summary}):
        fig, ax = plt.subplots(figsize=(7.5, 4.7))
        selected = [r for r in summary if r.branch == branch]
        for (dt, nw), pts in sorted(_group_by(selected, lambda r: (r.dt, r.n_walkers)).items()):
            pts = sorted(pts, key=lambda x: x.equilibration_tau)
            ax.errorbar(
                [p.equilibration_tau for p in pts],
                [p.extrapolated_n0_over_n for p in pts],
                yerr=[p.extrapolated_n0_stderr for p in pts],
                marker="o", capsize=3, label=f"dt={dt:g}, Nw={nw}",
            )
        ax.set(xlabel="DMC equilibration imaginary time", ylabel="2 × mixed − VMC k=0 occupation",
               title=f"{branch}: approximate coherence, not pure DMC")
        ax.grid(alpha=0.2)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / f"occupation_{branch}.png", dpi=160)
        plt.close(fig)

        # Use the highest-population, lowest-dt, longest-equilibration point
        # as an illustrative *noisy* g1 curve, not an extrapolated pure limit.
        reference = sorted(selected, key=lambda p: (p.dt, -p.n_walkers, -p.equilibration_tau))[0]
        matches = [r for r in curve if (r["branch"] == branch and r["dt"] == reference.dt
                    and r["n_walkers"] == reference.n_walkers
                    and r["equilibration_tau"] == reference.equilibration_tau)]
        fig, ax = plt.subplots(figsize=(7.5, 4.7))
        for key, label in (("vmc_g1", "VMC"), ("mixed_g1", "DMC mixed"),
                           ("extrapolated_g1", "2 mixed − VMC")):
            ax.errorbar([r["radius"] for r in matches], [r[key] for r in matches],
                        yerr=[r[key + "_stderr"] for r in matches],
                        marker="o", markersize=3, capsize=2, label=label)
        ax.set(xlabel="Separation r", ylabel="Normalized g1(r)",
               title=f"{branch}: finite-cell coherence ({reference.n_walkers} walkers)")
        ax.legend()
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(output / f"g1_{branch}.png", dpi=160)
        plt.close(fig)


def _group_by(rows, key):
    result = {}
    for r in rows:
        result.setdefault(key(r), []).append(r)
    return result


def run_projected_coherence(cfg: ProjectedCoherenceConfig, output_dir="outputs/bosons/projected_coherence"):
    """Sample trial ratios under VMC and DMC; save paired replica diagnostics.

    Returns (replica_rows, summary_rows). DMC is not guaranteed to have
    equilibrated, and mixed/extrapolated estimators are not pure estimators.
    """
    cell = validate_projected_config(cfg)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    potential = SoftCorePairPotential(cell, height=cfg.potential_height, cutoff=cfg.cutoff)
    replica_rows, summary_rows, curve_rows = [], [], []
    # VMC baseline is shared by each dt/population/projection comparison for
    # the same branch and replica, which induces covariance ACROSS grid points.
    vmc_baselines = {}
    for ib, branch in enumerate(cfg.branches):
        trial = _trial(cfg, cell, branch)
        for replica in range(cfg.replicas):
            seed = cfg.seed + 100_000 * ib + 10_000 * replica + 100
            rng = np.random.default_rng(seed + 7)
            initial = rng.uniform(0, cell.length, (cfg.vmc_walkers, cfg.n_particles, 3))
            obs = _observers(cfg, trial, cell, seed + 1000)
            vmc = run_vmc(
                potential, trial,
                VMCConfig(
                    n_walkers=cfg.vmc_walkers, n_steps=cfg.vmc_steps,
                    burn_in=cfg.vmc_burn_in, sample_every=cfg.vmc_sample_every,
                    proposal_sigma=cfg.proposal_sigma, adaptive_proposal=False,
                    proposal_mode="particle", kinetic_coefficient=cfg.kinetic_coefficient,
                    configuration_shape=(cfg.n_particles, 3), seed=seed,
                ), initial_walkers=initial, observables=obs,
            )
            vmc_baselines[(branch, replica)] = (
                seed, obs[0].result().g1, obs[1].result().n0_over_n,
                float(np.mean(obs[2].result().s_k) / cfg.n_particles),
                float(np.mean(vmc.energy_history) / cfg.n_particles),
            )

        for idt, dt in enumerate(cfg.dt_values):
            for ipop, nw in enumerate(cfg.walker_populations):
                for iproj, tau in enumerate(cfg.equilibration_tau_values):
                    n_eq = int(np.ceil(tau / dt))
                    n_prod = int(np.ceil(cfg.production_tau / dt))
                    collected = []
                    for replica in range(cfg.replicas):
                        # Unique allocation across branches, grid points and replicas.
                        dmc_seed = (cfg.seed + 1_000_000 + ib * 10_000_000 +
                                    idt * 1_000_000 + ipop * 100_000 +
                                    iproj * 10_000 + replica * 100)
                        base = vmc_baselines[(branch, replica)]
                        init = np.random.default_rng(dmc_seed + 7).uniform(
                            0, cell.length, size=(nw, cfg.n_particles, 3)
                        )
                        obs = _observers(cfg, trial, cell, dmc_seed + 1000)
                        result = run_dmc_importance(
                            potential, trial,
                            DMCConfig(
                                n_walkers=nw, n_steps=n_eq + n_prod,
                                equilibration_steps=n_eq, adaptation_steps=0,
                                dt=dt, adaptive_dt=False, log_interval=0,
                                scheme=cfg.scheme, configuration_shape=(cfg.n_particles, 3),
                                kinetic_coefficient=cfg.kinetic_coefficient, seed=dmc_seed,
                            ), initial_walkers=init, observables=obs,
                            observable_stride=cfg.observable_stride,
                        )
                        mixed_g = obs[0].result().g1
                        mixed_n0 = float(obs[1].result().n0_over_n)
                        vmc_g = base[1]
                        extrap_g = 2.0 * mixed_g - vmc_g
                        extrap_n0 = 2.0 * mixed_n0 - base[2]
                        record = ProjectedReplica(
                            branch=branch, dt=dt, n_walkers=nw,
                            equilibration_tau=tau,
                            actual_equilibration_tau=n_eq * dt,
                            actual_production_tau=n_prod * dt, replica=replica,
                            vmc_seed=base[0], dmc_seed=dmc_seed,
                            n_observed_dmc_configurations=obs[0].result().n_configurations,
                            vmc_n0_over_n=base[2], mixed_n0_over_n=mixed_n0,
                            extrapolated_n0_over_n=extrap_n0,
                            vmc_s_over_n=base[3],
                            mixed_s_over_n=float(np.mean(obs[2].result().s_k) / cfg.n_particles),
                            vmc_energy_per_particle=base[4],
                            dmc_mixed_energy_per_particle=float(np.mean(result.local_energy_history) / cfg.n_particles),
                            dmc_acceptance=float(result.production_acceptance),
                        )
                        collected.append((record, vmc_g, mixed_g, extrap_g))
                        replica_rows.append(record)
                    key = (branch, dt, nw, tau)
                    summary_rows.append(ProjectedSummary(
                        branch=branch, dt=dt, n_walkers=nw, equilibration_tau=tau,
                        actual_equilibration_tau=n_eq * dt,
                        actual_production_tau=n_prod * dt, replicas=cfg.replicas,
                        vmc_n0_over_n=_mean_sem([r[0].vmc_n0_over_n for r in collected])[0],
                        vmc_n0_stderr=_mean_sem([r[0].vmc_n0_over_n for r in collected])[1],
                        mixed_n0_over_n=_mean_sem([r[0].mixed_n0_over_n for r in collected])[0],
                        mixed_n0_stderr=_mean_sem([r[0].mixed_n0_over_n for r in collected])[1],
                        extrapolated_n0_over_n=_mean_sem([r[0].extrapolated_n0_over_n for r in collected])[0],
                        extrapolated_n0_stderr=_mean_sem([r[0].extrapolated_n0_over_n for r in collected])[1],
                        vmc_s_over_n=float(np.mean([r[0].vmc_s_over_n for r in collected])),
                        mixed_s_over_n=float(np.mean([r[0].mixed_s_over_n for r in collected])),
                        dmc_energy_per_particle=_mean_sem([r[0].dmc_mixed_energy_per_particle for r in collected])[0],
                        dmc_energy_stderr=_mean_sem([r[0].dmc_mixed_energy_per_particle for r in collected])[1],
                        dmc_acceptance=float(np.mean([r[0].dmc_acceptance for r in collected])),
                    ))
                    for j, r in enumerate(obs[0].radii):
                        vals = np.array([[rec[1][j], rec[2][j], rec[3][j]] for rec in collected])
                        means = np.mean(vals, axis=0)
                        sems = np.std(vals, axis=0, ddof=1) / np.sqrt(cfg.replicas)
                        curve_rows.append(dict(
                            branch=branch, dt=dt, n_walkers=nw,
                            equilibration_tau=tau, radius=float(r),
                            vmc_g1=float(means[0]), vmc_g1_stderr=float(sems[0]),
                            mixed_g1=float(means[1]), mixed_g1_stderr=float(sems[1]),
                            extrapolated_g1=float(means[2]),
                            extrapolated_g1_stderr=float(sems[2]),
                        ))
                    last = summary_rows[-1]
                    print(f"{branch:9s} dt={dt:.3g} Nw={nw:4d} tau_eq={tau:g}: "
                          f"VMC={last.vmc_n0_over_n:.4f}, mixed={last.mixed_n0_over_n:.4f}, "
                          f"2mix-VMC={last.extrapolated_n0_over_n:.4f}")
    _save_csv(output / "replicas.csv", replica_rows, ProjectedReplica)
    _save_csv(output / "summary.csv", summary_rows, ProjectedSummary)
    _save_curve(output / "g1.csv", curve_rows)
    with (output / "protocol.json").open("w", encoding="utf-8") as f:
        json.dump({
            "config": asdict(cfg),
            "disclaimer": "VMC, importance-sampled DMC mixed, and first-order extrapolated trial-ratio estimators; NOT pure DMC OBDM/BEC",
            "sampling": "replica means; DMC after resampling; VMC baseline reused across grid points",
            "physical_model": "periodic finite-range soft-core Bose Hamiltonian; fixed trial amplitudes",
        }, f, indent=2)
    _plots(summary_rows, curve_rows, output)
    return replica_rows, summary_rows
