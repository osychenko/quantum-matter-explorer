"""Reproducible two-factor DMC timestep/population diagnostic for periodic bosons.

A VMC training grid picks one fixed importance-sampling trial per family.
Independent VMC validation is separate from DMC production. DMC scans use
fixed imaginary-time equilibration and production durations, with distinct
seeds and pre-equilibrated initial configurations for every replica.

The linear dt + 1/Nw fit is a *descriptive model*, not a proof of asymptotic
convergence or of the time-step order of either propagator. Replica errors do
not cover equilibration, trial bias, model choice, or population correlations.
"""

from __future__ import annotations

import csv
import json
import platform
from dataclasses import asdict, dataclass, fields
from math import ceil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .bosons import SoftCorePairPotential
from .cells import CubicCell
from .dmc import DMCConfig, run_dmc_importance
from .order import DensityWaveJastrowTrial, canonical_modes
from .stats import block_statistics
from .vmc import VMCConfig, run_vmc


@dataclass(frozen=True)
class DMCBenchmarkConfig:
    n_particles: int = 8
    reduced_density: float = 0.7  # N * cutoff**3 / cell.volume
    gamma: float = 8.0            # potential_height * cutoff**2 / D
    cutoff: float = 1.0
    kinetic_coefficient: float = 0.5
    jastrow_amplitudes: tuple[float, ...] = (0.3, 0.8)
    order_amplitudes: tuple[float, ...] = (0.0, 0.06)
    mode_indices: tuple[tuple[int, int, int], ...] = ((2, 0, 0), (0, 2, 0), (0, 0, 2))
    dt_values: tuple[float, ...] = (0.04, 0.02, 0.01)
    walker_counts: tuple[int, ...] = (64, 128, 256)
    schemes: tuple[str, ...] = ("metropolis",)
    replicas: int = 3
    vmc_walkers: int = 64
    vmc_steps: int = 150
    vmc_burn_in: int = 50
    vmc_sample_every: int = 2
    vmc_block_size: int = 5
    proposal_sigma: float = 0.30
    warmup_sweeps: int = 30
    equilibration_tau: float = 3.0
    production_tau: float = 8.0
    block_tau: float = 0.5
    bootstrap_samples: int = 200
    seed: int = 2026


@dataclass(frozen=True)
class TrainingCandidate:
    branch: str
    jastrow_amplitude: float
    order_amplitude: float
    seed: int
    energy_per_particle: float
    within_run_block_stderr_per_particle: float
    acceptance: float


@dataclass(frozen=True)
class VMCValidation:
    branch: str
    jastrow_amplitude: float
    order_amplitude: float
    n_replicas: int
    energy_per_particle: float
    energy_replica_stderr: float
    acceptance: float


@dataclass(frozen=True)
class DMCReplica:
    branch: str
    scheme: str
    dt: float
    n_walkers: int
    replica: int
    seed: int
    warmup_seed: int
    n_steps: int
    equilibration_steps: int
    production_steps: int
    actual_equilibration_tau: float
    actual_production_tau: float
    mixed_energy_per_particle: float
    mixed_block_stderr_per_particle: float
    growth_energy_per_particle: float
    growth_block_stderr_per_particle: float
    acceptance: float


@dataclass(frozen=True)
class DMCSummary:
    branch: str
    scheme: str
    dt: float
    n_walkers: int
    n_replicas: int
    mixed_energy_per_particle: float
    mixed_replica_stderr: float
    growth_energy_per_particle: float
    growth_replica_stderr: float
    acceptance: float


@dataclass(frozen=True)
class Extrapolation:
    branch: str
    scheme: str
    estimator: str
    intercept: float
    bootstrap_replica_stderr: float
    dt_slope: float
    inverse_walker_slope: float
    n_grid_points: int
    bootstrap_samples: int
    model: str = "E = E0 + a*dt + b/n_walkers (descriptive only)"


def _positive_grid(values, label: str, *, integer=False, allow_zero=False):
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{label} must be a nonempty numeric sequence")
    try:
        data = np.asarray(values, dtype=float)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{label} must be a numeric sequence") from exc
    if data.ndim != 1 or not len(data) or not np.all(np.isfinite(data)):
        raise ValueError(f"{label} must be nonempty and finite")
    if np.any(data < 0 if allow_zero else data <= 0):
        raise ValueError(f"{label} must be {'nonnegative' if allow_zero else 'positive'}")
    if integer and np.any(data != np.rint(data)):
        raise ValueError(f"{label} must contain integers")
    if len(set(data.tolist())) != len(data):
        raise ValueError(f"{label} values must be distinct")
    return tuple(int(x) if integer else float(x) for x in data)


def validate_benchmark(cfg: DMCBenchmarkConfig):
    """Validate parameters and return (cell, potential, canonical modes)."""
    if not isinstance(cfg.n_particles, int) or cfg.n_particles < 2:
        raise ValueError("n_particles must be an integer >= 2")
    if not isinstance(cfg.replicas, int) or cfg.replicas < 2:
        raise ValueError("replicas must be an integer >= 2")
    if not isinstance(cfg.vmc_walkers, int) or cfg.vmc_walkers < 2:
        raise ValueError("vmc_walkers must be >= 2")
    if not isinstance(cfg.seed, int) or cfg.seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    for label in ("reduced_density", "gamma", "cutoff", "kinetic_coefficient", "proposal_sigma",
                  "equilibration_tau", "production_tau", "block_tau"):
        value = getattr(cfg, label)
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{label} must be positive and finite")
    if (not isinstance(cfg.vmc_steps, int) or not isinstance(cfg.vmc_burn_in, int)
            or not isinstance(cfg.vmc_sample_every, int) or not isinstance(cfg.vmc_block_size, int)
            or not (0 <= cfg.vmc_burn_in < cfg.vmc_steps)
            or cfg.vmc_sample_every < 1 or cfg.vmc_block_size < 1):
        raise ValueError("invalid VMC step/burn-in/block configuration")
    n_vmc = len(range(cfg.vmc_burn_in, cfg.vmc_steps, cfg.vmc_sample_every))
    if n_vmc < 2 * cfg.vmc_block_size:
        raise ValueError("need at least two complete VMC blocks")
    if not isinstance(cfg.warmup_sweeps, int) or cfg.warmup_sweeps < 6:
        raise ValueError("warmup_sweeps must be >= 6")
    if not isinstance(cfg.bootstrap_samples, int) or cfg.bootstrap_samples < 2:
        raise ValueError("bootstrap_samples must be >= 2")
    walkers = _positive_grid(cfg.walker_counts, "walker_counts", integer=True)
    if len(walkers) < 2 or min(walkers) < 2:
        raise ValueError("at least two walker counts >= 2 required")
    steps = _positive_grid(cfg.dt_values, "dt_values")
    if len(steps) < 2:
        raise ValueError("at least two timesteps required")
    ja = _positive_grid(cfg.jastrow_amplitudes, "jastrow_amplitudes", allow_zero=True)
    order = _positive_grid(cfg.order_amplitudes, "order_amplitudes", allow_zero=True)
    if 0.0 not in order or not any(a > 0 for a in order):
        raise ValueError("order_amplitudes must contain 0 and a positive value")
    if not cfg.schemes or len(set(cfg.schemes)) != len(cfg.schemes) or any(
        x not in ("metropolis", "second_order") for x in cfg.schemes
    ):
        raise ValueError("schemes must contain distinct supported propagators")
    for dt in steps:
        prod_steps = ceil(cfg.production_tau / dt)
        blk_steps = ceil(cfg.block_tau / dt)
        if prod_steps < 2 * blk_steps:
            raise ValueError("production_tau must allow >= 2 full blocks at each dt")
        if ceil(cfg.equilibration_tau / dt) < 1:
            raise ValueError("equilibration_tau too short")
    length = cfg.cutoff * (cfg.n_particles / cfg.reduced_density) ** (1.0 / 3.0)
    cell = CubicCell(length)
    potential = SoftCorePairPotential(
        cell, height=cfg.gamma * cfg.kinetic_coefficient / cfg.cutoff**2,
        cutoff=cfg.cutoff,
    )
    modes = canonical_modes(cfg.mode_indices)
    return cell, potential, modes


def _trial(cell, cfg, modes, ja, order):
    return DensityWaveJastrowTrial(
        cell, jastrow_amplitude=float(ja), order_amplitude=float(order),
        cutoff=cfg.cutoff, indices=modes,
    )


def _vmc_config(cfg, walkers, seed, steps=None, burn_in=None):
    return VMCConfig(
        n_walkers=walkers,
        configuration_shape=(cfg.n_particles, 3),
        kinetic_coefficient=cfg.kinetic_coefficient,
        proposal_mode="particle", n_steps=cfg.vmc_steps if steps is None else steps,
        burn_in=cfg.vmc_burn_in if burn_in is None else burn_in,
        sample_every=cfg.vmc_sample_every,
        proposal_sigma=cfg.proposal_sigma, adaptive_proposal=True, seed=seed,
    )


def _uniform_start(cell, cfg, n_walkers, seed):
    return np.random.default_rng(seed).uniform(
        0, cell.length, size=(n_walkers, cfg.n_particles, 3)
    )


def _mean_stderr(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("need >= 2 finite independent replica means")
    return float(np.mean(values)), float(np.std(values, ddof=1) / np.sqrt(len(values)))


def _save_csv(path, cls, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[v.name for v in fields(cls)])
        writer.writeheader()
        writer.writerows(asdict(r) for r in records)
    return path


def fit_bias_plane(points: list[DMCReplica], *, bootstrap_samples=200, seed=1):
    """Fit E(dt,W)=E0+a*dt+b/W, bootstrap across independent replicas.

    Each cell of the (dt, W) grid must have the same number of independent
    replica runs. The estimated standard error captures *replica sampling only*
    and does not quantify model inadequacy or uncontrolled systematic errors.
    """
    if not points:
        raise ValueError("no replica results")
    branches = {(p.branch, p.scheme) for p in points}
    if len(branches) != 1:
        raise ValueError("fit requires one branch and scheme at a time")
    if not isinstance(bootstrap_samples, int) or bootstrap_samples < 2:
        raise ValueError("bootstrap_samples must be >= 2")
    grid = sorted({(p.dt, p.n_walkers) for p in points})
    dt_values = {d for d, _ in grid}
    n_values = {w for _, w in grid}
    if len(dt_values) < 2 or len(n_values) < 2 or len(grid) != len(dt_values)*len(n_values):
        raise ValueError("complete two-factor grid with >= 2 values per factor required")
    grouped = {}
    for p in points:
        grouped.setdefault((p.dt, p.n_walkers), []).append(p)
    n_replicas = len(grouped[grid[0]])
    if n_replicas < 2 or any(len(grouped[key]) != n_replicas for key in grid):
        raise ValueError("at least two replicas per grid cell, with equal counts")
    if any(len({p.replica for p in grouped[key]}) != n_replicas for key in grid):
        raise ValueError("replica indices must be distinct within each cell")
    dt_scale = max(dt_values)
    n_scale = min(n_values)
    x = np.array([[1.0, dt/dt_scale, n_scale/w] for dt, w in grid], dtype=float)
    if np.linalg.matrix_rank(x) != 3:
        raise ValueError("design matrix is rank deficient")
    rng = np.random.default_rng(seed)
    results = []
    branch, scheme = next(iter(branches))
    for estimator, attr in (("mixed", "mixed_energy_per_particle"),
                            ("growth", "growth_energy_per_particle")):
        by_cell = [np.array([getattr(p, attr) for p in grouped[key]], dtype=float)
                   for key in grid]
        if not all(np.all(np.isfinite(cell_values)) for cell_values in by_cell):
            raise ValueError("nonfinite replica energies")
        y = np.array([a.mean() for a in by_cell])
        coeff = np.linalg.lstsq(x, y, rcond=None)[0]
        boots = np.empty(bootstrap_samples)
        for index in range(bootstrap_samples):
            means = np.array([a[rng.integers(len(a), size=len(a))].mean()
                              for a in by_cell])
            boots[index] = np.linalg.lstsq(x, means, rcond=None)[0][0]
        results.append(Extrapolation(
            branch=branch, scheme=scheme, estimator=estimator,
            intercept=float(coeff[0]), bootstrap_replica_stderr=float(boots.std(ddof=1)),
            dt_slope=float(coeff[1] / dt_scale),
            inverse_walker_slope=float(coeff[2] * n_scale),
            n_grid_points=len(grid), bootstrap_samples=bootstrap_samples,
        ))
    return results


def _plot_bias(rows: list[DMCSummary], vmc_reference: VMCValidation, extrapolations, path):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    for n in sorted({r.n_walkers for r in rows}):
        x = sorted((r for r in rows if r.n_walkers == n), key=lambda r: r.dt)
        axes[0].errorbar([r.dt for r in x], [r.mixed_energy_per_particle for r in x],
                         yerr=[r.mixed_replica_stderr for r in x],
                         marker="o", capsize=2, label=f"W={n}")
    axes[0].axhline(vmc_reference.energy_per_particle, linestyle="--", color="0.5",
                    label="VMC validation")
    axes[0].set(xlabel="DMC timestep dt", ylabel="Mixed energy / particle")
    axes[0].legend(fontsize=8)
    for dt in sorted({r.dt for r in rows}):
        x = sorted((r for r in rows if r.dt == dt), key=lambda r: 1/r.n_walkers)
        axes[1].errorbar([1.0/r.n_walkers for r in x],
                         [r.mixed_energy_per_particle for r in x],
                         yerr=[r.mixed_replica_stderr for r in x],
                         marker="o", capsize=2, label=f"dt={dt:g}")
    axes[1].set(xlabel="Inverse DMC population 1/W", ylabel="Mixed energy / particle")
    axes[1].legend(fontsize=8)
    for a in axes:
        a.grid(alpha=0.25)
    name = rows[0].branch
    scheme = rows[0].scheme
    fig.suptitle(f"DMC diagnostics: {name}, {scheme} — descriptive only")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def run_dmc_benchmark(cfg: DMCBenchmarkConfig, output_dir="outputs/bosons/dmc_benchmark"):
    """Run VMC selection/validation and DMC two-factor replica benchmark.

    Returns (training_candidates, vmc_validations, dmc_summaries, extrapolations).
    Output paths hold full per-replica data and a protocol manifest.
    """
    cell, potential, modes = validate_benchmark(cfg)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    training = []
    # Preselect trial parameters using training VMC only.
    candidate_id = 0
    for ja in cfg.jastrow_amplitudes:
        for oa in cfg.order_amplitudes:
            branch = "liquid" if oa == 0 else "modulated"
            seed = cfg.seed + 10000 + candidate_id * 100
            trial = _trial(cell, cfg, modes, ja, oa)
            res = run_vmc(potential, trial, _vmc_config(cfg, cfg.vmc_walkers, seed),
                          initial_walkers=_uniform_start(cell, cfg, cfg.vmc_walkers, seed + 1))
            estimate = block_statistics(res.energy_history, block_size=cfg.vmc_block_size)
            training.append(TrainingCandidate(
                branch, float(ja), float(oa), seed,
                float(estimate.mean / cfg.n_particles),
                float(estimate.stderr / cfg.n_particles),
                float(res.production_acceptance),
            ))
            candidate_id += 1
    selected = {branch: min((r for r in training if r.branch == branch),
                            key=lambda r: r.energy_per_particle)
                for branch in ("liquid", "modulated")}
    vmc_validations = []
    replicas = []
    summaries = []
    extrapolations = []
    for branch_id, (branch, selection) in enumerate(selected.items()):
        trial = _trial(cell, cfg, modes, selection.jastrow_amplitude,
                       selection.order_amplitude)
        val_e, val_a = [], []
        for r in range(cfg.replicas):
            seed = cfg.seed + 100000 + branch_id * 10000 + r * 100
            run = run_vmc(potential, trial, _vmc_config(cfg, cfg.vmc_walkers, seed),
                          initial_walkers=_uniform_start(cell, cfg, cfg.vmc_walkers, seed+1))
            val_e.append(float(run.energy_history.mean() / cfg.n_particles))
            val_a.append(float(run.production_acceptance))
        energy, stderr = _mean_stderr(val_e)
        vmc_reference = VMCValidation(
            branch, selection.jastrow_amplitude, selection.order_amplitude,
            cfg.replicas, energy, stderr, float(np.mean(val_a)),
        )
        vmc_validations.append(vmc_reference)
        print(f"VMC {branch}: E/N = {energy:.7g} +/- {stderr:.2g}; "
              f"a={selection.jastrow_amplitude:g}, lambda={selection.order_amplitude:g}")
        for scheme_id, scheme in enumerate(cfg.schemes):
            scheme_replicas = []
            scheme_summaries = []
            for n_index, nw in enumerate(cfg.walker_counts):
                for dt_index, dt in enumerate(cfg.dt_values):
                    cell_replicas = []
                    eq_steps = ceil(cfg.equilibration_tau / dt)
                    prod_steps = ceil(cfg.production_tau / dt)
                    blocks = ceil(cfg.block_tau / dt)
                    for rep in range(cfg.replicas):
                        grid_run_id = (
                            (((branch_id * len(cfg.schemes) + scheme_id)
                              * len(cfg.walker_counts) + n_index)
                             * len(cfg.dt_values) + dt_index)
                            * cfg.replicas + rep
                        )
                        seed = cfg.seed + 1_000_000 + grid_run_id * 100
                        warmup_seed = seed + 37
                        # Fresh, independently seeded VMC equilibrated |Psi_T|^2 starts.
                        start_vmc = run_vmc(
                            potential, trial,
                            _vmc_config(cfg, nw, warmup_seed,
                                        steps=cfg.warmup_sweeps,
                                        burn_in=cfg.warmup_sweeps // 2),
                            initial_walkers=_uniform_start(cell, cfg, nw, warmup_seed + 1),
                        )
                        dmc_cfg = DMCConfig(
                            n_walkers=nw, configuration_shape=(cfg.n_particles, 3),
                            kinetic_coefficient=cfg.kinetic_coefficient,
                            n_steps=eq_steps+prod_steps,
                            equilibration_steps=eq_steps, adaptation_steps=0,
                            dt=dt, adaptive_dt=False, scheme=scheme,
                            log_interval=0, seed=seed,
                        )
                        run = run_dmc_importance(
                            potential, trial, dmc_cfg,
                            initial_walkers=start_vmc.walkers,
                        )
                        mixed = block_statistics(run.local_energy_history, block_size=blocks)
                        growth = block_statistics(run.reference_energy_history, block_size=blocks)
                        item = DMCReplica(
                            branch, scheme, dt, nw, rep, seed, warmup_seed,
                            dmc_cfg.n_steps, eq_steps, prod_steps,
                            eq_steps*dt, prod_steps*dt,
                            float(mixed.mean / cfg.n_particles),
                            float(mixed.stderr / cfg.n_particles),
                            float(growth.mean / cfg.n_particles),
                            float(growth.stderr / cfg.n_particles),
                            float(run.production_acceptance),
                        )
                        replicas.append(item)
                        scheme_replicas.append(item)
                        cell_replicas.append(item)
                    em, ems = _mean_stderr([p.mixed_energy_per_particle for p in cell_replicas])
                    eg, egs = _mean_stderr([p.growth_energy_per_particle for p in cell_replicas])
                    row = DMCSummary(
                        branch, scheme, dt, nw, cfg.replicas,
                        em, ems, eg, egs,
                        float(np.nanmean([p.acceptance for p in cell_replicas])) if scheme=="metropolis" else float("nan"),
                    )
                    summaries.append(row)
                    scheme_summaries.append(row)
                    print(f"  DMC {branch:9s} {scheme:12s} dt={dt:g} W={nw:4d}: "
                          f"E_mix/N={em:.7g} +/- {ems:.2g}")
            fitted = fit_bias_plane(
                scheme_replicas, bootstrap_samples=cfg.bootstrap_samples,
                seed=cfg.seed + 90_000_000 + branch_id*1000 + scheme_id,
            )
            extrapolations.extend(fitted)
            _plot_bias(scheme_summaries, vmc_reference, fitted,
                       output / f"bias_{branch}_{scheme}.png")
    _save_csv(output/"training.csv", TrainingCandidate, training)
    _save_csv(output/"vmc_validation.csv", VMCValidation, vmc_validations)
    _save_csv(output/"dmc_replicas.csv", DMCReplica, replicas)
    _save_csv(output/"dmc_summary.csv", DMCSummary, summaries)
    _save_csv(output/"extrapolation.csv", Extrapolation, extrapolations)
    manifest = {
        "configuration": asdict(cfg),
        "cell_length": cell.length,
        "potential_height": potential.height,
        "protocol": "independent VMC selection/validation; DMC dt and population grid",
        "seed_policy": "deterministic nonoverlapping seed ranges for selections and replicas",
        "python": platform.python_version(),
        "numpy": np.__version__,
        "caveat": "Descriptive linear-plane fit: bootstrap SEM excludes systematic error",
    }
    (output/"protocol.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    return training, vmc_validations, summaries, extrapolations
