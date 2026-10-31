from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Sequence
import csv

import numpy as np

from .evaluation import evaluate_local_energy
from .stats import Estimate, block_statistics
from .configuration import initial_configuration_walkers, require_positive_kinetic

Array = np.ndarray


@dataclass
class ScanPoint2D:
    alpha: float
    beta: float
    estimate: object
    production_acceptance: float


def scan_alpha_beta(
    alphas,
    betas,
    *,
    trial_factory,
    potential,
    config,
    block_size=20,
):
    """
    Grid scan over alpha and beta.

    trial_factory must have signature:

        trial_factory(alpha, beta) -> trial
    """

    points = []

    run_index = 0

    for alpha in alphas:
        for beta in betas:

            trial = trial_factory(
                alpha,
                beta,
            )

            # Give each grid point a reproducible,
            # but different, random seed.
            run_config = replace(
                config,
                seed=config.seed + run_index,
            )

            result = run_vmc(
                potential,
                trial,
                run_config,
            )

            estimate = block_statistics(
                result.energy_history,
                block_size=block_size,
            )

            point = ScanPoint2D(
                alpha=float(alpha),
                beta=float(beta),
                estimate=estimate,
                production_acceptance=(
                    result.production_acceptance
                ),
            )

            points.append(point)

            print(
                f"alpha={alpha:8.5f} "
                f"beta={beta:8.5f} "
                f"E={estimate.mean: .8f} "
                f"+/- {estimate.stderr:.8f} "
                f"accept={result.production_acceptance:.4f}"
            )

            run_index += 1

    return points


@dataclass(frozen=True)
class VMCConfig:
    n_walkers: int = 1_000
    n_steps: int = 1_000
    burn_in: int = 250
    sample_every: int = 2
    proposal_sigma: float = 0.8
    adaptive_proposal: bool = True
    target_acceptance: float = 0.50
    adapt_interval: int = 25
    adapt_rate: float = 0.5
    initial_sigma: float = 1.5
    seed: int = 12345
    configuration_shape: tuple[int, ...] = (3,)
    kinetic_coefficient: float = 0.5


@dataclass
class VMCResult:
    walkers: Array

    energy_history: Array
    acceptance_history: Array
    proposal_sigma_history: Array

    local_energy_mean: float
    local_energy_variance: float
    local_energy_std: float
    n_local_energy_samples: int

    production_acceptance: float

@dataclass(frozen=True)
class VMCScanPoint:
    parameter: float
    estimate: Estimate
    production_acceptance: float
    proposal_sigma: float


def _initial_walkers(config: VMCConfig, rng, initial_walkers=None):
    return initial_configuration_walkers(
        n_walkers=config.n_walkers,
        configuration_shape=config.configuration_shape,
        initial_sigma=config.initial_sigma,
        rng=rng,
        initial_walkers=initial_walkers,
    )


def run_vmc(
    potential,
    trial,
    config: VMCConfig = VMCConfig(),
    *,
    initial_walkers=None,
) -> VMCResult:
    """
    Metropolis VMC sampling |Psi_T|^2.

    For many-body configurations, this version moves all coordinates at once;
    particle-by-particle proposals will be introduced in a later patch.

    The energy history contains the mean local energy
    across walkers at each measurement step.

    The local-energy variance is calculated from the
    individual walker local energies, not from the
    already-averaged energy history.
    """

    if config.burn_in >= config.n_steps:
        raise ValueError(
            "burn_in must be smaller than n_steps"
        )

    if config.sample_every <= 0:
        raise ValueError(
            "sample_every must be positive"
        )

    require_positive_kinetic(config.kinetic_coefficient)

    rng = np.random.default_rng(
        config.seed
    )

    walkers = _initial_walkers(
        config,
        rng,
        initial_walkers,
    )

    n_walkers = len(walkers)

    log_psi = np.asarray(
        trial.log_psi(walkers),
        dtype=np.float64,
    )

    sigma = float(
        config.proposal_sigma
    )

    # ========================================================
    # Histories
    # ========================================================

    energies = []
    acceptance = []
    sigma_history = []
    acceptance_window = []

    # ========================================================
    # Raw local-energy accumulators
    #
    # These are used for the zero-variance diagnostic.
    # ========================================================

    local_energy_sum = 0.0
    local_energy_sum_sq = 0.0
    local_energy_count = 0

    # ========================================================
    # Metropolis loop
    # ========================================================

    for step in range(
        config.n_steps
    ):

        sigma_history.append(
            sigma
        )

        # ----------------------------------------------------
        # Proposal
        # ----------------------------------------------------

        proposed = (
            walkers
            + sigma
            * rng.normal(
                size=walkers.shape
            )
        )

        log_psi_proposed = np.asarray(
            trial.log_psi(proposed),
            dtype=np.float64,
        )

        # ----------------------------------------------------
        # Metropolis acceptance for |Psi|^2
        # ----------------------------------------------------

        log_acceptance = np.minimum(
            0.0,
            2.0
            * (
                log_psi_proposed
                - log_psi
            ),
        )

        accepted = (
            np.log(
                rng.random(
                    n_walkers
                )
            )
            < log_acceptance
        )

        # ----------------------------------------------------
        # Accept moves
        # ----------------------------------------------------

        walkers[accepted] = (
            proposed[accepted]
        )

        log_psi[accepted] = (
            log_psi_proposed[
                accepted
            ]
        )

        rate = float(
            np.mean(accepted)
        )

        acceptance.append(
            rate
        )

        acceptance_window.append(
            rate
        )

        # ----------------------------------------------------
        # Adapt proposal width during burn-in only
        # ----------------------------------------------------

        if (
            config.adaptive_proposal
            and step < config.burn_in
            and (
                step + 1
            ) % config.adapt_interval == 0
        ):

            mean_acceptance = float(
                np.mean(
                    acceptance_window[
                        -config.adapt_interval:
                    ]
                )
            )

            adjustment = np.exp(
                config.adapt_rate
                * (
                    mean_acceptance
                    - config.target_acceptance
                )
            )

            adjustment = float(
                np.clip(
                    adjustment,
                    0.8,
                    1.25,
                )
            )

            sigma *= adjustment

        # ----------------------------------------------------
        # Measurements
        # ----------------------------------------------------

        if (
            step >= config.burn_in
            and (
                step
                - config.burn_in
            ) % config.sample_every == 0
        ):

            e_local = (
                evaluate_local_energy(
                    potential,
                    trial,
                    walkers,
                    kinetic_coefficient=config.kinetic_coefficient,
                )
            )

            # -----------------------------------------------
            # Energy history:
            #
            # one mean energy per MC measurement step.
            #
            # This is what block_statistics() uses.
            # -----------------------------------------------

            mean_e = float(
                np.mean(e_local)
            )

            energies.append(
                mean_e
            )

            # -----------------------------------------------
            # Zero-variance diagnostic:
            #
            # accumulate ALL walker local energies.
            # -----------------------------------------------

            local_energy_sum += float(
                np.sum(e_local)
            )

            local_energy_sum_sq += float(
                np.dot(
                    e_local,
                    e_local,
                )
            )

            local_energy_count += (
                e_local.size
            )

    # ========================================================
    # Final checks
    # ========================================================

    if local_energy_count < 2:
        raise ValueError(
            "Not enough local-energy samples: "
            f"count={local_energy_count}. "
            f"n_steps={config.n_steps}, "
            f"burn_in={config.burn_in}, "
            f"sample_every={config.sample_every}"
        )

    # ========================================================
    # Local-energy distribution statistics
    # ========================================================

    local_energy_mean = (
        local_energy_sum
        / local_energy_count
    )

    local_energy_variance = (
        local_energy_sum_sq
        - local_energy_count
        * local_energy_mean**2
    ) / (
        local_energy_count - 1
    )

    # Protect against a tiny negative result caused purely
    # by floating-point roundoff.
    local_energy_variance = max(
        0.0,
        float(local_energy_variance),
    )

    local_energy_std = float(
        np.sqrt(
            local_energy_variance
        )
    )

    acceptance_array = np.asarray(
        acceptance,
        dtype=np.float64,
    )

    energy_array = np.asarray(
        energies,
        dtype=np.float64,
    )

    sigma_array = np.asarray(
        sigma_history,
        dtype=np.float64,
    )

    production_acceptance = float(
        np.mean(
            acceptance_array[
                config.burn_in:
            ]
        )
    )

    print(
        f"VMC samples: "
        f"steps={len(energies)}, "
        f"walker samples={local_energy_count}"
    )

    return VMCResult(
        walkers=walkers,

        energy_history=energy_array,

        acceptance_history=(
            acceptance_array
        ),

        proposal_sigma_history=(
            sigma_array
        ),

        local_energy_mean=float(
            local_energy_mean
        ),

        local_energy_variance=(
            local_energy_variance
        ),

        local_energy_std=(
            local_energy_std
        ),

        n_local_energy_samples=int(
            local_energy_count
        ),

        production_acceptance=(
            production_acceptance
        ),
    )

def scan_parameter(
    values: Sequence[float],
    *,
    trial_factory: Callable[[float], object],
    potential,
    config: VMCConfig,
    block_size: int = 20,
) -> list[VMCScanPoint]:
    """Generic one-dimensional VMC parameter scan."""
    points: list[VMCScanPoint] = []

    for i, value in enumerate(values):
        trial = trial_factory(float(value))
        run_config = VMCConfig(**{**config.__dict__, "seed": config.seed + i})
        result = run_vmc(potential, trial, run_config)
        estimate = block_statistics(result.energy_history, block_size=block_size)
        prod_accept = (
            result.production_acceptance
        )
        point = VMCScanPoint(
            parameter=float(value),
            estimate=estimate,
            production_acceptance=prod_accept,
            proposal_sigma=float(result.proposal_sigma_history[-1]),
        )
        points.append(point)

        print(
            f"param={value:8.5f} "
            f"E={estimate.mean: .8f} +/- {estimate.stderr:.8f} "
            f"accept={prod_accept:.4f} sigma={point.proposal_sigma:.4f}"
        )

    return points


def save_vmc_2d_scan(
    points,
    path,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "alpha,beta,energy,stderr,acceptance\n"
        )

        for p in points:

            f.write(
                f"{p.alpha},"
                f"{p.beta},"
                f"{p.estimate.mean},"
                f"{p.estimate.stderr},"
                f"{p.production_acceptance}\n"
            )


def save_dmc_dt_scan(
    points,
    output,
):
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "scheme",
            "dt",
            "mixed_energy",
            "mixed_stderr",
            "growth_energy",
            "growth_stderr",
            "acceptance",
        ])

        for p in points:
            writer.writerow([
                p.scheme,
                p.dt,
                p.mixed.mean,
                p.mixed.stderr,
                p.growth.mean,
                p.growth.stderr,
                p.production_acceptance,
            ])