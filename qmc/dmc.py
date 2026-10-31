from __future__ import annotations

from dataclasses import dataclass, replace  
from typing import Sequence

import numpy as np

from .evaluation import evaluate_trial_quantities
from .resampling import systematic_resample
from .stats import Estimate, block_statistics
from .propagators import drift_diffusion_second_order, drift_diffusion_metropolis
from .configuration import initial_configuration_walkers, require_positive_kinetic

Array = np.ndarray


@dataclass(frozen=True)
class DMCTimestepPoint:
    scheme: str
    dt: float
    mixed: Estimate
    growth: Estimate
    production_acceptance: float


@dataclass(frozen=True)
class DMCConfig:
    n_walkers: int = 30_000
    n_steps: int = 6_000
    adaptation_steps: int = 750
    equilibration_steps: int = 1_500
    dt: float = 0.01
    adaptive_dt: bool = True
    target_acceptance: float = 0.98
    adapt_interval: int = 25
    adapt_rate: float = 0.5
    dt_min: float = 5e-4
    dt_max: float = 0.03
    initial_sigma: float = 2.0
    seed: int = 12345
    log_interval: int = 500
    scheme: str = "metropolis"
    configuration_shape: tuple[int, ...] = (3,)
    kinetic_coefficient: float = 0.5

@dataclass
class DMCResult:
    walkers: Array
    reference_energy_history: Array
    local_energy_history: Array
    acceptance_history: Array
    dt_history: Array
    final_dt: float
    adaptation_acceptance: float
    fixed_equilibration_acceptance: float
    production_acceptance: float


def _mean_or_nan(values):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    finite = values[
        np.isfinite(values)
    ]

    if finite.size == 0:
        return float("nan")

    return float(
        np.mean(finite)
    )


def _initial_walkers(config: DMCConfig, rng, initial_walkers=None):
    return initial_configuration_walkers(
        n_walkers=config.n_walkers,
        configuration_shape=config.configuration_shape,
        initial_sigma=config.initial_sigma,
        rng=rng,
        initial_walkers=initial_walkers,
    )


def _adapt_dt(
    dt,
    mean_acceptance,
    adaptation_number,
    config,
):
    # Acceptance difference over a symmetric tolerance scale
    tolerance = 0.02

    error = (
        mean_acceptance
        - config.target_acceptance
    ) / tolerance

    # Treat deviations larger than tolerance equally
    # in either direction.
    error = float(
        np.clip(
            error,
            -1.0,
            1.0,
        )
    )

    gamma = (
        config.adapt_rate
        / np.sqrt(adaptation_number)
    )

    factor = float(
        np.exp(
            gamma * error
        )
    )

    factor = float(
        np.clip(
            factor,
            0.8,
            1.25,
        )
    )

    new_dt = float(
        np.clip(
            dt * factor,
            config.dt_min,
            config.dt_max,
        )
    )

    return (
        new_dt,
        error,
        gamma,
        factor,
    )


def run_dmc_importance(
    potential,
    trial,
    config: DMCConfig = DMCConfig(),
    *,
    initial_walkers=None,
) -> DMCResult:
    """
    Importance-sampled fixed-population DMC.

    Supported schemes:

        "metropolis"
            First-order drift-diffusion proposal with
            Metropolis-Hastings correction.

        "second_order"
            Symmetric split propagation:

                diffusion(dt/2)
                -> RK2 drift(dt)
                -> diffusion(dt/2)

            together with symmetric endpoint branching.
    """

    if not (
        0
        <= config.adaptation_steps
        <= config.equilibration_steps
        < config.n_steps
    ):
        raise ValueError(
            "Require "
            "0 <= adaptation_steps "
            "<= equilibration_steps "
            "< n_steps"
        )

    scheme = config.scheme.lower()

    if scheme not in {
        "metropolis",
        "second_order",
    }:
        raise ValueError(
            f"Unknown DMC scheme: {config.scheme!r}. "
            "Expected 'metropolis' or 'second_order'."
        )

    # Adaptive dt currently relies on an acceptance rate.
    #
    # The split second-order propagator does not have a
    # Metropolis accept/reject step, so that controller does
    # not apply.
    if scheme == "second_order":
        if config.adaptive_dt:
            raise ValueError(
                "adaptive_dt is not supported for "
                "scheme='second_order'. "
                "Use adaptive_dt=False."
            )

        if config.adaptation_steps != 0:
            raise ValueError(
                "For scheme='second_order', set "
                "adaptation_steps=0."
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

    (
        log_psi,
        drift,
        e_local,
    ) = evaluate_trial_quantities(
        potential,
        trial,
        walkers,
        kinetic_coefficient=config.kinetic_coefficient,
    )

    e_ref = float(
        np.mean(e_local)
    )

    dt = float(
        config.dt
    )

    reference_energy_history = []
    local_energy_history = []

    acceptance_history = []
    dt_history = []

    acceptance_window = []

    # ========================================================
    # DMC loop
    # ========================================================

    for step in range(
        config.n_steps
    ):

        dt_history.append(
            dt
        )

        # ----------------------------------------------------
        # Keep the OLD local energy.
        #
        # Symmetric branching needs both endpoints:
        #
        #     E_L(R_old)
        #     E_L(R_new)
        # ----------------------------------------------------

        old_e_local = e_local

        # ====================================================
        # 1. Propagation
        # ====================================================

        if scheme == "metropolis":

            (
                moved_walkers,
                moved_log_psi,
                moved_drift,
                moved_e_local,
                rate,
            ) = drift_diffusion_metropolis(
                walkers,
                log_psi,
                drift,
                e_local,
                potential,
                trial,
                dt,
                rng,
                kinetic_coefficient=config.kinetic_coefficient,
            )

        else:

            (
                moved_walkers,
                moved_log_psi,
                moved_drift,
                moved_e_local,
                rate,
            ) = drift_diffusion_second_order(
                walkers,
                potential,
                trial,
                dt,
                rng,
                kinetic_coefficient=config.kinetic_coefficient,
            )

        acceptance_history.append(
            rate
        )

        if np.isfinite(rate):
            acceptance_window.append(
                rate
            )

        # ====================================================
        # 2. Symmetric endpoint branching
        #
        # exp[
        #   -dt/2 * (
        #       E_L(old)
        #       + E_L(new)
        #       - 2 E_ref
        #   )
        # ]
        # ====================================================

        effective_e = 0.5 * (
            old_e_local
            + moved_e_local
        )

        log_weights = (
            -dt
            * (
                effective_e
                - e_ref
            )
        )

        # ----------------------------------------------------
        # Numerical stabilization
        # ----------------------------------------------------

        shift = float(
            np.max(log_weights)
        )

        weights = np.exp(
            log_weights
            - shift
        )

        mean_weight = float(
            np.mean(weights)
        )

        if (
            not np.isfinite(mean_weight)
            or mean_weight <= 0.0
        ):
            raise FloatingPointError(
                "Invalid mean DMC branching weight."
            )

        log_mean_weight = (
            shift
            + np.log(mean_weight)
        )

        # ====================================================
        # 3. Reference / growth energy
        # ====================================================

        e_ref -= (
            log_mean_weight
            / dt
        )

        # ====================================================
        # 4. Fixed-population stochastic reconfiguration
        # ====================================================

        weight_sum = float(
            np.sum(weights)
        )

        if (
            not np.isfinite(weight_sum)
            or weight_sum <= 0.0
        ):
            raise FloatingPointError(
                "Invalid total DMC branching weight."
            )

        probabilities = (
            weights
            / weight_sum
        )

        indices = systematic_resample(
            probabilities,
            rng,
        )

        walkers = moved_walkers[
            indices
        ]

        log_psi = moved_log_psi[
            indices
        ]

        drift = moved_drift[
            indices
        ]

        e_local = moved_e_local[
            indices
        ]

        # ====================================================
        # 5. Adaptive dt
        #
        # Metropolis scheme only.
        # ====================================================

        if (
            scheme == "metropolis"
            and config.adaptive_dt
            and step < config.adaptation_steps
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

            adaptation_number = (
                (step + 1)
                // config.adapt_interval
            )

            (
                dt,
                error,
                gamma,
                factor,
            ) = _adapt_dt(
                dt,
                mean_acceptance,
                adaptation_number,
                config,
            )

            print(
                f"  adapting dt: "
                f"accept={mean_acceptance:.4f} "
                f"target={config.target_acceptance:.4f} "
                f"error={error:+.4f} "
                f"factor={factor:.4f} "
                f"dt={dt:.6f}"
            )

        # ====================================================
        # 6. Production measurements
        # ====================================================

        if step >= config.equilibration_steps:

            reference_energy_history.append(
                e_ref
            )

            # Mixed estimator.
            local_energy_history.append(
                float(
                    np.mean(e_local)
                )
            )

        # ====================================================
        # 7. Logging
        # ====================================================

        if (
            config.log_interval
            and step % config.log_interval == 0
        ):

            if np.isfinite(rate):
                rate_text = (
                    f"{rate:7.4f}"
                )
            else:
                rate_text = "    n/a"

            print(
                f"step={step:5d} "
                f"E_ref={e_ref:12.6f} "
                f"<E_L>={np.mean(e_local):12.6f} "
                f"accept={rate_text} "
                f"dt={dt:.6f}"
            )

    # ========================================================
    # Results
    # ========================================================

    acceptance_history = np.asarray(
        acceptance_history,
        dtype=np.float64,
    )

    return DMCResult(
        walkers=walkers,

        reference_energy_history=np.asarray(
            reference_energy_history,
            dtype=np.float64,
        ),

        local_energy_history=np.asarray(
            local_energy_history,
            dtype=np.float64,
        ),

        acceptance_history=(
            acceptance_history
        ),

        dt_history=np.asarray(
            dt_history,
            dtype=np.float64,
        ),

        final_dt=dt,

        adaptation_acceptance=(
            _mean_or_nan(
                acceptance_history[
                    :config.adaptation_steps
                ]
            )
        ),

        fixed_equilibration_acceptance=(
            _mean_or_nan(
                acceptance_history[
                    config.adaptation_steps:
                    config.equilibration_steps
                ]
            )
        ),

        production_acceptance=(
            _mean_or_nan(
                acceptance_history[
                    config.equilibration_steps:
                ]
            )
        ),
    )


def scan_dmc_dt(
    dt_values: Sequence[float],
    *,
    potential,
    trial,
    config: DMCConfig,
    schemes: Sequence[str] = (
        "metropolis",
        "second_order",
    ),
    equilibration_tau: float = 15.0,
    production_tau: float = 30.0,
    block_tau: float = 2.0,
) -> list[DMCTimestepPoint]:
    """
    Fixed-time-step DMC convergence scan.

    All dt values use approximately the same:

        equilibration imaginary time
        production imaginary time
        block imaginary time

    This is essential for a fair timestep comparison.
    """

    if equilibration_tau <= 0:
        raise ValueError(
            "equilibration_tau must be positive"
        )

    if production_tau <= 0:
        raise ValueError(
            "production_tau must be positive"
        )

    if block_tau <= 0:
        raise ValueError(
            "block_tau must be positive"
        )

    valid_schemes = {
        "metropolis",
        "second_order",
    }

    for scheme in schemes:
        if scheme not in valid_schemes:
            raise ValueError(
                f"Unknown DMC scheme {scheme!r}"
            )

    points = []

    for i, dt_value in enumerate(
        dt_values
    ):
        dt = float(dt_value)

        if dt <= 0:
            raise ValueError(
                "DMC dt must be positive."
            )

        # ====================================================
        # Convert physical imaginary-time intervals to steps
        # ====================================================

        equilibration_steps = int(
            np.ceil(
                equilibration_tau
                / dt
            )
        )

        production_steps = int(
            np.ceil(
                production_tau
                / dt
            )
        )

        n_steps = (
            equilibration_steps
            + production_steps
        )

        # Block size should also correspond to approximately
        # the same imaginary-time interval for every dt.
        block_size = max(
            1,
            int(
                np.ceil(
                    block_tau
                    / dt
                )
            ),
        )

        n_blocks = (
            production_steps
            // block_size
        )

        if n_blocks < 2:
            raise ValueError(
                f"Not enough blocks for dt={dt}. "
                f"production_steps={production_steps}, "
                f"block_size={block_size}"
            )

        actual_equilibration_tau = (
            equilibration_steps
            * dt
        )

        actual_production_tau = (
            production_steps
            * dt
        )

        actual_block_tau = (
            block_size
            * dt
        )

        # ====================================================
        # Run both propagators at this dt
        # ====================================================

        for scheme in schemes:

            run_config = replace(
                config,

                dt=dt,

                n_steps=n_steps,
                equilibration_steps=(
                    equilibration_steps
                ),

                adaptation_steps=0,
                adaptive_dt=False,

                scheme=scheme,

                # Same initial seed for the two schemes
                # at the same dt.
                seed=config.seed + i,
            )

            print()
            print(
                "========================================"
            )

            print(
                f"Fixed-dt DMC: "
                f"scheme={scheme}, "
                f"dt={dt:.6f}"
            )

            print(
                f"equilibration: "
                f"{equilibration_steps} steps "
                f"(tau={actual_equilibration_tau:.3f})"
            )

            print(
                f"production:    "
                f"{production_steps} steps "
                f"(tau={actual_production_tau:.3f})"
            )

            print(
                f"block:         "
                f"{block_size} steps "
                f"(tau={actual_block_tau:.3f}), "
                f"{n_blocks} blocks"
            )

            print(
                "========================================"
            )

            result = run_dmc_importance(
                potential,
                trial,
                run_config,
            )

            mixed = block_statistics(
                result.local_energy_history,
                block_size=block_size,
            )

            growth = block_statistics(
                result.reference_energy_history,
                block_size=block_size,
            )

            point = DMCTimestepPoint(
                scheme=scheme,
                dt=dt,
                mixed=mixed,
                growth=growth,
                production_acceptance=(
                    result.production_acceptance
                ),
            )

            points.append(
                point
            )

            if np.isfinite(
                result.production_acceptance
            ):
                acceptance_text = (
                    f"{result.production_acceptance:.4f}"
                )
            else:
                acceptance_text = "n/a"

            print(
                f"scheme={scheme:12s} "
                f"dt={dt:.6f} "
                f"mixed={mixed.mean:.8f} "
                f"+/- {mixed.stderr:.8f} "
                f"growth={growth.mean:.8f} "
                f"+/- {growth.stderr:.8f} "
                f"accept={acceptance_text}"
            )

    return points