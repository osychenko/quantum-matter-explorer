import numpy as np

from .evaluation import evaluate_trial_quantities
from .configuration import configuration_norm2, walker_broadcast_mask, require_positive_kinetic


def drift_diffusion_metropolis(
    walkers,
    log_psi,
    drift,
    e_local,
    potential,
    trial,
    dt,
    rng,
    *,
    kinetic_coefficient=0.5,
):
    """
    First-order drift-diffusion proposal with
    Metropolis-Hastings correction.

    Proposal:

        R' = R + 2*D*dt * grad(log Psi_T) + sqrt(2*D*dt) * eta

    Returns the post-accept/reject state.
    """

    n_walkers = len(walkers)
    D = require_positive_kinetic(kinetic_coefficient)
    drift_dt = 2.0 * D * dt

    # --------------------------------------------------------
    # Drift-diffusion proposal
    # --------------------------------------------------------

    proposed = (
        walkers
        + drift_dt * drift
        + np.sqrt(2.0 * D * dt)
        * rng.normal(size=walkers.shape)
    )

    (
        log_psi_p,
        drift_p,
        e_local_p,
    ) = evaluate_trial_quantities(
        potential,
        trial,
        proposed,
        kinetic_coefficient=D,
    )

    # --------------------------------------------------------
    # Green-function ratio
    # --------------------------------------------------------

    forward = (
        proposed
        - walkers
        - drift_dt * drift
    )

    reverse = (
        walkers
        - proposed
        - drift_dt * drift_p
    )

    forward_sq = configuration_norm2(forward)
    reverse_sq = configuration_norm2(reverse)

    log_green_ratio = (
        -1.0
        / (4.0 * D * dt)
        * (
            reverse_sq
            - forward_sq
        )
    )

    # --------------------------------------------------------
    # Metropolis-Hastings ratio
    #
    # Target distribution of this move is Psi_T^2.
    # --------------------------------------------------------

    log_acceptance = (
        2.0
        * (
            log_psi_p
            - log_psi
        )
        + log_green_ratio
    )

    log_acceptance = np.minimum(
        0.0,
        log_acceptance,
    )

    accepted = (
        np.log(
            rng.random(n_walkers)
        )
        < log_acceptance
    )

    acceptance_rate = float(
        np.mean(accepted)
    )

    # --------------------------------------------------------
    # Apply accept/reject
    # --------------------------------------------------------

    new_walkers = np.where(
        walker_broadcast_mask(accepted, walkers.ndim),
        proposed,
        walkers,
    )

    new_log_psi = np.where(
        accepted,
        log_psi_p,
        log_psi,
    )

    new_drift = np.where(
        walker_broadcast_mask(accepted, walkers.ndim),
        drift_p,
        drift,
    )

    new_e_local = np.where(
        accepted,
        e_local_p,
        e_local,
    )

    return (
        new_walkers,
        new_log_psi,
        new_drift,
        new_e_local,
        acceptance_rate,
    )


def drift_diffusion_second_order(
    walkers,
    potential,
    trial,
    dt,
    rng,
    *,
    kinetic_coefficient=0.5,
):
    """
    Symmetric second-order split propagator:

        diffusion(dt/2)
        -> drift(dt), integrated with midpoint RK2
        -> diffusion(dt/2)

    There is no Metropolis accept/reject step in this
    propagator.

    Returns:
        walkers
        log_psi
        drift
        e_local
        acceptance_rate = NaN
    """

    # --------------------------------------------------------
    # 1. Half diffusion
    #
    # Full diffusion has coordinate variance 2*D*dt,
    # therefore half diffusion has coordinate variance D*dt.
    # --------------------------------------------------------

    D = require_positive_kinetic(kinetic_coefficient)
    drift_dt = 2.0 * D * dt

    y = (
        walkers
        + np.sqrt(D * dt)
        * rng.normal(size=walkers.shape)
    )

    # --------------------------------------------------------
    # 2. Drift flow using second-order midpoint RK2
    #
    # dR / dtau = 2*D*grad(log Psi_T)
    # --------------------------------------------------------

    drift_y = np.asarray(
        trial.grad_log_psi(y),
        dtype=np.float64,
    )

    if drift_y.shape != walkers.shape:
        raise ValueError(
            "trial.grad_log_psi() must return "
            f"shape {walkers.shape}, "
            f"got {drift_y.shape}"
        )

    midpoint = (
        y
        + 0.5 * drift_dt * drift_y
    )

    drift_mid = np.asarray(
        trial.grad_log_psi(midpoint),
        dtype=np.float64,
    )

    if drift_mid.shape != walkers.shape:
        raise ValueError(
            "trial.grad_log_psi() must return "
            f"shape {walkers.shape}, "
            f"got {drift_mid.shape}"
        )

    z = (
        y
        + drift_dt * drift_mid
    )

    # --------------------------------------------------------
    # 3. Second half diffusion
    # --------------------------------------------------------

    proposed = (
        z
        + np.sqrt(D * dt)
        * rng.normal(size=walkers.shape)
    )

    # --------------------------------------------------------
    # 4. Quantities at final endpoint
    # --------------------------------------------------------

    (
        log_psi,
        drift,
        e_local,
    ) = evaluate_trial_quantities(
        potential,
        trial,
        proposed,
        kinetic_coefficient=D,
    )

    # There is no acceptance probability in this propagator.
    acceptance_rate = float("nan")

    return (
        proposed,
        log_psi,
        drift,
        e_local,
        acceptance_rate,
    )