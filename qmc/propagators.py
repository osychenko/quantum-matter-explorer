import numpy as np

from .evaluation import evaluate_trial_quantities


def drift_diffusion_metropolis(
    walkers,
    log_psi,
    drift,
    e_local,
    potential,
    trial,
    dt,
    rng,
):
    """
    First-order drift-diffusion proposal with
    Metropolis-Hastings correction.

    Proposal:

        R' = R + dt * grad(log Psi_T) + sqrt(dt) * eta

    Returns the post-accept/reject state.
    """

    n_walkers = len(walkers)

    # --------------------------------------------------------
    # Drift-diffusion proposal
    # --------------------------------------------------------

    proposed = (
        walkers
        + dt * drift
        + np.sqrt(dt)
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
    )

    # --------------------------------------------------------
    # Green-function ratio
    # --------------------------------------------------------

    forward = (
        proposed
        - walkers
        - dt * drift
    )

    reverse = (
        walkers
        - proposed
        - dt * drift_p
    )

    forward_sq = np.sum(
        forward * forward,
        axis=1,
    )

    reverse_sq = np.sum(
        reverse * reverse,
        axis=1,
    )

    log_green_ratio = (
        -0.5
        / dt
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
        accepted[:, None],
        proposed,
        walkers,
    )

    new_log_psi = np.where(
        accepted,
        log_psi_p,
        log_psi,
    )

    new_drift = np.where(
        accepted[:, None],
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
    # Full diffusion has variance dt, therefore half
    # diffusion has variance dt/2.
    # --------------------------------------------------------

    y = (
        walkers
        + np.sqrt(0.5 * dt)
        * rng.normal(size=walkers.shape)
    )

    # --------------------------------------------------------
    # 2. Drift flow using second-order midpoint RK2
    #
    # dR / dtau = grad(log Psi_T)
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
        + 0.5 * dt * drift_y
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
        + dt * drift_mid
    )

    # --------------------------------------------------------
    # 3. Second half diffusion
    # --------------------------------------------------------

    proposed = (
        z
        + np.sqrt(0.5 * dt)
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