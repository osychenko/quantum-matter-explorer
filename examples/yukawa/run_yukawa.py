from __future__ import annotations

from dataclasses import replace
from functools import partial
from pathlib import Path

import numpy as np

from qmc.dmc import DMCConfig, run_dmc_importance, scan_dmc_dt
from qmc.plotting import (
    plot_dmc_diagnostics,
    plot_dmc_dt_acceptance,
    plot_dmc_dt_error,
    plot_dmc_dt_scan,
    plot_dmc_energy,
    plot_radial_distribution,
    plot_vmc_2d_scan,
)

from qmc.potentials import yukawa_potential
from qmc.stats import block_statistics
from qmc.trials import YukawaCuspTrial
from qmc.vmc import VMCConfig, run_vmc, scan_alpha_beta, save_vmc_2d_scan, save_dmc_dt_scan

G = 1.0
MU = 0.5
REFERENCE_ENERGY = -0.1481170218
OUT = Path("outputs/yukawa")
PRODUCTION_DT = 0.01

def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Physical system
    # ========================================================

    potential = partial(
        yukawa_potential,
        g=G,
        mu=MU,
    )

    # Cusp-preserving trial.
    #
    # g is fixed by the potential.
    # VMC optimizes alpha and beta.
    trial_factory = lambda alpha, beta: YukawaCuspTrial(
        g=G,
        mu=MU,
        alpha=alpha,
        beta=beta,
    )

    # ========================================================
    # VMC: coarse 2D parameter scan
    # ========================================================

    coarse_config = VMCConfig(
        n_walkers=1_000,
        n_steps=1_000,
        burn_in=250,
        seed=12345,
    )

    coarse_alphas = np.linspace(
        0.60,
        1.00,
        9,
    )

    coarse_betas = np.linspace(
        0.25,
        2.00,
        8,
    )

    print()
    print("==========================")
    print("VMC coarse alpha/beta scan")
    print("==========================")

    coarse_points = scan_alpha_beta(
        coarse_alphas,
        coarse_betas,
        trial_factory=trial_factory,
        potential=potential,
        config=coarse_config,
        block_size=20,
    )

    best_coarse = min(
        coarse_points,
        key=lambda p: p.estimate.mean,
    )

    print()
    print(
        "Best coarse: "
        f"alpha={best_coarse.alpha:.6f}, "
        f"beta={best_coarse.beta:.6f}, "
        f"E={best_coarse.estimate.mean:.6f}"
    )

    # ========================================================
    # VMC: fine scan around coarse minimum
    # ========================================================

    fine_config = replace(
        coarse_config,
        n_walkers=2_000,
        n_steps=1_500,
        burn_in=300,
        seed=54321,
    )

    # Refine alpha locally
    fine_alphas = np.linspace(
        max(
            0.05,
            best_coarse.alpha - 0.05,
        ),
        best_coarse.alpha + 0.05,
        11,
    )

    # Refine beta locally.
    #
    # beta must remain positive.
    fine_betas = np.linspace(
        max(
            0.05,
            best_coarse.beta - 0.25,
        ),
        best_coarse.beta + 0.25,
        11,
    )

    print()
    print("==========================")
    print("VMC fine alpha/beta scan")
    print("==========================")

    fine_points = scan_alpha_beta(
        fine_alphas,
        fine_betas,
        trial_factory=trial_factory,
        potential=potential,
        config=fine_config,
        block_size=20,
    )

    best = min(
        fine_points,
        key=lambda p: p.estimate.mean,
    )

    best_alpha = best.alpha
    best_beta = best.beta

    optimized_trial = YukawaCuspTrial(
        g=G,
        mu=MU,
        alpha=best_alpha,
        beta=best_beta,
    )

    print()
    print("==========================")
    print("VMC optimized trial")
    print("==========================")

    print(
        f"alpha       = {best_alpha:.6f}"
    )

    print(
        f"beta        = {best_beta:.6f}"
    )

    print(
        f"E_VMC       = "
        f"{best.estimate.mean:.6f} "
        f"+/- {best.estimate.stderr:.6f} Ha"
    )

    print(
        f"acceptance  = "
        f"{best.production_acceptance:.4f}"
    )

    # ========================================================
    # Independent VMC validation of optimized trial
    # ========================================================

    validation_vmc_config = replace(
        fine_config,
        n_walkers=4_000,
        n_steps=3_000,
        burn_in=500,
        seed=24680,
    )

    vmc_result = run_vmc(
        potential,
        optimized_trial,
        validation_vmc_config,
    )

    vmc_estimate = block_statistics(
        vmc_result.energy_history,
        block_size=50,
    )

    print()
    print("==========================")
    print("VMC validation")
    print("==========================")

    print(
        f"alpha                 = "
        f"{best_alpha:.6f}"
    )

    print(
        f"beta                  = "
        f"{best_beta:.6f}"
    )

    print(
        f"E_VMC                 = "
        f"{vmc_estimate.mean:.8f} "
        f"+/- {vmc_estimate.stderr:.8f} Ha"
    )

    print(
        f"Var(E_L)              = "
        f"{vmc_result.local_energy_variance:.8e} Ha^2"
    )

    print(
        f"Std(E_L)              = "
        f"{vmc_result.local_energy_std:.8e} Ha"
    )

    print(
        f"local-energy samples  = "
        f"{vmc_result.n_local_energy_samples}"
    )


    # ========================================================
    # Fixed-dt DMC convergence scan
    # ========================================================

    dt_scan_config = DMCConfig(
        # A little cheaper than the final production run.
        n_walkers=15_000,
        n_steps=4_000,

        # No adaptation; 1000 steps are discarded.
        adaptation_steps=0,
        equilibration_steps=1_000,

        adaptive_dt=False,

        initial_sigma=2.0,
        seed=13579,

        # Don't spam the output for every scan.
        log_interval=1_000,
    )

    dt_values = [
        0.005,
        0.010,
        0.020,
        0.030,
        0.040,
        0.060,
    ]

    dt_points = scan_dmc_dt(
        dt_values,

        potential=potential,
        trial=optimized_trial,

        config=dt_scan_config,

        schemes=(
            "metropolis",
            "second_order",
        ),

        equilibration_tau=15.0,
        production_tau=30.0,
        block_tau=2.0,
    )    

    print()
    print("==========================")
    print("DMC dt convergence")
    print("==========================")

    print(
        f"{'scheme':14s} "
        f"{'dt':>8s} "
        f"{'mixed E':>25s} "
        f"{'growth E':>25s} "
        f"{'accept':>10s}"
    )

    for p in dt_points:

        if np.isfinite(
            p.production_acceptance
        ):
            acceptance_text = (
                f"{p.production_acceptance:.4f}"
            )
        else:
            acceptance_text = "n/a"

        print(
            f"{p.scheme:14s} "
            f"{p.dt:8.5f} "
            f"{p.mixed.mean: .8f} "
            f"+/- {p.mixed.stderr:.8f} "
            f"{p.growth.mean: .8f} "
            f"+/- {p.growth.stderr:.8f} "
            f"{acceptance_text:>10s}"
        )
        
    save_dmc_dt_scan(
        dt_points,
        OUT / "dmc_dt_scan.csv",
    )

    plot_dmc_dt_scan(
        dt_points,
        OUT / "dmc_dt_scan.png",
        reference_energy=REFERENCE_ENERGY,
    )

    plot_dmc_dt_error(
        dt_points,
        OUT / "dmc_dt_error.png",
        reference_energy=REFERENCE_ENERGY,
    )

    plot_dmc_dt_acceptance(
        dt_points,
        OUT / "dmc_dt_acceptance.png",
    )

    # ========================================================
    # DMC using the optimized cusp-preserving VMC trial
    # ========================================================

    trial = YukawaCuspTrial(
        g=G,
        mu=MU,
        alpha=best_alpha,
        beta=best_beta,
    )

    config = DMCConfig(
        n_walkers=30_000,
        n_steps=6_000,

        adaptation_steps=0,
        equilibration_steps=1_500,

        dt=PRODUCTION_DT,
        adaptive_dt=False,

        scheme="second_order",

        initial_sigma=2.0,
        seed=12345,
    )

    result = run_dmc_importance(
        potential,
        optimized_trial,
        config,
    )

    mixed = block_statistics(
        result.local_energy_history,
        block_size=100,
    )

    growth = block_statistics(
        result.reference_energy_history,
        block_size=100,
    )

    # ========================================================
    # Results
    # ========================================================

    print()
    print("==========================")
    print("Yukawa DMC")
    print("==========================")

    print(
        f"g                     = {G}"
    )

    print(
        f"mu                    = {MU}"
    )

    print(
        f"trial alpha (VMC)     = "
        f"{best_alpha:.6f}"
    )

    print(
        f"trial beta (VMC)      = "
        f"{best_beta:.6f}"
    )

    print(
        f"VMC energy            = "
        f"{best.estimate.mean:.6f} "
        f"+/- {best.estimate.stderr:.6f} Ha"
    )

    print(
        f"mixed <E_L>           = "
        f"{mixed.mean:.6f} "
        f"+/- {mixed.stderr:.6f} Ha"
    )

    print(
        f"growth/reference E    = "
        f"{growth.mean:.6f} "
        f"+/- {growth.stderr:.6f} Ha"
    )

    print(
        f"reference target      ~ "
        f"{REFERENCE_ENERGY:.4f} Ha"
    )

    print(
        f"final dt              = "
        f"{result.final_dt:.6f}"
    )

    print(
        f"production acceptance = "
        f"{result.production_acceptance:.4f}"
    )

    # ========================================================
    # DMC plots
    # ========================================================

    plot_dmc_energy(
        result,
        OUT / "dmc_energy.png",
        reference_energy=REFERENCE_ENERGY,
    )

    plot_radial_distribution(
        result.walkers,
        OUT / "dmc_radial_distribution.png",
        title=(
            "Yukawa DMC radial distribution "
            f"(mu={MU}, "
            f"alpha={best_alpha:.3f}, "
            f"beta={best_beta:.3f})"
        ),
    )

    plot_dmc_diagnostics(
        result,
        config.adaptation_steps,
        config.equilibration_steps,
        config.target_acceptance,
        OUT,
    )

    # ========================================================
    # Save VMC scan results
    # ========================================================

    all_points = (
        coarse_points
        + fine_points
    )

    save_vmc_2d_scan(
        all_points,
        OUT / "vmc_alpha_beta_scan.csv",
    )

    plot_vmc_2d_scan(
        all_points,
        best,
        OUT / "vmc_alpha_beta_scan.png",
    )


if __name__ == "__main__":
    main()
