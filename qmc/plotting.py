from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# Common helpers
# ============================================================


def _prepare_output(output: Path) -> None:
    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


def _save(fig, output: Path) -> None:
    _prepare_output(output)

    fig.tight_layout()

    fig.savefig(
        output,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


# ============================================================
# VMC 1D scan
# ============================================================


def plot_vmc_scan(
    points,
    best_parameter,
    output: Path,
):
    x = np.asarray(
        [
            p.parameter
            for p in points
        ],
        dtype=float,
    )

    y = np.asarray(
        [
            p.estimate.mean
            for p in points
        ],
        dtype=float,
    )

    e = np.asarray(
        [
            p.estimate.stderr
            for p in points
        ],
        dtype=float,
    )

    order = np.argsort(x)

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.errorbar(
        x[order],
        y[order],
        yerr=e[order],
        marker="o",
        linestyle="none",
        capsize=3,
        label="VMC",
    )

    ax.axvline(
        best_parameter,
        linestyle="--",
        label=(
            f"Best alpha = "
            f"{best_parameter:.4f}"
        ),
    )

    ax.set_xlabel("alpha")
    ax.set_ylabel("VMC energy [Ha]")

    ax.set_title(
        "Yukawa trial-wavefunction optimization"
    )

    ax.legend()

    _save(
        fig,
        output,
    )


# ============================================================
# VMC 2D alpha/beta scan
# ============================================================


def plot_vmc_2d_scan(
    points,
    best,
    output: Path,
):
    alpha = np.asarray(
        [
            p.alpha
            for p in points
        ],
        dtype=float,
    )

    beta = np.asarray(
        [
            p.beta
            for p in points
        ],
        dtype=float,
    )

    energy = np.asarray(
        [
            p.estimate.mean
            for p in points
        ],
        dtype=float,
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    scatter = ax.scatter(
        alpha,
        beta,
        c=energy,
        s=60,
    )

    ax.scatter(
        [best.alpha],
        [best.beta],
        marker="x",
        s=160,
        linewidths=2,
        label=(
            "VMC minimum "
            f"({best.alpha:.3f}, "
            f"{best.beta:.3f})"
        ),
    )

    ax.set_xlabel("alpha")
    ax.set_ylabel("beta")

    ax.set_title(
        "Yukawa VMC alpha/beta optimization"
    )

    colorbar = fig.colorbar(
        scatter,
        ax=ax,
    )

    colorbar.set_label(
        "VMC energy [Ha]"
    )

    ax.legend()

    _save(
        fig,
        output,
    )


# ============================================================
# DMC production energy
# ============================================================


def plot_dmc_energy(
    result,
    output: Path,
    reference_energy=None,
):
    mixed = np.asarray(
        result.local_energy_history,
        dtype=float,
    )

    growth = np.asarray(
        result.reference_energy_history,
        dtype=float,
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        mixed,
        label="Mixed <E_L>",
    )

    ax.plot(
        growth,
        alpha=0.5,
        label="Growth / E_ref",
    )

    if reference_energy is not None:
        ax.axhline(
            reference_energy,
            linestyle="--",
            label=(
                f"Reference "
                f"{reference_energy:.8f} Ha"
            ),
        )

    ax.set_xlabel(
        "Production DMC step"
    )

    ax.set_ylabel(
        "Energy [Ha]"
    )

    ax.set_title(
        "DMC production energy"
    )

    ax.legend()

    _save(
        fig,
        output,
    )


# ============================================================
# Radial walker distribution
# ============================================================


def plot_radial_distribution(
    walkers,
    output: Path,
    title="Radial walker distribution",
):
    walkers = np.asarray(
        walkers,
        dtype=float,
    )

    radius = np.linalg.norm(
        walkers,
        axis=1,
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.hist(
        radius,
        bins=120,
        density=True,
    )

    ax.set_xlabel("r [a0]")

    # More accurate than simply "walker density":
    # this is the probability density of the radial
    # coordinate, not a 3D density.
    ax.set_ylabel(
        "Radial probability density"
    )

    ax.set_title(
        title
    )

    _save(
        fig,
        output,
    )


# ============================================================
# DMC diagnostics
# ============================================================


def plot_dmc_diagnostics(
    result,
    adaptation_steps,
    equilibration_steps,
    target_acceptance,
    output_dir: Path,
):
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # dt history
    # --------------------------------------------------------

    dt_history = np.asarray(
        result.dt_history,
        dtype=float,
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        dt_history
    )

    # Only draw adaptation boundary when an actual
    # adaptation region exists.
    if adaptation_steps > 0:
        ax.axvline(
            adaptation_steps,
            linestyle=":",
            label="End adaptation",
        )

    if equilibration_steps > 0:
        ax.axvline(
            equilibration_steps,
            linestyle="--",
            label="End equilibration",
        )

    ax.set_xlabel("DMC step")
    ax.set_ylabel("dt")

    ax.set_title(
        "DMC time step"
    )

    if (
        adaptation_steps > 0
        or equilibration_steps > 0
    ):
        ax.legend()

    _save(
        fig,
        output_dir / "dmc_dt.png",
    )

    # --------------------------------------------------------
    # Acceptance history
    #
    # Second-order propagation has no Metropolis
    # accept/reject step, so its history contains NaN.
    # --------------------------------------------------------

    acceptance = np.asarray(
        result.acceptance_history,
        dtype=float,
    )

    finite = np.isfinite(
        acceptance
    )

    if not np.any(finite):
        # No acceptance concept for this propagator.
        # Do not create a meaningless empty plot.
        return

    steps = np.arange(
        len(acceptance)
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.plot(
        steps[finite],
        acceptance[finite],
        alpha=0.3,
        label="Per-step",
    )

    # --------------------------------------------------------
    # Smoothed acceptance
    # --------------------------------------------------------

    finite_values = acceptance[
        finite
    ]

    if len(finite_values) >= 2:

        window = min(
            50,
            len(finite_values),
        )

        kernel = (
            np.ones(window)
            / window
        )

        smoothed = np.convolve(
            finite_values,
            kernel,
            mode="valid",
        )

        finite_steps = steps[
            finite
        ]

        smooth_steps = finite_steps[
            window - 1:
        ]

        ax.plot(
            smooth_steps,
            smoothed,
            label=(
                f"{window}-step average"
            ),
        )

    if (
        target_acceptance is not None
        and np.isfinite(
            target_acceptance
        )
    ):
        ax.axhline(
            target_acceptance,
            linestyle="--",
            label="Target",
        )

    if adaptation_steps > 0:
        ax.axvline(
            adaptation_steps,
            linestyle=":",
            label="End adaptation",
        )

    if equilibration_steps > 0:
        ax.axvline(
            equilibration_steps,
            linestyle="--",
            label="End equilibration",
        )

    ax.set_xlabel("DMC step")
    ax.set_ylabel("Acceptance")

    ax.set_ylim(
        0.0,
        1.01,
    )

    ax.set_title(
        "Metropolis acceptance"
    )

    ax.legend()

    _save(
        fig,
        output_dir
        / "dmc_acceptance.png",
    )


# ============================================================
# DMC timestep convergence
# ============================================================


def plot_dmc_dt_scan(
    points,
    output: Path,
    reference_energy=None,
):
    """
    Compare fixed-dt convergence of multiple DMC schemes.

    The primary plotted quantity is the mixed estimator
    <E_L>, since this is normally the energy estimator of
    interest in importance-sampled DMC.

    Growth/reference energies are shown with lighter,
    dashed curves for comparison.
    """

    if not points:
        raise ValueError(
            "No DMC timestep scan points supplied."
        )

    schemes = sorted(
        {
            p.scheme
            for p in points
        }
    )

    fig, ax = plt.subplots(
        figsize=(9, 6)
    )

    markers = [
        "o",
        "s",
        "^",
        "D",
        "v",
    ]

    for scheme_index, scheme in enumerate(
        schemes
    ):
        scheme_points = sorted(
            [
                p
                for p in points
                if p.scheme == scheme
            ],
            key=lambda p: p.dt,
        )

        dt = np.asarray(
            [
                p.dt
                for p in scheme_points
            ],
            dtype=float,
        )

        mixed = np.asarray(
            [
                p.mixed.mean
                for p in scheme_points
            ],
            dtype=float,
        )

        mixed_err = np.asarray(
            [
                p.mixed.stderr
                for p in scheme_points
            ],
            dtype=float,
        )

        growth = np.asarray(
            [
                p.growth.mean
                for p in scheme_points
            ],
            dtype=float,
        )

        growth_err = np.asarray(
            [
                p.growth.stderr
                for p in scheme_points
            ],
            dtype=float,
        )

        marker = markers[
            scheme_index
            % len(markers)
        ]

        # ----------------------------------------------------
        # Mixed estimator
        # ----------------------------------------------------

        mixed_artist = ax.errorbar(
            dt,
            mixed,
            yerr=mixed_err,
            marker=marker,
            linestyle="-",
            capsize=3,
            label=(
                f"{scheme}: mixed"
            ),
        )

        # Reuse the automatically selected Matplotlib color
        # for the corresponding growth estimator.
        color = (
            mixed_artist.lines[0]
            .get_color()
        )

        # ----------------------------------------------------
        # Growth/reference estimator
        # ----------------------------------------------------

        ax.errorbar(
            dt,
            growth,
            yerr=growth_err,
            marker=marker,
            linestyle="--",
            alpha=0.55,
            capsize=3,
            color=color,
            label=(
                f"{scheme}: growth"
            ),
        )

    if reference_energy is not None:
        ax.axhline(
            reference_energy,
            linestyle=":",
            label=(
                "Reference "
                f"{reference_energy:.10f} Ha"
            ),
        )

    ax.set_xlabel(
        "DMC time step dt"
    )

    ax.set_ylabel(
        "Energy [Ha]"
    )

    ax.set_title(
        "DMC time-step convergence"
    )

    ax.legend()

    _save(
        fig,
        output,
    )


# ============================================================
# DMC timestep error relative to benchmark
# ============================================================


def plot_dmc_dt_error(
    points,
    output: Path,
    reference_energy,
):
    """
    Plot absolute mixed-estimator error:

        |E_DMC(dt) - E_reference|

    This is particularly useful when comparing the
    time-step scaling of different propagators.
    """

    if reference_energy is None:
        raise ValueError(
            "reference_energy is required "
            "for the timestep-error plot."
        )

    schemes = sorted(
        {
            p.scheme
            for p in points
        }
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    markers = [
        "o",
        "s",
        "^",
        "D",
        "v",
    ]

    for scheme_index, scheme in enumerate(
        schemes
    ):
        scheme_points = sorted(
            [
                p
                for p in points
                if p.scheme == scheme
            ],
            key=lambda p: p.dt,
        )

        dt = np.asarray(
            [
                p.dt
                for p in scheme_points
            ],
            dtype=float,
        )

        energy = np.asarray(
            [
                p.mixed.mean
                for p in scheme_points
            ],
            dtype=float,
        )

        error = np.abs(
            energy
            - reference_energy
        )

        marker = markers[
            scheme_index
            % len(markers)
        ]

        ax.plot(
            dt,
            error,
            marker=marker,
            label=scheme,
        )

    ax.set_xlabel(
        "DMC time step dt"
    )

    ax.set_ylabel(
        "|E(dt) - E_reference| [Ha]"
    )

    ax.set_title(
        "DMC time-step bias"
    )

    ax.set_yscale("log")

    ax.legend()

    _save(
        fig,
        output,
    )


# ============================================================
# DMC acceptance vs dt
# ============================================================


def plot_dmc_dt_acceptance(
    points,
    output: Path,
):
    """
    Acceptance rate versus timestep.

    Only schemes with an actual Metropolis acceptance
    probability are included. Second-order split propagation
    therefore disappears automatically.
    """

    schemes = sorted(
        {
            p.scheme
            for p in points
        }
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    plotted = False

    for scheme in schemes:

        scheme_points = sorted(
            [
                p
                for p in points
                if (
                    p.scheme == scheme
                    and np.isfinite(
                        p.production_acceptance
                    )
                )
            ],
            key=lambda p: p.dt,
        )

        if not scheme_points:
            continue

        plotted = True

        dt = np.asarray(
            [
                p.dt
                for p in scheme_points
            ],
            dtype=float,
        )

        acceptance = np.asarray(
            [
                p.production_acceptance
                for p in scheme_points
            ],
            dtype=float,
        )

        ax.plot(
            dt,
            acceptance,
            marker="o",
            label=scheme,
        )

    if not plotted:
        plt.close(fig)
        return

    ax.set_xlabel(
        "DMC time step dt"
    )

    ax.set_ylabel(
        "Production acceptance"
    )

    ax.set_ylim(
        0.0,
        1.01,
    )

    ax.set_title(
        "DMC acceptance vs time step"
    )

    ax.legend()

    _save(
        fig,
        output,
    )