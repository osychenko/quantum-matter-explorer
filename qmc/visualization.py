"""Headless Matplotlib visualizations of periodic configurations and observables.

Cell snapshots show *a sampled configuration*, not classical particle tracks or
individual bosons with well-defined time-dependent trajectories.
"""

from __future__ import annotations

from itertools import permutations, product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .cells import BCCCell, PeriodicCell
from .observables import PairCorrelationResult, StructureFactorResult


def _save(fig, output):
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


def _primitive_wireframe(cell):
    bits = np.array(list(product((0, 1), repeat=3)), dtype=float)
    vertices = bits @ cell.vectors
    edges = [
        (i, j) for i in range(8) for j in range(i + 1, 8)
        if int(np.count_nonzero(bits[i] != bits[j])) == 1
    ]
    return vertices, edges


def _bcc_wireframe(cell):
    # Wigner--Seitz vertices are permutations of (0, +/-L/4, +/-L/2).
    # The truncated octahedron has 24 vertices and 36 equal edges.
    values = set()
    for perm in set(permutations((0.0, 0.25, 0.5))):
        for sign in product((-1, 1), repeat=3):
            values.add(tuple(round(a * b, 8) for a, b in zip(perm, sign)))
    vertices = cell.length * np.asarray(sorted(values), dtype=float)
    edge_length2 = cell.length**2 / 8.0
    edges = [
        (i, j) for i in range(len(vertices)) for j in range(i + 1, len(vertices))
        if np.isclose(np.sum((vertices[i] - vertices[j]) ** 2), edge_length2,
                      rtol=1e-9, atol=1e-12)
    ]
    return vertices, edges


def plot_cell_snapshot(positions, cell: PeriodicCell, output, *, walker_index=0):
    """Plot one sampled N-particle configuration inside its periodic cell.

    For BCC, show the actual truncated-octahedral Wigner--Seitz polyhedron.
    Other cells display their primitive parallelepiped. Input may be (N,3)
    or (W,N,3), with walker_index selecting a configuration from the latter.
    """
    if not isinstance(cell, PeriodicCell):
        raise TypeError("cell must be a PeriodicCell")
    xyz = np.asarray(positions, dtype=float)
    if xyz.ndim == 3:
        if not 0 <= walker_index < len(xyz):
            raise IndexError("walker_index out of range")
        xyz = xyz[walker_index]
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) == 0:
        raise ValueError("positions must have shape (N,3) or (W,N,3)")
    if not np.all(np.isfinite(xyz)):
        raise ValueError("positions must be finite")
    if isinstance(cell, BCCCell):
        vertices, edges = _bcc_wireframe(cell)
        points = cell.wrap(xyz, centered=True)
        shape_name = "BCC Wigner–Seitz (truncated octahedron)"
    else:
        vertices, edges = _primitive_wireframe(cell)
        points = cell.wrap(xyz)
        shape_name = "Periodic primitive cell"

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    for a, b in edges:
        ax.plot(*np.stack([vertices[a], vertices[b]]).T, linewidth=1.35,
                alpha=0.8, color="0.5")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=45,
               alpha=0.9, depthshade=True)
    lo = vertices.min(axis=0)
    hi = vertices.max(axis=0)
    span = max(float(np.max(hi - lo)), 1e-12)
    center = (lo + hi) / 2.0
    for setter, mid in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
        setter(mid - 0.55 * span, mid + 0.55 * span)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"{shape_name} — {len(points)} bosons")
    fig.text(0.5, 0.02, "One Monte Carlo configuration, not physical trajectories",
             ha="center", fontsize=9)
    return _save(fig, output)


def plot_pair_correlation(data: PairCorrelationResult, output):
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.plot(data.radius, data.g_r, marker="o", markersize=3)
    ax.axhline(1.0, linestyle="--", alpha=0.6, label="Uniform ideal gas")
    ax.set(xlabel="Pair separation r", ylabel="g(r)",
           title=f"Pair correlation — N={data.n_particles}")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.2)
    ax.legend()
    return _save(fig, output)


def plot_structure_factor(data: StructureFactorResult, output):
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.scatter(data.k_magnitude, data.s_k, alpha=0.8)
    ax.axhline(1.0, linestyle="--", alpha=0.6, label="Uncorrelated reference")
    ax.set(xlabel="Wavevector magnitude |k|", ylabel="S(k)",
           title=f"Static structure factor — N={data.n_particles}")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.2)
    ax.legend()
    # Points at equal |k| may be different directions. They are not radially
    # averaged: Bragg peaks/directional symmetry can be significant.
    return _save(fig, output)
