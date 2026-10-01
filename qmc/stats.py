from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Estimate:
    mean: float
    stderr: float
    n_blocks: int
    block_size: int


def block_statistics(values, block_size: int = 50) -> Estimate:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("block_statistics expects a 1D array")
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    n_blocks = len(values) // block_size
    if n_blocks < 2:
        raise ValueError(
            f"Not enough samples for block statistics: {len(values)} samples with "
            f"block_size={block_size} gives {n_blocks} complete blocks."
        )

    trimmed = values[: n_blocks * block_size]
    block_means = trimmed.reshape(n_blocks, block_size).mean(axis=1)
    return Estimate(
        mean=float(block_means.mean()),
        stderr=float(block_means.std(ddof=1) / np.sqrt(n_blocks)),
        n_blocks=n_blocks,
        block_size=block_size,
    )
