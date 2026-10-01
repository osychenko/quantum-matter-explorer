from __future__ import annotations

import numpy as np


def systematic_resample(probabilities, rng: np.random.Generator):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    n = len(probabilities)
    positions = (rng.random() + np.arange(n)) / n
    cumulative = np.cumsum(probabilities)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions, side="right")
