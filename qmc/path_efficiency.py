"""Pilot/held-out CPU-efficiency comparisons for Patch 16 open-path moves.

ESS per CPU second is measured for the *midpoint potential observable*.
Elapsed process CPU time is platform-dependent and not reproducible bitwise.
A pilot search and independent validation prevent quoting the same chains
for tuning and confirming the tuned mixture. This does not prove superiority.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import process_time

import numpy as np

from .interacting_path import _integer, _positive
from .path_staging import StagedInteractingPath, StagedPathConfig, autocorrelation_estimate


@dataclass(frozen=True)
class EfficiencyConfig:
    bridge_probabilities: tuple[float, ...] = (0.0, 0.4, 0.8)
    pilot_replicas: int = 2
    validation_replicas: int = 2
    pilot_steps: int = 500
    validation_steps: int = 1200
    burn_in: int = 100
    thin: int = 4
    n_links: int = 8
    time_step: float = 0.2
    move_sigma: float = 0.4
    refresh_probability: float = 0.12
    translate_probability: float = 0.12
    bridge_lengths: tuple[int, ...] = (2, 4)
    seed: int = 2026

    def validate(self):
        if not isinstance(self.bridge_probabilities, (tuple, list)) or not self.bridge_probabilities:
            raise ValueError("bridge_probabilities must be nonempty")
        ps = np.asarray(self.bridge_probabilities, dtype=float)
        if (np.any(~np.isfinite(ps)) or np.any(ps < 0) or np.any(ps > 1)
                or len(set(ps.tolist())) != len(ps) or not np.any(ps == 0)):
            raise ValueError("bridge probabilities must be unique [0,1] and include baseline 0")
        for name, minimum in (("pilot_replicas", 2), ("validation_replicas", 2),
                              ("pilot_steps", 2), ("validation_steps", 2),
                              ("burn_in", 0), ("thin", 1), ("n_links", 4), ("seed", 0)):
            _integer(getattr(self, name), name, minimum)
        if self.n_links % 2:
            raise ValueError("n_links must be even")
        if self.burn_in >= min(self.pilot_steps, self.validation_steps):
            raise ValueError("burn_in must be less than every run length")
        if (self.pilot_steps-self.burn_in+self.thin-1)//self.thin < 4:
            raise ValueError("pilot must have at least four measurements")
        _positive(self.time_step, "time_step")
        _positive(self.move_sigma, "move_sigma")
        for name in ("refresh_probability", "translate_probability"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0 or value > 1:
                raise ValueError(f"{name} must be in [0,1]")
        if self.refresh_probability+self.translate_probability > 1:
            raise ValueError("refresh/translate probabilities sum exceeds 1")
        if not self.bridge_lengths or any(_integer(v,"bridge length",2) >= self.n_links for v in self.bridge_lengths):
            raise ValueError("invalid bridge lengths")
        return self


def benchmark_chain(cell, potential, trial, *, n_particles, cfg: EfficiencyConfig,
                    bridge_probability, seed, n_steps, phase, replica) -> dict:
    """Execute a single deterministic-seed chain and measure per-process CPU time."""
    model_cfg = StagedPathConfig(
        time_step=cfg.time_step, n_links=cfg.n_links, n_steps=n_steps,
        burn_in=cfg.burn_in, thin=cfg.thin, move_sigma=cfg.move_sigma,
        refresh_probability=cfg.refresh_probability,
        translate_probability=cfg.translate_probability,
        bridge_probability=float(bridge_probability),
        bridge_lengths=cfg.bridge_lengths, seed=int(seed))
    sampler = StagedInteractingPath(cell, potential, trial, n_particles, model_cfg)
    series = []
    start = process_time()
    for step in range(model_cfg.n_steps):
        sampler.step()
        if step >= model_cfg.burn_in and (step-model_cfg.burn_in) % model_cfg.thin == 0:
            # Only midpoint V is collected: the expensive thermodynamic energy
            # observable cannot mask the actual cost of each move mixture.
            values = np.asarray(potential(sampler.path[model_cfg.n_links//2: model_cfg.n_links//2+1]), dtype=float)
            if values.shape != (1,) or not np.isfinite(values[0]):
                raise ValueError("potential must yield finite midpoint value")
            series.append(float(values[0]))
    cpu = process_time()-start
    if cpu <= 0:
        raise RuntimeError("CPU clock resolution too coarse")
    diag = autocorrelation_estimate(series)
    return dict(phase=phase, bridge_probability=float(bridge_probability),
                replica=int(replica), seed=int(seed), n_steps=int(n_steps),
                n_samples=diag.n_samples, midpoint_mean=diag.mean,
                chain_stderr=diag.stderr, tau_int=diag.tau_integrated,
                effective_samples=diag.effective_samples, cpu_seconds=cpu,
                effective_samples_per_cpu_second=diag.effective_samples/cpu,
                accept_staging=sampler._counts['staging'][0]/sampler._counts['staging'][1]
                if sampler._counts['staging'][1] else float('nan'),
                accept_bisection=sampler._counts['bisection'][0]/sampler._counts['bisection'][1]
                if sampler._counts['bisection'][1] else float('nan'))


def compare_bridge_probabilities(cell, potential, trial, *, n_particles: int,
                                 cfg: EfficiencyConfig = EfficiencyConfig()):
    """Pilot-select on mean ESS/CPU, validate winner and baseline on new chains.

    Returns (pilot_rows, validation_rows, selected_probability).  Pilot and
    validation seeds cannot overlap. For fair comparison every mixture uses
    the same number of attempts, but *not* the same CPU time; ESS/CPU corrects
    for actual measured process time. A selected optimum is dataset-specific.
    """
    cfg.validate()
    pilot_rows, valid_rows = [], []
    for index, p in enumerate(cfg.bridge_probabilities):
        for replica in range(cfg.pilot_replicas):
            seed = cfg.seed + 10_000*index + 31*replica
            pilot_rows.append(benchmark_chain(
                cell,potential,trial,n_particles=n_particles,cfg=cfg,
                bridge_probability=p,seed=seed,n_steps=cfg.pilot_steps,
                phase="pilot",replica=replica))
    scores = {p: float(np.mean([r['effective_samples_per_cpu_second']
                                for r in pilot_rows if r['bridge_probability']==p]))
              for p in cfg.bridge_probabilities}
    selected = float(max(cfg.bridge_probabilities, key=lambda p:scores[p]))
    for index, p in enumerate(sorted({0.0, selected})):
        for replica in range(cfg.validation_replicas):
            seed = cfg.seed + 5_000_000 + 10_000*index + 31*replica
            valid_rows.append(benchmark_chain(
                cell,potential,trial,n_particles=n_particles,cfg=cfg,
                bridge_probability=p,seed=seed,n_steps=cfg.validation_steps,
                phase="validation",replica=replica))
    return pilot_rows, valid_rows, selected
