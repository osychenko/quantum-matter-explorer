"""Patch 38: bounded live VMC laboratories reusing the research engine.

These are variational / trial-state demonstrations, NOT certified DMC/PIMC,
finite-temperature phase diagrams, or independently sampled thermal states.
"""
from __future__ import annotations

from functools import partial
from math import sqrt

import numpy as np


def _summary(values):
    x = np.asarray(values, dtype=float)
    return float(np.mean(x)), float(np.std(x, ddof=1) / np.sqrt(len(x)))


def harmonic_vmc(alpha: float, omega: float) -> dict:
    """Gaussian ansatz in 3D; compare sampled VMC to known exact integrals.

    The research VMC sampler is used without alteration. The exact ground-state
    energy is 3*omega/2 ONLY at D=1/2 with V=omega^2*r^2/2.
    """
    from .potentials import harmonic_oscillator
    from .trials import GaussianTrial
    from .vmc import VMCConfig, run_vmc

    seeds = [380010, 380011, 380012]
    means, accepts = [], []
    for seed in seeds:
        cfg = VMCConfig(n_walkers=128, n_steps=260, burn_in=60,
                        sample_every=2, proposal_sigma=.75,
                        adaptive_proposal=False, seed=seed,
                        configuration_shape=(3,), kinetic_coefficient=.5)
        run = run_vmc(partial(harmonic_oscillator, omega=omega), GaussianTrial(alpha), cfg)
        means.append(float(np.mean(run.energy_history)))
        accepts.append(float(run.production_acceptance))
    mean, sem = _summary(means)
    # Gaussian Psi=exp(-alpha*r^2/2) has normalized |Psi|^2 variance 1/(2 alpha)
    # in each direction. <H> = 3*alpha/4 + 3*omega^2/(4*alpha).
    trial_expectation = .75 * (alpha + omega * omega / alpha)
    exact_ground = 1.5 * omega
    return dict(mean=mean, sem=sem, trial_expectation=trial_expectation,
                exact_ground=exact_ground, seeds=seeds, acceptance=float(np.mean(accepts)),
                n_walkers=128, n_steps=260, burn_in=60, sample_every=2,
                n_replicas=3)


def softcore_structure_vmc(amplitude: float) -> dict:
    """VMC trial-state g(r), directional S(k): the Hamiltonian is held fixed.

    The external knob is the Jastrow TRIAL amplitude, not the potential height.
    Results are finite-N trial-state estimates and never phase evidence.
    """
    from .bosons import SoftCorePairPotential, SoftCoreJastrowTrial
    from .cells import CubicCell
    from .observables import PairCorrelation, StaticStructureFactor
    from .vmc import VMCConfig, run_vmc

    n, L, cutoff, height = 6, 5., 1.2, 3.
    cell = CubicCell(L)
    potential = SoftCorePairPotential(cell, height=height, cutoff=cutoff)
    trial = SoftCoreJastrowTrial(cell, amplitude=amplitude, cutoff=cutoff)
    modes = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]], dtype=int)
    gr_chains, sk_chains, energies, acceptance = [], [], [], []
    seeds = [380110, 380111, 380112]
    for seed in seeds:
        cfg = VMCConfig(n_walkers=48, n_steps=150, burn_in=40,
                        sample_every=5, proposal_sigma=.55,
                        adaptive_proposal=False, proposal_mode='particle',
                        seed=seed, configuration_shape=(n, 3), kinetic_coefficient=.5)
        positions = np.random.default_rng(seed + 10000).uniform(0., L, (48, n, 3))
        gr = PairCorrelation(cell, n, n_bins=12, r_max=1.5)
        sk = StaticStructureFactor(cell, n, modes)
        run = run_vmc(potential, trial, cfg, initial_walkers=positions,
                      observables=(gr, sk))
        gr_chains.append(gr.result().g_r)
        sk_chains.append(sk.result().s_k)
        energies.append(float(np.mean(run.energy_history) / n))
        acceptance.append(float(run.production_acceptance))
    gr_arr = np.asarray(gr_chains)
    sk_arr = np.asarray(sk_chains)
    e_mean, e_sem = _summary(energies)
    r = gr.result().radius
    return dict(energy_per_particle=e_mean, energy_sem=e_sem,
                radius=r.tolist(), gr=gr_arr.mean(axis=0).tolist(),
                gr_sem=(gr_arr.std(axis=0, ddof=1) / np.sqrt(len(seeds))).tolist(),
                k_indices=modes.tolist(), k_magnitudes=np.linalg.norm(cell.wavevectors(modes), axis=1).tolist(),
                sk=sk_arr.mean(axis=0).tolist(),
                sk_sem=(sk_arr.std(axis=0, ddof=1) / np.sqrt(len(seeds))).tolist(),
                seeds=seeds, acceptance=float(np.mean(acceptance)),
                n_particles=n, cell_length=L, kinetic_coefficient=.5,
                cutoff=cutoff, potential_height=height, n_walkers=48,
                n_steps=150, burn_in=40, sample_every=5, n_replicas=3)
