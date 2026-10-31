"""Primitive-action 3D open-path weight, detailed balance, and reference tests."""

from dataclasses import replace

import numpy as np
import pytest

from qmc.cells import CubicCell, OrthorhombicCell
from qmc.interacting_path import InteractingPathConfig, PrimitiveInteractingPath


class ConstantPotential:
    def __init__(self, height=0.0, cell=None):
        self.height, self.cell = height, cell

    def __call__(self, R):
        return np.full(len(R), self.height, dtype=float)


class UniformTrial:
    def log_psi(self, R):
        return np.zeros(len(R))

    def grad_log_psi(self, R):
        return np.zeros_like(R)

    def laplacian_log_psi(self, R):
        return np.zeros(len(R))


class CosinePotential:
    def __init__(self, cell, amplitude=1.0):
        self.cell, self.amplitude = cell, amplitude

    def __call__(self, R):
        return self.amplitude * np.sum(
            1.0 - np.cos(2 * np.pi * R[..., 0] / self.cell.length), axis=1
        )


@pytest.fixture
def config():
    return InteractingPathConfig(time_step=0.2, n_links=6, n_steps=80,
                                 burn_in=15, thin=5, seed=324)


def create(config, potential=None, n=2, trial=None):
    cell = CubicCell(3.5)
    return PrimitiveInteractingPath(
        cell, potential if potential is not None else ConstantPotential(0.),
        trial if trial is not None else UniformTrial(), n, config)


def test_config_validation(config):
    assert config.beta == pytest.approx(0.6)
    modifications = (
        dict(time_step=0), dict(time_step=float('nan')), dict(n_links=5),
        dict(n_links=0), dict(n_steps=1), dict(burn_in=80),
        dict(thin=0), dict(move_sigma=-1), dict(refresh_probability=-0.1),
        dict(translate_probability=1.01),
        dict(refresh_probability=0.75, translate_probability=0.4),
        dict(seed=-1), dict(seed=True), dict(n_steps=17, burn_in=15, thin=8),
    )
    for kwargs in modifications:
        with pytest.raises(ValueError):
            replace(config, **kwargs).validate()


def test_path_shape_and_weight(config):
    sampler = create(config)
    assert sampler.path.shape == (7, 2, 3)
    assert np.all(np.isfinite(sampler.path))
    assert np.isfinite(sampler.log_weight(sampler.path))
    assert np.all(sampler.path >= 0)
    assert np.all(sampler.path < 3.5)
    with pytest.raises(ValueError, match='path'):
        sampler.log_weight(sampler.path[:-1])


def test_constant_potential_action_and_endpoints(config):
    sampler = create(config, ConstantPotential(2.5))
    path = sampler.path
    free = create(config, ConstantPotential(0.0))
    np.testing.assert_allclose(path, free.path)
    assert sampler.log_weight(path) - free.log_weight(path) == pytest.approx(-2.5 * 1.2)
    assert sampler.local_energy_at_endpoints() == pytest.approx(2.5)
    assert sampler._action_potential(path) == pytest.approx(2.5 * 1.2)


def test_all_moves_accept_constant_potential(config):
    sampler = create(config, ConstantPotential(1.8))
    for _ in range(12):
        sampler.move_bead()  # Kinetic action makes this a nontrivial MH step.
        assert sampler.move_translate()  # Free paths are translation invariant.
        assert sampler.move_refresh()  # Exact independent free-path proposals.
    result = sampler.run()
    assert result.acceptances["translate"] == 1
    assert result.acceptances["refresh"] == 1
    assert 0 <= result.acceptances["bead"] <= 1


def test_local_log_acceptance_exact(config):
    potential = CosinePotential(CubicCell(3.5), 1.6)
    sampler = create(config, potential)
    base = sampler.path.copy()
    recorded = []
    original = sampler._accept
    try:
        sampler._accept = lambda k, p, d: recorded.append((k, p.copy(), d))
        sampler.move_bead(bead=0, particle=1)
        sampler.move_bead(bead=3, particle=0)
        sampler.move_bead(bead=6, particle=1)
        sampler.move_translate(particle=0)
        sampler.move_refresh()
    finally:
        sampler._accept = original
    assert len(recorded) == 5
    for kind, proposed, ratio in recorded:
        if kind == 'refresh':
            # Independent free proposal ratio cancels its free-link action.
            correct = (-sampler._action_potential(proposed)
                       + sampler._action_potential(base))
        else:
            correct = sampler.log_weight(proposed) - sampler.log_weight(base)
        assert ratio == pytest.approx(correct, abs=1e-10)


def test_periodic_invariance_of_full_weight(config):
    potential = CosinePotential(CubicCell(3.5), 0.9)
    sampler = create(config, potential)
    path = sampler.path.copy()
    shifted = path.copy()
    shifted[2, 0] += [3.5, -7.0, 0.0]
    shifted[-1, 1] += [0.0, 3.5, 10.5]
    assert sampler.log_weight(shifted) == pytest.approx(sampler.log_weight(path))


def test_reproducible_run(config):
    a, b = create(config), create(config)
    ra, rb = a.run(), b.run()
    np.testing.assert_array_equal(ra.midpoint_potential, rb.midpoint_potential)
    np.testing.assert_array_equal(ra.thermodynamic_energy, rb.thermodynamic_energy)
    np.testing.assert_array_equal(ra.final_path, rb.final_path)
    assert ra.attempted == rb.attempted
    assert len(ra.midpoint_potential) == 13
    assert sum(ra.attempted.values()) == config.n_steps


def test_constant_energy_exact_but_thermodynamic_noisy(config):
    sampler = create(config, ConstantPotential(1.7))
    r = sampler.run()
    np.testing.assert_allclose(r.midpoint_potential, 1.7)
    np.testing.assert_allclose(r.endpoint_local_energy, 1.7)
    assert np.all(np.isfinite(r.thermodynamic_energy))
    assert all(0 <= x <= 1 for x in r.acceptances.values() if np.isfinite(x))


def test_one_particle_softcore_exact_free_state(config):
    from qmc.bosons import SoftCorePairPotential, SoftCoreJastrowTrial
    cell = CubicCell(3.5)
    sampler = PrimitiveInteractingPath(
        cell, SoftCorePairPotential(cell, height=3, cutoff=1.4),
        SoftCoreJastrowTrial(cell, amplitude=0.5, cutoff=1.4), 1, config)
    r = sampler.run()
    np.testing.assert_allclose(r.midpoint_potential, 0.)
    np.testing.assert_allclose(r.endpoint_local_energy, 0.)


def test_nonconstant_interacting_softcore_smoke(config):
    from qmc.bosons import SoftCorePairPotential, SoftCoreJastrowTrial
    cell = CubicCell(3.5)
    sampler = PrimitiveInteractingPath(
        cell, SoftCorePairPotential(cell, height=2.0, cutoff=1.4),
        SoftCoreJastrowTrial(cell, amplitude=0.6, cutoff=1.4), 3,
        replace(config, n_steps=110, burn_in=25))
    result = sampler.run()
    assert len(result.midpoint_potential) == 17
    assert np.all(result.midpoint_potential >= 0)
    assert np.all(np.isfinite(result.endpoint_local_energy))
    assert np.all(np.isfinite(result.thermodynamic_energy))


def test_wrong_cell_and_invalid_trial(config):
    cell = CubicCell(4.0)
    with pytest.raises(ValueError, match='same periodic cell'):
        PrimitiveInteractingPath(cell, CosinePotential(CubicCell(3.5)), UniformTrial(), 1, config)
    with pytest.raises(TypeError, match='log_psi'):
        PrimitiveInteractingPath(cell, ConstantPotential(0), object(), 1, config)
    with pytest.raises(TypeError, match='OrthorhombicCell'):
        PrimitiveInteractingPath(object(), ConstantPotential(0), UniformTrial(), 1, config)
    with pytest.raises(ValueError):
        PrimitiveInteractingPath(cell, ConstantPotential(0), UniformTrial(), 0, config)


def test_free_thermodynamic_energy_average_is_zero():
    cfg = InteractingPathConfig(time_step=0.25, n_links=4, n_steps=1900,
                                burn_in=150, thin=1, refresh_probability=1.0,
                                translate_probability=0, seed=222)
    sampler = create(cfg, n=1)
    e = sampler.run().thermodynamic_energy
    # With refresh-only the free paths are independent, and E=0 exactly in
    # expectation. This broad tolerance is appropriate for a noisy estimator.
    assert abs(e.mean()) < 0.6


def test_exact_fourier_cosine_projected_midpoint():
    """Independent Fourier-basis 1-particle benchmark for diagonal midpoint V."""
    L, D, amp, beta = 3.5, 0.5, 1.0, 0.9
    modes = np.arange(-8, 9)
    k = 2 * np.pi * modes / L
    h = np.diag(D*k*k + amp)
    for i in range(len(modes)-1):
        h[i, i+1] = h[i+1, i] = -amp/2
    eigenvalues, eigenvectors = np.linalg.eigh(h)
    initial = np.zeros(len(modes)); initial[8] = 1
    projected = eigenvectors @ (np.exp(-beta * (eigenvalues-eigenvalues[0])) * (eigenvectors.T @ initial))
    projected /= np.linalg.norm(projected)
    expectation = float(amp - amp * np.dot(projected[:-1], projected[1:]))
    cell = CubicCell(L)
    cfg = InteractingPathConfig(time_step=0.15, n_links=12, n_steps=5500,
                                burn_in=800, thin=3, move_sigma=0.5,
                                refresh_probability=0.18, translate_probability=0.13,
                                seed=104)
    sampler = PrimitiveInteractingPath(cell, CosinePotential(cell, amp), UniformTrial(), 1, cfg, kinetic_coefficient=D)
    data = sampler.run().midpoint_potential
    # Primitive timestep bias and autocorrelation: use a deliberately
    # non-tight finite-sample tolerance, NOT a precision benchmark.
    assert abs(data.mean() - expectation) < 0.18


def test_thermodynamic_energy_matches_action_derivative(config):
    cell = CubicCell(3.5)
    sampler = create(config, CosinePotential(cell, 1.25), n=2)
    target = sampler.thermodynamic_energy()
    dt, m, eps = config.time_step, config.n_links, 1e-5
    sampler.config = replace(config, time_step=dt + eps)
    plus = sampler.log_weight(sampler.path)
    sampler.config = replace(config, time_step=dt - eps)
    minus = sampler.log_weight(sampler.path)
    sampler.config = config
    numerical = -(plus-minus)/(2*eps*m)
    assert target == pytest.approx(numerical, rel=2e-6, abs=2e-6)


def test_interacting_particle_exchange_symmetry(config):
    from qmc.bosons import SoftCorePairPotential, SoftCoreJastrowTrial
    cell = CubicCell(3.5)
    sampler = PrimitiveInteractingPath(
        cell, SoftCorePairPotential(cell, height=2, cutoff=1.4),
        SoftCoreJastrowTrial(cell, amplitude=0.8, cutoff=1.4), 3, config)
    path = sampler.path.copy()
    assert sampler.log_weight(path) == pytest.approx(sampler.log_weight(path[:, ::-1]))


def test_quick_example_output_contract(tmp_path):
    import csv
    from examples.bosons.run_interacting_path import run_experiment
    out = tmp_path / 'experiment'
    raw, summary = run_experiment(out, quick=True)
    assert len(raw) == 12
    assert len(summary) == 6
    assert len({row['seed'] for row in raw}) == 12
    for name in ('replicas.csv', 'summary.csv', 'protocol.json',
                 'timestep_convergence.png', 'thermodynamic_energy.png'):
        assert (out / name).stat().st_size > 100
    with (out / 'summary.csv').open() as f:
        parsed = list(csv.DictReader(f))
    assert len(parsed) == len(summary)
    assert all(float(r['endpoint_energy']) == 0 for r in parsed if r['case'] == 'free_reference')
    assert all(float(r['exact_endpoint_energy']) > 0 for r in parsed if r['case'] == 'cosine_reference')
