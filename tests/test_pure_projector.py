"""Exact finite-basis pure OBDM regression tests for Patch 12."""

import csv
import json

import numpy as np
import pytest

from qmc.pure_projector import BoseHubbardRing, ExactBosonProjector
from examples.bosons.run_pure_projector import run_experiment


def engine(L=4, N=3, t=1., U=2., V=0.):
    return ExactBosonProjector(BoseHubbardRing(L, N, t, U, V))


@pytest.mark.parametrize("L,N,dimension", [(3, 1, 3), (4, 3, 20), (4, 4, 35), (5, 2, 15)])
def test_dimension_and_occupation_basis(L, N, dimension):
    e = engine(L, N)
    assert e.model.dimension == len(e.basis) == dimension
    assert np.all(e.basis >= 0)
    np.testing.assert_array_equal(e.basis.sum(axis=1), N)
    assert len(np.unique(e.basis, axis=0)) == dimension
    np.testing.assert_allclose(e.hamiltonian, e.hamiltonian.T, atol=0, rtol=0)


@pytest.mark.parametrize("N", [1, 2, 4])
def test_free_bosons_exact_coherence(N):
    e = engine(4, N, t=1., U=0., V=0.)
    state = e.measure(float("inf"))
    np.testing.assert_allclose(state.g1, 1., atol=2e-13)
    assert state.n0_fraction == pytest.approx(1., abs=2e-13)
    assert state.condensate_eigenvalue_fraction == pytest.approx(1., abs=2e-13)
    assert state.energy == pytest.approx(-2.0 * N, abs=2e-12)
    # The coherent seed is exactly the free-boson ground state.
    e0 = e.measure(0., e.trial("coherent"))
    np.testing.assert_allclose(e0.obdm, state.obdm, atol=2e-13)
    assert e0.energy == pytest.approx(state.energy, abs=2e-12)


def test_unit_filled_mott_zero_hopping():
    e = engine(4, 4, t=0., U=4., V=0.)
    m = e.measure(float("inf"), e.trial("coherent"))
    assert m.gap == pytest.approx(4.)
    np.testing.assert_allclose(m.g1, [1, 0, 0, 0], atol=1e-13)
    np.testing.assert_allclose(m.obdm, np.eye(4), atol=1e-13)
    np.testing.assert_allclose(m.occupations, 1., atol=1e-13)
    assert m.n0_fraction == pytest.approx(1. / 4., abs=1e-13)


def test_obdm_hermitian_positive_and_trace():
    e = engine(5, 3, t=1., U=3., V=0.2)
    for kind in ("coherent", "gutzwiller", "localized"):
        r = e.measure(0.5, e.trial(kind))
        np.testing.assert_allclose(r.obdm, r.obdm.conj().T, atol=1e-12)
        assert np.min(np.linalg.eigvalsh(r.obdm)) >= -1e-12
        assert np.trace(r.obdm).real == pytest.approx(3, abs=1e-12)
        assert r.g1[0] == pytest.approx(1, abs=1e-12)
        assert -1e-13 <= r.n0_fraction <= 1 + 1e-13
        assert r.condensate_eigenvalue_fraction >= r.n0_fraction - 1e-12
        assert np.sum(r.occupations) == pytest.approx(3, abs=1e-12)


def test_projection_agrees_with_manual_spectral_formula():
    e = engine(4, 3)
    trial = e.trial("gutzwiller", penalty=0.9)
    beta = 0.31
    eigenvectors = e.eigenvectors
    factors = np.exp(-beta * (e.energies - e.energies[0]))
    manually = eigenvectors @ (factors * (eigenvectors.T @ trial))
    manually /= np.linalg.norm(manually)
    np.testing.assert_allclose(e.projected_state(beta, trial), manually, atol=1e-13)
    # Nonzero beta is a symmetric projection: the observable uses |psi_beta|^2.
    measured = e.measure(beta, trial)
    np.testing.assert_allclose(measured.obdm, e.obdm(manually), atol=1e-13)


@pytest.mark.parametrize("kind", ["coherent", "gutzwiller", "localized", "uniform"])
def test_finite_beta_converges_to_ground_state(kind):
    e = engine(4, 3, t=1., U=2., V=.3)
    trial = e.trial(kind)
    pure = e.measure(float("inf"), trial)
    approx = e.measure(6., trial)
    np.testing.assert_allclose(approx.obdm, pure.obdm, atol=1e-7)
    assert approx.energy == pytest.approx(pure.energy, abs=1e-7)
    assert pure.n0_fraction == pytest.approx(e.measure(float("inf")).n0_fraction, abs=1e-13)


def test_projection_lowers_variational_energy():
    e = engine(5, 3)
    trial = e.trial("localized")
    energies = [e.measure(beta, trial).energy for beta in (0., 0.1, 0.3, 1., 4.)]
    assert all(a >= b - 1e-10 for a, b in zip(energies[:-1], energies[1:]))
    assert energies[-1] > e.energies[0] - 1e-10


def test_hamiltonian_three_site_bonds_are_not_double_counted():
    e = engine(3, 1, t=1., U=0., V=0.)
    np.testing.assert_allclose(e.hamiltonian, -np.ones((3, 3)) + np.eye(3))
    np.testing.assert_allclose(e.energies, [-2., 1., 1.], atol=1e-13)


def test_one_particle_on_site_or_neighbor_interactions_vanish():
    a = engine(4, 1, t=0.8, U=0., V=0.)
    b = engine(4, 1, t=0.8, U=100., V=50.)
    np.testing.assert_allclose(a.hamiltonian, b.hamiltonian)
    np.testing.assert_allclose(a.measure(float("inf")).g1, 1., atol=1e-12)


@pytest.mark.parametrize("kwargs", [
    {"n_sites": 2}, {"n_sites": True}, {"n_bosons": 0},
    {"n_bosons": 1.2}, {"hopping": -1}, {"onsite_u": -0.1},
    {"nearest_v": float("nan")}, {"hopping": float("inf")},
    {"max_dimension": 0}, {"n_sites": 10, "n_bosons": 10},
])
def test_bad_models(kwargs):
    with pytest.raises(ValueError):
        BoseHubbardRing(**kwargs)


def test_bad_trials_and_projection():
    e = engine()
    with pytest.raises(ValueError, match="unknown"):
        e.trial("other")
    with pytest.raises(ValueError, match="penalty"):
        e.trial("coherent", penalty=-1)
    with pytest.raises(ValueError, match="beta"):
        e.projected_state(-.1)
    with pytest.raises(ValueError, match="beta"):
        e.projected_state(float("nan"))
    with pytest.raises(ValueError, match="trial"):
        e.projected_state(1., np.ones((3,)))
    with pytest.raises(ValueError, match="trial"):
        e.projected_state(1., np.full(e.model.dimension, np.nan))
    with pytest.raises(ValueError, match="nonzero"):
        e.projected_state(0, np.zeros(e.model.dimension))
    with pytest.raises(ValueError, match="state"):
        e.obdm(np.ones((2,)))
    with pytest.raises(ValueError, match="nonzero"):
        e.obdm(np.zeros(e.model.dimension))


def test_zero_ground_overlap_detected_at_infinite_beta():
    e = engine(4, 4, t=0., U=4.)
    with pytest.raises(ValueError, match="zero overlap"):
        e.projected_state(float("inf"), e.trial("localized"))


def test_quick_example_and_outputs(tmp_path):
    a, ground = run_experiment(output_dir=tmp_path / "a", quick=True)
    b, other = run_experiment(output_dir=tmp_path / "b", quick=True)
    assert ground.n0_fraction == other.n0_fraction
    assert {"coherent", "gutzwiller", "localized"} == set(a)
    for name in ("projection.csv", "obdm.csv", "natural_occupations.csv", "protocol.json"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()
    for name in ("projection_convergence.png", "obdm_comparison.png"):
        assert (tmp_path / "a" / name).stat().st_size > 1000
    with (tmp_path / "a" / "projection.csv").open() as f:
        data = list(csv.DictReader(f))
    assert len(data) == 3 * 7
    assert len([x for x in data if x["beta"] == "infinity"]) == 3
    assert all(float(x["n0_fraction"]) >= 0 for x in data)
    protocol = json.loads((tmp_path / "a" / "protocol.json").read_text())
    assert protocol["hilbert_dimension"] == 20
    assert protocol["exact_ground_n0_fraction"] == pytest.approx(ground.n0_fraction)
