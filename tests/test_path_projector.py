"""Sign-free open-path Monte Carlo against an independent exact projector."""

from dataclasses import replace

import numpy as np
import pytest

from qmc.path_projector import OpenPathProjector, PathConfig
from qmc.pure_projector import BoseHubbardRing, ExactBosonProjector


def small_model():
    return BoseHubbardRing(n_sites=3, n_bosons=2, hopping=1.0,
                           onsite_u=1.7, nearest_v=0.25)


@pytest.mark.parametrize('bad', [0, -1, float('inf'), float('nan')])
def test_bad_dt(bad):
    with pytest.raises(ValueError):
        PathConfig(dt=bad)


@pytest.mark.parametrize('field,bad', [
    ('half_steps', 0), ('n_samples', 0), ('sample_stride', 0),
    ('gibbs_updates', -1), ('burn_in', -1), ('seed', -1),
    ('half_steps', 2.5), ('n_samples', True),
])
def test_bad_integer_config(field, bad):
    with pytest.raises(ValueError):
        PathConfig(**{field: bad})


def test_nonpositive_trial_fails():
    m = small_model()
    ref = ExactBosonProjector(m)
    with pytest.raises(ValueError, match='strictly positive'):
        OpenPathProjector(m, PathConfig(), ref.trial('localized'))
    with pytest.raises(ValueError):
        OpenPathProjector(m, PathConfig(), np.ones(m.dimension + 1))
    with pytest.raises(ValueError, match='hopping'):
        OpenPathProjector(replace(m, hopping=0), PathConfig())


def test_kernel_positive_and_exact_spectral_decomposition():
    m = small_model()
    cfg = PathConfig(dt=0.31)
    sampler = OpenPathProjector(m, cfg)
    assert np.all(sampler.T > 0)
    np.testing.assert_allclose(sampler.T, sampler.T.T, atol=1e-13)
    np.testing.assert_allclose(sampler.T.sum(axis=1), sampler.rowsums)
    np.testing.assert_allclose(sampler.Q.sum(axis=1), 1., atol=1e-13)
    ref = sampler.exact
    expected = (ref.eigenvectors * np.exp(-cfg.dt*(ref.energies - ref.energies[0]))) @ ref.eigenvectors.T
    np.testing.assert_allclose(sampler.T, expected, atol=1e-13)


def test_midpoint_bridge_is_exact_conditional_obdm():
    m = small_model()
    cfg = PathConfig(dt=.2, half_steps=2)
    sampler = OpenPathProjector(m, cfg)
    basis = sampler.exact.basis
    for l in range(m.dimension):
        for r in range(m.dimension):
            sampler.path[cfg.half_steps - 1] = l
            sampler.path[cfg.half_steps + 1] = r
            est = sampler.midpoint_obdm()
            np.testing.assert_allclose(np.trace(est), m.n_bosons, atol=1e-12)
            # Operator insertion conditioned on the two outer beads.
            B = sampler.T @ sampler.T
            den = B[l, r]
            assert den > 0
            for i in range(m.n_sites):
                for j in range(m.n_sites):
                    O = np.zeros_like(B)
                    for b, occ in enumerate(basis):
                        if i == j:
                            O[b,b] = occ[i]
                        elif occ[j]:
                            v = occ.copy(); v[j] -= 1; v[i] += 1
                            a = sampler.exact._index[tuple(v)]
                            O[a,b] = np.sqrt(occ[j] * (occ[i] + 1))
                    assert est[i,j] == pytest.approx((sampler.T @ O @ sampler.T)[l,r] / den)


def test_stationary_bridge_average_equals_exact_symmetric_projector():
    m = small_model()
    cfg = PathConfig(dt=.23, half_steps=3)
    sampler = OpenPathProjector(m, cfg)
    # Sum over the *exact marginal* of the two beads around the midpoint.
    u = np.linalg.matrix_power(sampler.T, cfg.half_steps - 1) @ sampler.phi
    weights = u[:, None] * sampler._bridge_denominator * u[None, :]
    weights /= weights.sum()
    averaged = np.einsum('lr,ijlr,lr->ij', weights, sampler._operator_bridges,
                         1 / sampler._bridge_denominator, optimize=True)
    exact = sampler.exact.measure(cfg.beta, sampler.phi)
    np.testing.assert_allclose(averaged, exact.obdm, atol=1e-12)
    averaged_energy = np.sum(weights * sampler._energy_bridge / sampler._bridge_denominator)
    assert averaged_energy == pytest.approx(exact.energy, abs=1e-12)


def test_reptation_acceptance_matches_direct_mh_ratio():
    m = small_model()
    p = OpenPathProjector(m, PathConfig(dt=.3, half_steps=2, seed=9))
    x = np.array([0,1,2,1,0], dtype=int)
    z = 3
    y = np.r_[x[1:],z]
    def logweight(path):
        return (p.log_phi[path[0]] + p.log_phi[path[-1]]
                + sum(np.log(p.T[a,b]) for a,b in zip(path[:-1],path[1:])))
    direct = (logweight(y)-logweight(x)
              + np.log(p.Q[x[1], x[0]]) - np.log(p.Q[x[-1], z]))
    simplified = (p.log_phi[x[1]]+p.log_phi[z]-p.log_phi[x[0]]-p.log_phi[x[-1]]
                  + np.log(p.rowsums[x[-1]])-np.log(p.rowsums[x[1]]))
    assert direct == pytest.approx(simplified, abs=1e-12)


def test_gibbs_weight_is_valid_and_updates_path():
    p = OpenPathProjector(small_model(), PathConfig(seed=4))
    for position in [0, 1, len(p.path)//2, len(p.path)-1]:
        p.gibbs_update(position)
    for _ in range(60):
        p.step()
    assert np.all((p.path >= 0) & (p.path < p.model.dimension))
    assert p.attempted == 60
    assert 0 < p.accepted <= p.attempted


def test_samples_agree_with_exact_ground_projector_within_mc_tolerance():
    model = small_model()
    cfg = PathConfig(dt=.25, half_steps=4, burn_in=1200,
                     n_samples=4000, sample_stride=3, gibbs_updates=4, seed=205)
    measured = OpenPathProjector(model, cfg).run()
    expected = ExactBosonProjector(model).measure(cfg.beta)
    assert measured.beta == pytest.approx(cfg.beta)
    assert measured.samples.shape == (cfg.n_samples, 2)
    np.testing.assert_allclose(measured.g1, expected.g1, atol=.055)
    assert measured.n0_fraction == pytest.approx(expected.n0_fraction, abs=.04)
    assert measured.energy == pytest.approx(expected.energy, abs=.055)
    np.testing.assert_allclose(np.trace(measured.obdm), model.n_bosons, atol=1e-11)
    assert 0 < measured.reptation_acceptance <= 1


def test_run_is_deterministic_for_identical_seed():
    model = small_model()
    cfg = PathConfig(burn_in=20, n_samples=40, gibbs_updates=3, seed=89)
    a = OpenPathProjector(model, cfg).run()
    b = OpenPathProjector(model, cfg).run()
    assert np.array_equal(a.samples, b.samples)
    assert np.array_equal(a.obdm, b.obdm)
    assert a.energy == b.energy
    assert a.reptation_acceptance == b.reptation_acceptance


def test_single_particle_free_ring_condensate_exact():
    m = BoseHubbardRing(n_sites=3, n_bosons=1, hopping=1., onsite_u=0)
    p = OpenPathProjector(m, PathConfig(dt=.4, half_steps=2,
                                       burn_in=200, n_samples=1300,
                                       sample_stride=3, seed=17))
    result = p.run()
    assert result.n0_fraction == pytest.approx(1.0, abs=.055)
    np.testing.assert_allclose(result.g1, [1.,1.,1.], atol=.08)


def test_single_site_density_consistency():
    m = small_model()
    cfg = PathConfig(half_steps=3)
    p = OpenPathProjector(m, cfg)
    for _ in range(10):
        p.step()
        assert np.trace(p.midpoint_obdm()) == pytest.approx(m.n_bosons, abs=1e-11)


def test_exact_time_interpretation():
    c = PathConfig(dt=.2, half_steps=6)
    assert c.beta == pytest.approx(1.2)
    p = OpenPathProjector(small_model(), c)
    assert len(p.path) == 13  # twelve transfer links = total projection 2*beta


def test_example_quick_outputs_and_reproducibility(tmp_path):
    from examples.bosons.run_path_projector import ExperimentConfig, run_experiment
    cfg = ExperimentConfig(n_sites=3, n_bosons=2,
                           half_steps=(2,3), trials=("coherent", "gutzwiller"),
                           n_replicas=2, burn_in=10, n_samples=25,
                           sample_stride=2, gibbs_updates=2, seed=71)
    a, b, c = run_experiment(cfg, tmp_path / 'one')
    assert len(a) == 8 and len(b) == 4 and len(c) == 4 * cfg.n_sites
    for name in ('replicas.csv','summary.csv','g1.csv','protocol.json',
                 'projection_convergence.png','g1_comparison.png'):
        assert (tmp_path / 'one' / name).stat().st_size > 10
    a2, b2, c2 = run_experiment(cfg, tmp_path / 'two')
    assert a == a2 and b == b2 and c == c2


def test_experiment_rejects_missing_replicas(tmp_path):
    from examples.bosons.run_path_projector import ExperimentConfig, run_experiment
    with pytest.raises(ValueError, match='distinct positive'):
        run_experiment(ExperimentConfig(n_replicas=1), tmp_path)


def test_dense_dimension_safety_limit():
    m = BoseHubbardRing(n_sites=6, n_bosons=5, hopping=1.,
                        max_dimension=300)  # C(10,5) = 252
    with pytest.raises(ValueError, match='dimension'):
        OpenPathProjector(m, PathConfig())
