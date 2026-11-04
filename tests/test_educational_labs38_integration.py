"""Slow-ish scientific smoke with the *actual* QMC VMC/observable modules."""
import importlib.util
import math
import pytest

pytestmark=pytest.mark.skipif(importlib.util.find_spec('qmc.vmc') is None,
                             reason='Complete QMC research sources unavailable')


def test_actual_harmonic_vmc_matches_analytic_trial_energy():
    from qmc.educational_labs import harmonic_vmc
    for alpha in (1.0,1.35):
        data=harmonic_vmc(alpha,1.0)
        assert data['n_replicas']==3
        assert math.isfinite(data['mean'])
        assert math.isclose(data['mean'],data['trial_expectation'],abs_tol=.13)
        assert math.isclose(data['exact_ground'],1.5)
    assert harmonic_vmc(1.0,1.0)['sem'] < 1e-8


def test_actual_structural_vmc_returns_finite_references():
    from qmc.educational_labs import softcore_structure_vmc
    data=softcore_structure_vmc(0.0)
    assert len(data['radius'])==len(data['gr'])==12
    assert len(data['sk'])==4
    assert all(math.isfinite(v) and v>=0 for v in data['gr']+data['sk'])
    assert abs(sum(data['gr'][2:])/len(data['gr'][2:])-1) < .3
    assert abs(sum(data['sk'])/len(data['sk'])-1) < .25
