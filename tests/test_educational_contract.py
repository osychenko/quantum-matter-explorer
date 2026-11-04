"""MVP physics acceptance, versioned schema and public parameter restrictions."""
import json
import math
import pytest
from qmc.educational_contract import catalog, run_experiment, validate_result


def test_catalog_scopes_live_and_recorded_modes():
    presets = {x['id']: x for x in catalog()}
    assert presets['ideal-boson-addition-v1']['status'] == 'supported'
    assert presets['softcore-m8-baseline-recorded-v1']['mode'] == 'recorded_result'
    assert 'softcore-m12-exchange' not in presets


def test_ideal_green_analytic_identity_and_monotonicity():
    low = run_experiment('ideal-boson-addition-v1', {'beta': 1.0})
    high = run_experiment('ideal-boson-addition-v1', {'beta': 3.0})
    assert 1.0 < low['observables'][0]['value'] < 2.0
    assert low['observables'][0]['value'] < high['observables'][0]['value']
    assert high['observables'][0]['numerical_truncation_indicator'] < 1e-12
    assert high['observables'][0]['uncertainty'] is None
    validate_result(high)


def test_ideal_against_independent_brute_force_partition_sum():
    beta, L = 1.4, 2.8
    a = beta * 0.5 * (2 * math.pi / L)**2
    Z = sum(math.exp(-a*(i*i+j*j+k*k)) for i in range(-6,7)
             for j in range(-6,7) for k in range(-6,7))
    out = run_experiment('ideal-boson-addition-v1', {'beta':beta, 'cell_length':L})
    assert out['observables'][0]['value'] == pytest.approx(1 + 1/Z, abs=1e-12)


def test_frozen_recorded_provenance_and_no_unsupported_parameters():
    x = run_experiment('softcore-m8-baseline-recorded-v1')
    validate_result(x)
    assert x['execution_mode'] == 'recorded_result'
    assert x['preset_status'] == 'candidate_recording'
    assert x['observables'][0]['value'] == pytest.approx(1.685053810648142)
    with pytest.raises(ValueError):
        run_experiment('softcore-m8-baseline-recorded-v1', {'beta': 3.0})


@pytest.mark.parametrize('bad', [{'beta': float('nan')}, {'beta': 0.1},
                                  {'cell_length': 0.2}, {'steps': 10000},
                                  {'beta': True}, {'seed': 1}])
def test_reject_unapproved_experiment_inputs(bad):
    with pytest.raises(ValueError):
        run_experiment('ideal-boson-addition-v1', bad)


def test_stable_reproducible_serialization_and_fingerprint():
    x = run_experiment('ideal-boson-addition-v1')
    assert x == run_experiment('ideal-boson-addition-v1')
    assert len(x['fingerprint_sha256']) == 64
    assert json.loads(json.dumps(x, allow_nan=False)) == x
    x2 = run_experiment('ideal-boson-addition-v1', {'beta': 3.0})
    assert x['fingerprint_sha256'] != x2['fingerprint_sha256']


def test_reject_forged_mode_or_value():
    x = run_experiment('softcore-m8-baseline-recorded-v1')
    x['execution_mode'] = 'live_qmc'
    with pytest.raises(ValueError): validate_result(x)
    x = run_experiment('ideal-boson-addition-v1')
    x['observables'][0]['value'] = float('nan')
    with pytest.raises(ValueError): validate_result(x)
