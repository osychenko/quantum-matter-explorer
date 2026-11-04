"""Patch 37 scientific metadata, provenance, and restricted educational scope."""
import pytest

from qmc.educational_contract import catalog, run_experiment, validate_result
from qmc.educational_science import (
    IDEAL, RECORDED, science_map, validate_science_map,
)


def test_science_map_contract_and_parameter_paths():
    validate_science_map()
    result = science_map()
    assert result['schema_version'] == '1.0.0'
    assert set(result['profiles']) == {p['id'] for p in catalog()}
    assert sum(p['status'] == 'research_only' for p in result['problems']) == 1
    assert sum(p['status'] in ('recorded', 'interactive') for p in result['problems']) == 4


def test_science_map_research_cards_cannot_expose_live_presets():
    data = science_map()
    for problem in data['problems']:
        if problem['status'] == 'research_only':
            assert problem['preset_id'] is None
        else:
            assert problem['preset_id'] in data['profiles']
    with pytest.raises(ValueError, match='Unknown or research-only'):
        run_experiment('ground-state')
    with pytest.raises(ValueError, match='Unknown or research-only'):
        run_experiment('softcore-m12-exchange')


def test_science_map_is_read_only_copy():
    data = science_map()
    data['profiles'][IDEAL]['question'] = 'malicious replacement'
    data['problems'].clear()
    assert len(science_map()['problems']) == 5
    assert science_map()['profiles'][IDEAL]['question'] != 'malicious replacement'


def test_recorded_result_exposes_fixed_physics_parameters():
    result = run_experiment(RECORDED)
    validate_result(result)
    assert result['execution_mode'] == 'recorded_result'
    assert result['preset_status'] == 'candidate_recording'
    assert result['parameters']['M'] == 8
    assert result['parameters']['kinetic_coefficient'] == 0.5
    assert result['parameters']['n_background'] == 1
    assert result['parameters']['momentum_mode'] == [0, 0, 0]
    assert result['parameters']['t_fraction'] == .5
    assert result['provenance']['replicas'] == 8
    assert result['observables'][0]['uncertainty_kind'] == 'independent_replica_sem'


def test_ideal_and_recorded_metadata_have_distinct_comparison_protocols():
    data = science_map()
    ideal = data['profiles'][IDEAL]
    recorded = data['profiles'][RECORDED]
    assert ideal['method_id'] == 'analytic'
    assert recorded['method_id'] == 'worm'
    assert 'one parameter' in ideal['comparison']['title'].lower()
    assert 'spectral reference' in recorded['comparison']['title'].lower()
    assert 'M=8' in recorded['comparison']['changed']
    assert all('theory_ids' in p for p in data['profiles'].values())


def test_ideal_controls_still_respond_to_beta_and_cell_size():
    warm = run_experiment(IDEAL, {'beta': .5, 'cell_length': 2.8})
    cold = run_experiment(IDEAL, {'beta': 5., 'cell_length': 2.8})
    assert cold['observables'][0]['value'] > warm['observables'][0]['value']
