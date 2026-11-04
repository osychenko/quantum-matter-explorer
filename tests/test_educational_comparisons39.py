"""Patch 39: scientific semantics, curated parameter boundaries and consistent records."""
from copy import deepcopy
import math
import pytest

from qmc import educational_comparisons as comparisons
from qmc.educational_contract import run_experiment


def test_curated_registry_and_immutability():
    ids = {x['id'] for x in comparisons.comparison_catalog()}
    assert ids == {comparisons.OSC, comparisons.STRUCT, comparisons.IDEAL, comparisons.WORM}
    catalog = comparisons.comparison_catalog()
    catalog[0]['parameters'][0]['default'] = -1
    assert comparisons.comparison_catalog()[0]['parameters'][0]['default'] > 0
    assert comparisons.comparison_requires_live(comparisons.OSC)
    assert not comparisons.comparison_requires_live(comparisons.IDEAL)
    assert not comparisons.comparison_requires_live('malicious-method')


def test_recorded_versus_reference_is_same_observable_not_a_new_run():
    answer = comparisons.run_comparison(comparisons.WORM)
    a, b = answer['arms']
    assert a['primary_observable']['id'] == b['primary_observable']['id']
    assert b['evidence_kind'] == 'finite_basis_reference_from_recorded_experiment'
    assert a['experiment']['fingerprint_sha256'] == b['experiment']['fingerprint_sha256']
    assert answer['analysis']['difference_b_minus_a'] == pytest.approx(
        1.6910255956087066 - 1.685053810648142)
    assert answer['analysis']['difference_z'] is None
    assert 'continuum' in ' '.join(answer['interpretation']['not_supported'])


def test_ideal_interacting_not_causally_isolated():
    answer = comparisons.run_comparison(comparisons.IDEAL)
    a, b = answer['arms']
    assert a['experiment']['execution_mode']=='analytic_control'
    assert b['experiment']['execution_mode']=='recorded_result'
    assert a['experiment']['parameters']['beta'] == b['experiment']['parameters']['beta']
    assert a['experiment']['parameters']['cell_length'] == b['experiment']['parameters']['cell_length']
    assert answer['comparison_kind'] == 'confounded_cross_model_contrast'
    assert answer['analysis']['combined_sem'] is None
    assert 'NOT an isolated causal' in answer['analysis']['inference']
    assert answer['fingerprint_sha256'] == comparisons.run_comparison(comparisons.IDEAL)['fingerprint_sha256']


@pytest.mark.parametrize('cid,params', [
    ('unknown', {}),
    (comparisons.WORM, {'beta': 2.4}),
    (comparisons.OSC, {'alpha_left': float('nan')}),
    (comparisons.OSC, {'alpha_left': True}),
    (comparisons.OSC, {'alpha_left': 0.1}),
    (comparisons.OSC, {'alpha_left': 1.0, 'alpha_right': 1.0}),
    (comparisons.STRUCT, {'amplitude_left': 1.5}),
    (comparisons.STRUCT, {'amplitude_left': 0.5, 'amplitude_right': 0.5}),
    (comparisons.IDEAL, []),
])
def test_comparison_parameters_are_whitelisted(cid, params):
    with pytest.raises(ValueError):
        comparisons.run_comparison(cid,params)


def _fake_result(preset, params):
    # A minimal result satisfying the existing Patch 32/38 public schema,
    # used to test comparison logic without masking the actual research sampler.
    output = deepcopy(run_experiment('ideal-boson-addition-v1'))
    output['experiment_id'] = preset
    output['preset_status'] = 'educational_variational'
    output['execution_mode'] = 'bounded_live_vmc'
    output['provenance'] = dict(method='live_metropolis_vmc', seeds=[100, 101, 102], replicas=3)
    output['fingerprint_sha256'] = preset + ': ' + repr(sorted(params.items()))
    if preset == 'harmonic-oscillator-vmc-v1':
        a=params['trial_alpha'];w=params['omega']
        output['parameters'] = {'trial_alpha':a, 'omega':w, 'kinetic_coefficient':.5,
                                'dimensions':3,'n_walkers':128,'n_steps':260,'burn_in':60,'sample_every':2}
        output['observables'] = [dict(id='vmc_ground_energy', value=.75*(a+w*w/a),
                                    uncertainty=.04,units='energy',uncertainty_kind='independent_replica_sem',
                                    analytic_trial_expectation=.75*(a+w*w/a),exact_ground_energy=1.5*w)]
    else:
        amp=params['jastrow_amplitude']
        output['parameters'] = dict(jastrow_amplitude=amp, n_particles=6,
                                    cell_length=5.,kinetic_coefficient=.5,
                                    cutoff=1.2,potential_height=3.,n_walkers=48,n_steps=150,burn_in=40,sample_every=5)
        output['observables'] = [dict(id='variational_energy_per_particle',value=amp+.4,
                                    uncertainty=.02,units='energy per particle',uncertainty_kind='independent_replica_sem')]
        output['distributions'] = [
            dict(id='pair_correlation',label='g(r)',x=[.2,.4],
                 values=[1.0,1.0-amp/2],stderr=[.1,.1],x_units='length'),
            dict(id='structure_factor',label='S(k)',x=[1.2,1.4],indices=[[1,0,0],[1,1,0]],
                 values=[1.,1.+amp/3],stderr=[.1,.1],x_units='inverse length')]
    return output


def test_oscillator_comparison_keeps_hamiltonian_fixed(monkeypatch):
    monkeypatch.setattr(comparisons,'run_experiment',_fake_result)
    out = comparisons.run_comparison(comparisons.OSC,dict(alpha_left=.6,alpha_right=1.3,omega=1.2))
    assert out['comparison_kind']=='controlled_variational_choice'
    assert [a['experiment']['parameters']['omega'] for a in out['arms']] == [1.2,1.2]
    assert out['analysis']['combined_sem'] is None  # same seeds: unknown covariance
    assert out['analysis']['difference_z'] is None
    assert all(a['primary_observable']['exact_ground_energy']==pytest.approx(1.8) for a in out['arms'])


def test_structure_comparison_preserves_same_physical_model(monkeypatch):
    monkeypatch.setattr(comparisons,'run_experiment',_fake_result)
    out=comparisons.run_comparison(comparisons.STRUCT,dict(amplitude_left=0,amplitude_right=.8))
    assert out['arms'][0]['experiment']['parameters']['potential_height']==3.
    assert out['arms'][1]['experiment']['parameters']['potential_height']==3.
    assert out['arms'][1]['experiment']['distributions'][0]['id']=='pair_correlation'
    assert out['analysis']['difference_b_minus_a'] == pytest.approx(.8)
    assert 'NOT a measurement' in ' '.join(out['interpretation']['not_supported'])


def test_mismatched_observables_are_rejected():
    a = run_experiment('ideal-boson-addition-v1')
    bad = deepcopy(a)
    bad['observables'][0]['id'] = 'different_observable'
    with pytest.raises(ValueError, match='Incompatible observables'):
        comparisons._base(comparisons.IDEAL,{},[comparisons._arm('A',a),comparisons._arm('B',bad)],
                          held_fixed=[],changed=[],confounders=[],conclusions=[],cannot_conclude=[],relationship='')


def test_unmatched_structural_bins_rejected(monkeypatch):
    def altered(preset, params):
        result = _fake_result(preset, params)
        if params['jastrow_amplitude'] > .5:
            result['distributions'][0]['x'] = [.2,.45]
        return result
    monkeypatch.setattr(comparisons,'run_experiment',altered)
    with pytest.raises(ValueError, match='nonmatching grids'):
        comparisons.run_comparison(comparisons.STRUCT)


def test_same_model_not_assumed_when_physics_mismatches(monkeypatch):
    def altered(preset, params):
        result = _fake_result(preset, params)
        if params['trial_alpha'] > 1:
            result['parameters']['omega'] = 1.3
        return result
    monkeypatch.setattr(comparisons,'run_experiment',altered)
    with pytest.raises(ValueError, match='not held fixed'):
        comparisons.run_comparison(comparisons.OSC)
