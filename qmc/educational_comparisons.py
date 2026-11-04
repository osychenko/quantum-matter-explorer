"""Patch 39: curated scientific comparisons for the educational application.

Comparisons are server-designed *questions*, not permission to execute arbitrary
research samplers. Each arm preserves its original experiment result, including
method, parameter settings, scientific status and statistical qualifications.

No independent-difference SEM or significance is inferred for two VMC runs:
Patch 38 reuses seed IDs and does not record per-seed covariance. A difference
between two such runs is useful for exploration but not a formal hypothesis test.
"""
from __future__ import annotations

from hashlib import sha256
import json
from math import isfinite
from typing import Any

from .educational_contract import ENGINE_RELEASE, SCHEMA_VERSION, run_experiment, validate_result

COMPARISON_SCHEMA_VERSION = '1.0.0'
OSC = 'oscillator-trial-choice-v1'
STRUCT = 'softcore-jastrow-choice-v1'
IDEAL = 'ideal-vs-interacting-v1'
WORM = 'worm-vs-spectral-v1'

_CATALOG = (
    dict(id=OSC, title='Same oscillator, two trial wavefunctions',
         question='How does changing a Gaussian trial state change a VMC energy estimate?',
         comparison_kind='controlled_variational_choice',
         description='Harmonic trap held fixed; two trial widths and their analytical variational expectations.',
         parameters=[dict(id='alpha_left', label='Trial width A (α)', min=0.5, max=1.8, step=0.05, default=0.7),
                     dict(id='alpha_right', label='Trial width B (α)', min=0.5, max=1.8, step=0.05, default=1.3),
                     dict(id='omega', label='Common trap frequency (ω)', min=0.7, max=1.5, step=0.05, default=1.0)],
         live=True),
    dict(id=STRUCT, title='Same bosons, two variational structures',
         question='How do pair and density correlations change when we change the trial state?',
         comparison_kind='controlled_variational_choice',
         description='Hamiltonian and sampling settings held fixed; symmetric Jastrow amplitude changes.',
         parameters=[dict(id='amplitude_left', label='Trial strength A (a)', min=0.0, max=1.2, step=0.1, default=0.0),
                     dict(id='amplitude_right', label='Trial strength B (a)', min=0.0, max=1.2, step=0.1, default=1.0)],
         live=True),
    dict(id=IDEAL, title='Free versus interacting: an illustrative contrast',
         question='Which differences can we observe when the Hamiltonian and solver both change?',
         comparison_kind='confounded_cross_model_contrast',
         description='Same beta, box, zero-momentum addition observable; different interactions and evaluation methods.',
         parameters=[], live=False),
    dict(id=WORM, title='Worm Monte Carlo versus spectral benchmark',
         question='Does a measured interacting Green function agree with an independent finite-basis calculation?',
         comparison_kind='cross_method_reference_check',
         description='Same nominal interacting problem and observable; different numerical approximations.',
         parameters=[], live=False),
)
_CATALOG_MAP = {x['id']: x for x in _CATALOG}


def comparison_catalog() -> list[dict[str, Any]]:
    """Return a defensive copy, so callers cannot mutate the policy registry."""
    return json.loads(json.dumps(_CATALOG))


def comparison_requires_live(comparison_id: str) -> bool:
    return bool(_CATALOG_MAP.get(comparison_id, {}).get('live', False))


def _validated_parameters(comparison_id: str, supplied: dict[str, Any] | None) -> dict[str, float]:
    if comparison_id not in _CATALOG_MAP:
        raise ValueError('Unknown or research-only comparison: '+str(comparison_id))
    if supplied is None:
        supplied = {}
    if not isinstance(supplied, dict):
        raise ValueError('comparison parameters must be an object')
    fields = {x['id']: x for x in _CATALOG_MAP[comparison_id]['parameters']}
    extra = set(supplied)-set(fields)
    if extra:
        raise ValueError('Unsupported comparison parameter(s): '+', '.join(sorted(extra)))
    pars: dict[str, float] = {}
    for name, field in fields.items():
        value = supplied.get(name, field['default'])
        if (isinstance(value, bool) or not isinstance(value, (float, int))
                or not isfinite(value) or not field['min'] <= value <= field['max']):
            raise ValueError(f"{name} must be finite in [{field['min']}, {field['max']}]")
        pars[name] = float(value)
    if comparison_id == OSC and pars['alpha_left'] == pars['alpha_right']:
        raise ValueError('Choose two different trial widths to compare')
    if comparison_id == STRUCT and pars['amplitude_left'] == pars['amplitude_right']:
        raise ValueError('Choose two different Jastrow trial strengths to compare')
    return pars


def _run(preset: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    result = run_experiment(preset, params)
    validate_result(result)
    return result


def _arm(label: str, result: dict[str, Any], *, observable: dict[str, Any] | None = None,
         kind: str | None = None) -> dict[str, Any]:
    # Keep actual experiment result (incl. full parameters and full curves).
    return dict(label=label, evidence_kind=kind or result['execution_mode'],
                experiment_id=result['experiment_id'], experiment=result,
                primary_observable=observable or result['observables'][0])


def _reference_arm(label: str, result: dict[str, Any], value: float) -> dict[str, Any]:
    """The reference is a field of a recorded result, not a new experiment."""
    return dict(label=label, evidence_kind='finite_basis_reference_from_recorded_experiment',
                experiment_id=result['experiment_id'], experiment=result,
                primary_observable=dict(id='green_addition_midpoint_k0', value=value,
                                        uncertainty=None, units='dimensionless',
                                        uncertainty_kind='not_reported_reference_truncation'))


def _base(comparison_id: str, p: dict[str, float], arms: list[dict[str, Any]],
          *, held_fixed: list[str], changed: list[str], confounders: list[str],
          conclusions: list[str], cannot_conclude: list[str],
          relationship: str, reference: dict[str, Any] | None = None) -> dict[str, Any]:
    if len(arms) != 2:
        raise ValueError('A comparison requires exactly two arms')
    a, b = (arm['primary_observable'] for arm in arms)
    if a['id'] != b['id'] or a['units'] != b['units']:
        raise ValueError('Incompatible observables cannot be directly compared')
    delta = float(b['value'] - a['value'])
    if not isfinite(delta):
        raise ValueError('Non-finite comparison difference')
    # Deliberately do not derive sqrt(SEM_A²+SEM_B²): VMC runs use shared
    # seed IDs, and unknown covariance makes that number scientifically invalid.
    inference = ('Difference B − A is descriptive; covariance between the estimates is not '
                 'available, so no combined uncertainty or significance test is supplied.')
    if comparison_id == WORM:
        inference = ('Reference truncation and QMC time-step bias are not in the quoted SEM; '
                     'the numerical difference is a consistency screen, not proof of convergence.')
    if comparison_id == IDEAL:
        inference = ('The two Hamiltonians and computational approaches differ. This delta is '
                     'descriptive, NOT an isolated causal interaction effect.')
    payload = dict(comparison_schema_version=COMPARISON_SCHEMA_VERSION,
                   experiment_schema_version=SCHEMA_VERSION, engine_release=ENGINE_RELEASE,
                   comparison_id=comparison_id, title=_CATALOG_MAP[comparison_id]['title'],
                   question=_CATALOG_MAP[comparison_id]['question'],
                   comparison_kind=_CATALOG_MAP[comparison_id]['comparison_kind'],
                   requested_parameters=p, arms=arms,
                   controls=dict(held_fixed=held_fixed, changed=changed, confounders=confounders),
                   analysis=dict(observable_id=a['id'], units=a['units'],
                                 difference_b_minus_a=delta, combined_sem=None,
                                 difference_z=None, inference=inference,
                                 reference=reference),
                   interpretation=dict(relationship=relationship,
                                       supported_conclusions=conclusions,
                                       not_supported=cannot_conclude))
    digest = dict(comparison_id=comparison_id, parameters=p,
                  arm_fingerprints=[v['experiment']['fingerprint_sha256'] for v in arms],
                  evidence_kinds=[v['evidence_kind'] for v in arms])
    payload['fingerprint_sha256'] = sha256(json.dumps(digest, sort_keys=True,
                                                       separators=(',', ':')).encode()).hexdigest()
    return payload


def _same_setup(left: dict[str, Any], right: dict[str, Any], keys: tuple[str,...]) -> None:
    """Fail closed if a later sampler update silently changes a control."""
    for key in keys:
        if (key not in left['parameters'] or key not in right['parameters']
                or left['parameters'][key] != right['parameters'][key]):
            raise ValueError(f'Comparison control {key} was missing or not held fixed')
    for key in ('method', 'replicas', 'seeds'):
        if (key not in left['provenance'] or key not in right['provenance']
                or left['provenance'][key] != right['provenance'][key]):
            raise ValueError(f'Comparison sampling evidence {key} is missing or does not match')


def _same_structural_grid(left: dict[str, Any], right: dict[str, Any]) -> None:
    x = {c['id']: c for c in left.get('distributions', [])}
    y = {c['id']: c for c in right.get('distributions', [])}
    for key in ('pair_correlation', 'structure_factor'):
        if key not in x or key not in y:
            raise ValueError('Missing comparable structure distribution: ' + key)
        if (key == 'structure_factor' and
                (not x[key].get('indices') or not y[key].get('indices'))):
            raise ValueError('Structure-factor reciprocal directions are missing')
        if x[key]['x'] != y[key]['x'] or x[key].get('indices') != y[key].get('indices'):
            raise ValueError('Structure distributions use nonmatching grids or reciprocal directions')

def run_comparison(comparison_id: str, parameters: dict[str, Any] | None = None) -> dict[str, Any]:
    """Execute one of four strictly curated contrasts, never arbitrary pairings."""
    p = _validated_parameters(comparison_id, parameters)
    if comparison_id == OSC:
        omega = p['omega']
        left = _run('harmonic-oscillator-vmc-v1', dict(trial_alpha=p['alpha_left'], omega=omega))
        right = _run('harmonic-oscillator-vmc-v1', dict(trial_alpha=p['alpha_right'], omega=omega))
        _same_setup(left, right, ('omega', 'kinetic_coefficient', 'dimensions',
                                 'n_walkers', 'n_steps', 'burn_in', 'sample_every'))
        return _base(comparison_id, p, [_arm('Trial A', left), _arm('Trial B', right)],
            held_fixed=['Same 3D harmonic Hamiltonian, ω='+str(omega),
                        'Same kinetic coefficient D=0.5 and numerical sampler settings'],
            changed=['Only Gaussian trial width α: '+str(p['alpha_left'])+' → '+str(p['alpha_right'])],
            confounders=['VMC estimates use the same seed identifiers, so cross-arm sampling covariance is unknown'],
            conclusions=['The exact Gaussian trial expectation varies with α; the ground energy stays fixed.',
                         'The Monte Carlo estimates test the known variational expectations with sampling noise.'],
            cannot_conclude=['A difference between noisy VMC estimates alone does not prove one algorithm is better.',
                             'No exact excited-state or dynamical information is computed.'],
            relationship='Controlled change of variational wavefunction at fixed Hamiltonian.')
    if comparison_id == STRUCT:
        left = _run('softcore-structure-vmc-v1', dict(jastrow_amplitude=p['amplitude_left']))
        right = _run('softcore-structure-vmc-v1', dict(jastrow_amplitude=p['amplitude_right']))
        _same_setup(left, right, ('n_particles', 'cell_length', 'kinetic_coefficient',
                                 'cutoff', 'potential_height', 'n_walkers',
                                 'n_steps', 'burn_in', 'sample_every'))
        _same_structural_grid(left, right)
        return _base(comparison_id, p, [_arm('Trial A', left), _arm('Trial B', right)],
            held_fixed=['Same six-boson soft-core Hamiltonian: N=6, L=5, V₀=3, cutoff=1.2',
                        'Same VMC algorithm, numerical settings, and selected reciprocal indices'],
            changed=['Only Jastrow TRIAL amplitude a: '+str(p['amplitude_left'])+' → '+str(p['amplitude_right'])],
            confounders=['Shared seed identifiers; no joint covariance or simultaneous error bands',
                         'Three short VMC replicas yield exploratory structural estimates'],
            conclusions=['Differences in g(r) and directional S(k) describe the trial-state distributions.',
                         'These two simulations use the same physical repulsive Hamiltonian.'],
            cannot_conclude=['Trial-state differences are NOT a measurement of changing interaction strength.',
                             'No liquid-solid phase transition or thermodynamic limit is established.'],
            relationship='Controlled trial-state comparison, not a potential-strength scan.')
    if comparison_id == IDEAL:
        recorded = _run('softcore-m8-baseline-recorded-v1')
        pars = recorded['parameters']
        ideal = _run('ideal-boson-addition-v1',
                     dict(beta=pars['beta'], cell_length=pars['cell_length']))
        return _base(comparison_id, p, [_arm('Analytical ideal gas', ideal),
                                        _arm('Recorded interacting estimate', recorded)],
            held_fixed=['Same β=2.4, L=2.8, zero-momentum midpoint addition observable, N=1→2'],
            changed=['Hamiltonian: ideal gas → repulsive soft-core interaction',
                     'Method: truncated momentum sum → M=8 baseline worm QMC'],
            confounders=['Physical interaction and numerical approximations BOTH change',
                         'The interacting result is recorded, not recomputed live'],
            conclusions=['The two curated outputs are numerically different under the chosen definitions.'],
            cannot_conclude=['The difference cannot be attributed solely to interaction strength.',
                             'It is not a continuum-limit or phase-transition result.'],
            relationship='Matched observable and several parameters, but confounded cross-model contrast.')
    recorded = _run('softcore-m8-baseline-recorded-v1')
    obs = recorded['observables'][0]
    reference = float(obs['spectral_reference_k2'])
    return _base(comparison_id, p, [_arm('Recorded worm Monte Carlo', recorded),
                                    _reference_arm('Spectral K=2 reference', recorded, reference)],
        held_fixed=['Same nominal two-sector interacting Hamiltonian and zero-mode midpoint observable',
                    'Same β=2.4 and box length L=2.8'],
        changed=['Numerical approximation: finite-M=8 worm estimate versus finite-basis K=2 Galerkin reference'],
        confounders=['Finite time-step and finite-basis systematic biases are not quantified by the replica SEM',
                     'Spectral reference is a field of the recorded benchmark, not a new sampled run'],
        conclusions=['The measured estimator and independent spectral calculation can be screened for consistency.'],
        cannot_conclude=['Agreement within statistical error does not establish continuum convergence or exactness.'],
        relationship='Cross-method numerical reference check with systematic errors unresolved.',
        reference=dict(kind='finite_basis_k2', value=reference,
                       source='Recorded benchmark spectral_reference_k2',
                       uncertainty='Not provided for basis truncation'))
