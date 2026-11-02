"""Patch 26: selection, frozen validation, and sampler compatibility tests."""
from dataclasses import replace
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from qmc.adaptive_worm import (ProposalCandidate, candidate_set, choose_candidate,
                               run_candidate, separated_seed, summarize_runs)
from qmc.bosons import SoftCorePairPotential
from qmc.cells import CubicCell
from qmc.guided_worm import GuidedWormConfig


def _row(name, seed, green=1.7, sem=.15, ess=30., cpu=2.):
    return dict(candidate=name,seed=seed,green=green,jackknife_sem=sem,
                halfblock_sem=sem,ratio_ess=ess,cpu_seconds=cpu)


def test_candidate_family_is_unique_and_contains_control():
    quick=candidate_set(quick=True)
    full=candidate_set()
    assert quick[0].name=='baseline'
    assert len({x.name for x in full})==len(full)
    assert len(full)==len(quick)+1
    for candidate in full:
        assert candidate.validate() is candidate


@pytest.mark.parametrize('candidate',candidate_set())
def test_candidate_configuration_preserves_physics(candidate):
    base=GuidedWormConfig(time_step=.6,n_links=4,sector_fugacity=.79,
                          n_steps=300,burn_in=50)
    got=candidate.configure(base,seed=42,steps=400)
    assert got.time_step==base.time_step and got.n_links==base.n_links
    assert got.sector_fugacity==base.sector_fugacity
    assert got.seed==42 and got.n_steps==400
    assert got.burn_in==100
    assert base.seed != 42  # original unchanged


def test_bad_candidate_rejected():
    for item in (ProposalCandidate(''),ProposalCandidate('invalid,comma'),
                 ProposalCandidate('bad',guided_fraction=1.),
                 ProposalCandidate('bad',guided_move_probability=1.2),
                 ProposalCandidate('bad',multiple_try_probability=.5,
                                   guided_move_probability=.6)):
        with pytest.raises(ValueError):item.validate()
    with pytest.raises(ValueError):ProposalCandidate('ok').configure(GuidedWormConfig(),seed=-1,steps=300)
    with pytest.raises(ValueError):ProposalCandidate('ok').configure(GuidedWormConfig(),seed=1,steps=30)


def test_pilot_and_validation_seed_namespaces_disjoint():
    pilot={separated_seed(stage='pilot',model=im,links=links,candidate=ic,replica=ir)
           for im in range(2) for links in (4,6) for ic in range(4) for ir in range(5)}
    validation={separated_seed(stage='validation',model=im,links=links,candidate=ic,replica=ir)
           for im in range(2) for links in (4,6) for ic in range(4) for ir in range(5)}
    assert not pilot.intersection(validation)
    assert len(pilot)==len(validation)==80
    with pytest.raises(ValueError):separated_seed(stage='adaptive',model=0,links=4,candidate=1,replica=0)
    with pytest.raises(ValueError):separated_seed(stage='pilot',model=0,links=101,candidate=1,replica=0)


def test_summary_score_penalizes_error_and_cpu():
    good=summarize_runs([_row('baseline',3),_row('baseline',4)],reference=1.7,basis_spread=.1)
    bad=summarize_runs([_row('guided',5,green=2.8),_row('guided',6,green=2.8)],reference=1.7,basis_spread=.1)
    slow=summarize_runs([_row('guided',5,cpu=10.),_row('guided',6,cpu=10.)],reference=1.7,basis_spread=.1)
    assert good['pilot_score'] < bad['pilot_score']
    assert good['pilot_score'] < slow['pilot_score']
    assert good['eligible'] and good['diagnostics']=='no_flag'
    assert bad['eligible'] and 'reference_discrepancy' in bad['diagnostics']


def test_low_ess_and_block_instability_block_selection():
    low=summarize_runs([_row('guided',10,ess=2.),_row('guided',11,ess=15.)],
                       reference=1.7,basis_spread=0,min_ess=5.)
    assert not low['eligible']
    block=summarize_runs([dict(_row('guided',10),halfblock_sem=10.),
                          dict(_row('guided',11),halfblock_sem=10.)],
                         reference=1.7,basis_spread=0)
    assert not block['eligible']


def test_fallback_does_not_look_at_validation_data():
    summaries=[dict(candidate='baseline',eligible=False,pilot_score=100.),
               dict(candidate='guided',eligible=False,pilot_score=.0001)]
    name,why=choose_candidate(summaries)
    assert name=='baseline' and why.startswith('fallback')
    summaries[1]['eligible']=True
    assert choose_candidate(summaries)[0]=='guided'
    # The API cannot accept validation data at all: only prespecified pilot summaries.


def test_selection_tiebreak_is_deterministic():
    x=[dict(candidate='baseline',eligible=True,pilot_score=1.),
       dict(candidate='guided',eligible=True,pilot_score=1.)]
    assert choose_candidate(x)[0]=='baseline'
    assert choose_candidate(x[::-1])[0]=='baseline'
    with pytest.raises(ValueError):choose_candidate([x[0],x[0]])
    with pytest.raises(ValueError):choose_candidate([x[1]])


def test_bad_replica_set_rejected():
    a=_row('baseline',123)
    with pytest.raises(ValueError):summarize_runs([a,a],reference=1.0,basis_spread=.1)
    with pytest.raises(ValueError):summarize_runs([a],reference=1.0,basis_spread=.1)
    with pytest.raises(ValueError):summarize_runs([a,dict(_row('baseline',124),cpu_seconds=0.)],reference=1,basis_spread=.1)
    with pytest.raises(ValueError):summarize_runs([a,_row('baseline',124)],reference=1,basis_spread=-1)


def test_short_real_run_matches_unchanged_target_and_audit():
    cell=CubicCell(2.8)
    pot=SoftCorePairPotential(cell,height=0.,cutoff=1.2)
    base=GuidedWormConfig(n_links=4,time_step=.6,thin=3,seed=2,
                          n_steps=300,burn_in=60)
    baseline=candidate_set(quick=True)[0]
    result=run_candidate(cell,pot,base,baseline,seed=101,steps=260,
                         spectral_reference=1.986,n_blocks=4)
    assert np.isfinite(result['green'])
    assert result['cpu_seconds']>0 and 0<result['ratio_ess']<=result['steps']
    assert result['spectral_reference']==1.986
    assert 0<=result['global_refresh_acceptance']<=1


def test_quick_example_output_and_holdout_contract(tmp_path):
    from examples.bosons.run_adaptive_worm import run_experiment
    selections,validation=run_experiment(tmp_path,quick=True,
                    pilot_steps=320,validation_steps=320,
                    pilot_replicas=2,validation_replicas=2,min_ess=1.)
    assert len(selections)==1
    assert len(validation)==3
    selected=selections[0]['selected']
    assert selected in [c.name for c in candidate_set(quick=True)]
    files=('pilot_runs.csv','pilot_summary.csv','selected.csv',
           'validation_runs.csv','validation_summary.csv','spectral_reference.csv',
           'protocol.json','validation_accuracy.png','validation_efficiency.png')
    for filename in files:
        assert (tmp_path/filename).is_file() and (tmp_path/filename).stat().st_size>20
    pilot=list(csv.DictReader((tmp_path/'pilot_runs.csv').open()))
    held=list(csv.DictReader((tmp_path/'validation_runs.csv').open()))
    assert len(pilot)==len(held)==6
    assert not set(r['seed'] for r in pilot).intersection(r['seed'] for r in held)
    assert len({r['candidate'] for r in held})==3  # all arms, not just selected
    assert all(r['pilot_selected']==selected for r in held)
    manifest=json.loads((tmp_path/'protocol.json').read_text())
    assert manifest['selected'][0]['selected']==selected
    assert 'no online chain tuning' in manifest['methodology']
