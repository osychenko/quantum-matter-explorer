"""Patch 40 must keep the public numerical scope restricted."""
from qmc.educational_contract import catalog
from qmc.educational_science import science_map, validate_science_map


def test_new_theory_note_is_additive_and_preserves_curated_scope():
    validate_science_map()
    data = science_map()
    topics = {t['id']: t for t in data['theory']}
    assert 'imaginary-time-projection' in topics
    note = topics['imaginary-time-projection']
    assert 'ground state' in note['concept'].lower()
    assert 'finite-temperature' in note['interpretation'].lower()
    assert set(data['profiles']) == {p['id'] for p in catalog()}
    assert len([p for p in data['problems'] if p['status'] == 'research_only']) == 1


def test_theory_metadata_is_returned_as_an_independent_copy():
    data = science_map()
    data['theory'].clear()
    assert any(x['id'] == 'imaginary-time-projection' for x in science_map()['theory'])
