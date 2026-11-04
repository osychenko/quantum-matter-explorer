"""Contract tests ensure research-only configurations cannot enter the HTTP API."""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app)


def test_health_and_catalog():
    health = client.get('/api/v1/health')
    assert health.status_code == 200
    assert health.json()['schema_version'] == '1.0.0'
    cat = client.get('/api/v1/experiments')
    assert cat.status_code == 200
    assert {p['id'] for p in cat.json()['presets']} == {
        'ideal-boson-addition-v1', 'softcore-m8-baseline-recorded-v1',
        'harmonic-oscillator-vmc-v1', 'softcore-structure-vmc-v1'}


def test_analytic_run_matches_patch32_contract():
    from qmc.educational_contract import run_experiment
    payload = {'parameters': {'beta': 2.4, 'cell_length': 2.8}}
    x = client.post('/api/v1/experiments/ideal-boson-addition-v1/run', json=payload)
    assert x.status_code == 200
    assert x.json() == run_experiment('ideal-boson-addition-v1', payload['parameters'])
    assert client.post('/api/v1/experiments/ideal-boson-addition-v1/run', json=payload).json() == x.json()


def test_research_only_and_unexpected_inputs_are_rejected():
    assert client.post('/api/v1/experiments/softcore-m12-exchange/run', json={}).status_code == 422
    assert client.post('/api/v1/experiments/ideal-boson-addition-v1/run', json={
        'parameters': {'n_links': 12}}).status_code == 422
    assert client.post('/api/v1/experiments/ideal-boson-addition-v1/run', json={
        'parameters': {'beta': 7}}).status_code == 422
    assert client.post('/api/v1/experiments/ideal-boson-addition-v1/run', json={
        'parameters': {'beta': True}}).status_code == 422
    assert client.post('/api/v1/experiments/ideal-boson-addition-v1/run', json={
        'parameters': {}, 'arbitrary_jobs': True}).status_code == 422


def test_recorded_label_and_no_knobs():
    r = client.post('/api/v1/experiments/softcore-m8-baseline-recorded-v1/run', json={})
    assert r.status_code == 200
    assert r.json()['execution_mode'] == 'recorded_result'
    assert r.json()['preset_status'] == 'candidate_recording'
    assert client.post('/api/v1/experiments/softcore-m8-baseline-recorded-v1/run',
                       json={'parameters': {'beta': 2.5}}).status_code == 422


def test_bounded_analytic_sweep():
    req = {'beta': 2.4, 'cell_length': 2.8, 'variable': 'beta', 'count': 11}
    res = client.post('/api/v1/experiments/ideal-boson-addition-v1/sweep', json=req)
    assert res.status_code == 200
    points = res.json()['points']
    assert len(points) == 11
    assert points[0]['x'] == .5
    assert points[-1]['x'] == 5.0
    assert points[-1]['green'] > points[0]['green']
    assert client.post('/api/v1/experiments/ideal-boson-addition-v1/sweep',
                       json={**req, 'count': 1000}).status_code == 422
    assert client.post('/api/v1/experiments/ideal-boson-addition-v1/sweep',
                       json={**req, 'variable': 'n_particles'}).status_code == 422


def test_science_map_read_only_and_research_only_cards():
    response = client.get('/api/v1/science-map')
    assert response.status_code == 200
    data = response.json()
    assert data['schema_version'] == '1.0.0'
    assert set(data['profiles']) == {
        'ideal-boson-addition-v1', 'softcore-m8-baseline-recorded-v1',
        'harmonic-oscillator-vmc-v1', 'softcore-structure-vmc-v1'
    }
    assert len(data['problems']) == 5
    for problem in data['problems']:
        if problem['status'] == 'research_only':
            assert problem['preset_id'] is None


def test_research_execution_not_accidentally_exposed_in_new_labs(monkeypatch):
    from qmc import educational_labs
    monkeypatch.setattr(educational_labs, 'harmonic_vmc', lambda a,w: dict(
        mean=1.5, sem=.01, trial_expectation=1.5, exact_ground=1.5,
        seeds=[380010,380011,380012], acceptance=.5,
        n_walkers=128,n_steps=260,burn_in=60,sample_every=2,n_replicas=3))
    r=client.post('/api/v1/experiments/harmonic-oscillator-vmc-v1/run',
                  json={'parameters':{'trial_alpha':1.0,'omega':1.0}})
    assert r.status_code == 200
    assert r.json()['execution_mode'] == 'bounded_live_vmc'
    assert r.json()['observables'][0]['exact_ground_energy'] == 1.5
    for invalid in ({'n_walkers':100000}, {'seed':0}, {'omega':0.01}, {'omega':True}):
        response=client.post('/api/v1/experiments/harmonic-oscillator-vmc-v1/run',
                             json={'parameters':invalid})
        assert response.status_code == 422
    for invalid in ({'potential_height':200}, {'n_particles':1000}, {'jastrow_amplitude':2}):
        response=client.post('/api/v1/experiments/softcore-structure-vmc-v1/run',
                             json={'parameters':invalid})
        assert response.status_code == 422


def test_parallel_vmc_request_gets_429():
    from app.api import _LIVE_VMC_SLOT
    assert _LIVE_VMC_SLOT.acquire(blocking=False)
    try:
        response=client.post('/api/v1/experiments/harmonic-oscillator-vmc-v1/run',json={})
        assert response.status_code==429
    finally:
        _LIVE_VMC_SLOT.release()
