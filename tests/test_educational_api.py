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
        'ideal-boson-addition-v1', 'softcore-m8-baseline-recorded-v1'}


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
