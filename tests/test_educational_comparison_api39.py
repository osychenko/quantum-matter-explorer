"""Patch 39 API: bounded comparisons and explicit parameter errors."""
from threading import BoundedSemaphore
import pytest
pytest.importorskip('fastapi')
pytest.importorskip('httpx')
from fastapi.testclient import TestClient
import app.api as module

client = TestClient(module.app)


def test_comparison_catalog_and_exact_reference_route():
    response=client.get('/api/v1/comparisons')
    assert response.status_code==200
    assert len(response.json()['comparisons'])==4
    check=client.post('/api/v1/comparisons/worm-vs-spectral-v1/run',json={'parameters':{}})
    assert check.status_code==200
    assert check.json()['analysis']['combined_sem'] is None


def test_comparison_rejects_arbitrary_model_and_parameters():
    assert client.post('/api/v1/comparisons/arbitrary-vmc/run',json={'parameters':{}}).status_code==422
    assert client.post('/api/v1/comparisons/ideal-vs-interacting-v1/run',json={'parameters':{'bad':2}}).status_code==422
    assert client.post('/api/v1/comparisons/ideal-vs-interacting-v1/run',json={'parameters':{}, 'extra':'bad'}).status_code==422
    assert client.post('/api/v1/comparisons/oscillator-trial-choice-v1/run',json={'parameters':{'omega':500}}).status_code==422


def test_full_pair_occupies_single_live_slot(monkeypatch):
    lock=BoundedSemaphore(1)
    monkeypatch.setattr(module,'_LIVE_VMC_SLOT',lock)
    lock.acquire()
    try:
        busy=client.post('/api/v1/comparisons/oscillator-trial-choice-v1/run',json={'parameters':{}})
        assert busy.status_code==429
        recorded=client.post('/api/v1/comparisons/worm-vs-spectral-v1/run',json={'parameters':{}})
        assert recorded.status_code==200
    finally:
        lock.release()


def test_lock_is_released_after_validation_error(monkeypatch):
    lock=BoundedSemaphore(1)
    monkeypatch.setattr(module,'_LIVE_VMC_SLOT',lock)
    result=client.post('/api/v1/comparisons/oscillator-trial-choice-v1/run',
                       json={'parameters':{'alpha_left':-8}})
    assert result.status_code==422
    assert lock.acquire(blocking=False)
    lock.release()
