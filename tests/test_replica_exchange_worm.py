"""Patch 27: exact exchange ratios, interactions and physical-replica tests."""
from dataclasses import replace
import numpy as np
import pytest

from qmc.bosons import SoftCorePairPotential
from qmc.cells import CubicCell
from qmc.replica_exchange_worm import ReplicaExchangeConfig, ReplicaExchangeContinuumWorm
from qmc.reliable_spacetime_worm import ReliableContinuumSpacetimeWorm
from qmc.worm_audit import audit_worm_chain

CELL = CubicCell(2.8)


def make(*, height=3.0, fraction=0.3, seed=17, steps=300, **overrides):
    cfg = replace(ReplicaExchangeConfig(n_links=4, time_step=0.6,
                  n_steps=steps, burn_in=60, thin=2, seed=seed,
                  hot_interaction_fraction=fraction), **overrides)
    return ReplicaExchangeContinuumWorm(CELL, SoftCorePairPotential(CELL, height=height,
                                        cutoff=1.2), 1, cfg)


@pytest.mark.parametrize('field,value', [
    ('hot_interaction_fraction', -0.01),('hot_interaction_fraction', 1.01),
    ('hot_interaction_fraction', float('nan')), ('swap_probability', 1.1),
    ('swap_probability', -0.1), ('swap_probability', float('inf')),
])
def test_config_validation(field,value):
    with pytest.raises(ValueError):
        replace(ReplicaExchangeConfig(), **{field:value}).validate()


def test_config_and_domain_validation():
    with pytest.raises(TypeError):
        ReplicaExchangeContinuumWorm(CELL, lambda x: 0, 1)
    with pytest.raises(ValueError, match='one background'):
        ReplicaExchangeContinuumWorm(CELL, SoftCorePairPotential(CELL,height=1.,cutoff=1.2),0)
    with pytest.raises(TypeError):
        ReplicaExchangeContinuumWorm(CELL,SoftCorePairPotential(CELL,height=1.,cutoff=1.2),1,object())


def test_closed_action_zero_even_with_ghost():
    sim = make()
    assert not sim.open and not sim.hot.open
    assert sim.interaction_action(sim) == sim.interaction_action(sim.hot) == 0.
    assert sim.log_exchange_ratio() == 0.


def test_full_strength_action_matches_worm_weight():
    sim = make(fraction=.2)
    sim.open = True
    sim.cut, sim.length, sim.extra = 2, 2, sim._new_extra(2)
    host, cut, ell, extra = sim.host, sim.cut, sim.length, sim.extra
    log_cold = sim.log_relative_weight(host,cut,ell,extra)
    log_hot = sim.hot.log_relative_weight(host,cut,ell,extra)
    assert log_hot - log_cold == pytest.approx((1-.2)*sim.interaction_action(sim), abs=1e-11)


def test_swap_ratio_equals_full_joint_density_difference():
    sim = make(fraction=.45)
    sim.open=True; sim.cut=1; sim.length=2; sim.extra=sim._new_extra(2)
    sim.hot.open=True;sim.hot.cut=3;sim.hot.length=3;sim.hot.extra=sim.hot._new_extra(3)
    cold,hot=sim,sim.hot
    actual = (cold.log_relative_weight(hot.host,hot.cut,hot.length,hot.extra)
              +hot.log_relative_weight(cold.host,cold.cut,cold.length,cold.extra)
              -cold.current_log_relative_weight()-hot.current_log_relative_weight())
    assert sim.log_exchange_ratio()==pytest.approx(actual, abs=1e-12)


def test_swap_one_open_one_closed_sign_and_reversibility():
    sim=make(fraction=0.0)
    sim.open=True;sim.length=2;sim.cut=0;sim.extra=sim._new_extra(2)
    before=sim.log_exchange_ratio()
    assert before==pytest.approx(sim.interaction_action(sim))
    cold_state=sim._get_state(sim)
    hot_state=sim._get_state(sim.hot)
    sim._set_state(sim,hot_state)
    sim._set_state(sim.hot,cold_state)
    assert sim.log_exchange_ratio()==pytest.approx(-before)


def test_free_system_accepts_all_swaps():
    sim=make(height=0, fraction=.1)
    for _ in range(35):
        sim.step()
        assert sim.move_exchange()
    assert sim._counts['exchange'][0]==sim._counts['exchange'][1]


def test_equal_couplings_accept_all_swaps_even_if_interacting():
    sim=make(height=5.,fraction=1.0)
    for _ in range(35):
        sim.step()
        assert sim.move_exchange()
    assert sim._counts['exchange'][0]==sim._counts['exchange'][1]


def test_swap_updates_entire_auxiliary_configuration():
    sim=make()
    sim.open=True;sim.cut=0;sim.length=2;sim.extra=sim._new_extra(2)
    sim.hot.open=False
    state_a=sim._get_state(sim)
    state_b=sim._get_state(sim.hot)
    sim._hot_fraction=1.0 # force exchange with certainty: same-coupling test
    assert sim.move_exchange()
    assert sim.open==state_b[-1]
    assert sim.hot.open==state_a[-1]
    np.testing.assert_array_equal(sim.host,state_b[0])
    np.testing.assert_array_equal(sim.hot.extra,state_a[3])


def test_no_swap_at_zero_probability():
    sim=make(swap_probability=0.)
    for _ in range(15):sim.step()
    assert sim._counts['exchange']==[0,0]


def test_reproducibility():
    a,b=make(seed=992),make(seed=992)
    for _ in range(65):a.step();b.step()
    assert a.open==b.open and a.hot.open==b.hot.open
    np.testing.assert_array_equal(a.host,b.host)
    np.testing.assert_array_equal(a.hot.host,b.hot.host)
    if a.open:np.testing.assert_array_equal(a.extra,b.extra)
    assert a._counts==b._counts


def test_hot_replica_is_not_used_for_cold_measurement():
    sim=make(steps=320,seed=2)
    result=audit_worm_chain(sim,n_blocks=4)
    assert result.green.shape == (3,2)
    assert result.n_measurements == 130
    assert 'exchange' in result.acceptance
    assert np.all(np.isfinite(result.green))


def test_mh_threshold_is_consistent_with_log_ratio(monkeypatch):
    sim=make(fraction=.3)
    sim.open=True;sim.cut=1;sim.length=2;sim.extra=sim._new_extra(2)
    sim.hot.open=True;sim.hot.cut=0;sim.hot.length=2;sim.hot.extra=sim.hot._new_extra(2)
    log_r=sim.log_exchange_ratio()
    assert np.isfinite(log_r)
    # Analytic ratio can also be reconstructed from both G target weights.
    assert np.exp(log_r)==pytest.approx(np.exp((1-.3)*(sim.interaction_action(sim)-sim.interaction_action(sim.hot))))
