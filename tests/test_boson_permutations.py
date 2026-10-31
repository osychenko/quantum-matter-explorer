"""Patch 17: exact free Bose exchange reference, detailed balance, CPU tuning."""

from dataclasses import replace
from math import isfinite

import numpy as np
import pytest

from qmc.bosons import SoftCoreJastrowTrial, SoftCorePairPotential
from qmc.cells import CubicCell
from qmc.boson_permutations import (
    BosonPermutationPaths, PermutationPathConfig, free_one_particle_partition,
    free_two_boson_exchange_probability, permutation_cycles,
)
from qmc.path_efficiency import (
    EfficiencyConfig, benchmark_chain, compare_bridge_probabilities,
)


class ZeroPotential:
    def __call__(self, x):
        return np.zeros(len(x))


def make(*, particles=2, seed=42, interacting=False, **cfg_args):
    cell = CubicCell(2.4)
    pot = SoftCorePairPotential(cell, height=2, cutoff=1) if interacting else ZeroPotential()
    cfg = PermutationPathConfig(n_steps=150, burn_in=30, thin=3, seed=seed, **cfg_args)
    return BosonPermutationPaths(cell, pot, particles, cfg)


def test_cycle_decomposition_and_invalid_permutation():
    assert permutation_cycles([0, 1, 2]) == (1, 1, 1)
    assert permutation_cycles(np.array([1, 2, 0])) == (3,)
    assert permutation_cycles([1, 0, 3, 2, 4]) == (2, 2, 1)
    for invalid in ([1, 1], [-1, 0], [0, 3], [[0, 1]], [], [0., 1.]):
        with pytest.raises(ValueError):
            permutation_cycles(invalid)


def test_analytic_two_boson_exchange_limits():
    cell = CubicCell(2.4)
    low = free_two_boson_exchange_probability(cell, 1e-2)
    high = free_two_boson_exchange_probability(cell, 15.)
    assert low < .01
    assert high == pytest.approx(.5, abs=1e-7)
    assert free_one_particle_partition(cell, 1e-4) > free_one_particle_partition(cell, 1.0) > 1.0


def test_poisson_dual_partition_matches_reciprocal_series():
    cell=CubicCell(2.4)
    for beta in (.01, .1, 1., 3.):
        partition = free_one_particle_partition(cell,beta)
        a=beta*.5*(2*np.pi/cell.length)**2
        k=np.arange(-250,251)
        reference=(np.exp(-a*k*k).sum())**3
        assert partition==pytest.approx(reference,rel=2e-14)


@pytest.mark.parametrize('change', [
    {'n_beads':3}, {'n_beads':2}, {'n_steps':10}, {'thin':0},
    {'burn_in':150}, {'swap_probability':1.1}, {'swap_probability':np.nan},
    {'bridge_probability':.9}, {'refresh_probability':.9},
    {'bridge_lengths':(8,)}, {'bridge_lengths':(1,)},
    {'bridge_lengths':(2,2)}, {'seed':-1},
])
def test_invalid_config(change):
    with pytest.raises(ValueError):
        replace(PermutationPathConfig(n_steps=150,burn_in=30,thin=3),**change).validate()


def test_n_too_large():
    with pytest.raises(ValueError,match='N<=5'):
        make(particles=6)


def test_path_target_is_exact_product_of_links_and_action():
    sampler=make(interacting=True)
    sampler.permutation[:]=[1,0]
    x=sampler.path
    tau=sampler.config.time_step
    free=float(sampler.kernel.log_kernel(x[1:]-x[:-1],tau).sum()
               +sampler.kernel.log_kernel(x[0,[1,0]]-x[-1],tau).sum())
    expected=free-tau*float(np.sum(sampler.potential(x)))
    assert sampler.log_weight()==pytest.approx(expected,abs=1e-12)


def test_exchange_swap_ratio_is_full_target_difference():
    sampler=make(interacting=True)
    before=sampler.log_weight()
    perm=sampler.permutation.copy();perm[[0,1]]=perm[[1,0]]
    expected=sampler.log_weight(permutation=perm)-before
    captured=[]
    sampler._accept=lambda kind,path,p,log_ratio:captured.append((kind,log_ratio)) or True
    sampler.move_swap(0,1)
    assert captured[0][0]=='swap'
    assert captured[0][1]==pytest.approx(expected,abs=1e-12)


def test_bridge_metropolis_ratio_has_no_kinetic_terms():
    sampler=make(interacting=True)
    x=sampler.path.copy()
    captured=[]
    sampler._accept=lambda kind,path,p,log_ratio:captured.append((path.copy(),log_ratio)) or True
    sampler.move_bridge(length=4,start=1,particle=0)
    proposed,ratio=captured[0]
    assert np.array_equal(proposed[1,0],x[1,0])
    assert np.array_equal(proposed[5,0],x[5,0])
    np.testing.assert_allclose(proposed[:,1],x[:,1])
    # Bridge proposal is the exact conditional FREE action; its reverse/forward
    # proposal ratio cancels the free kinetic term in the target ratio.
    tau=sampler.config.time_step
    kin_delta=float(sampler.kernel.log_kernel(proposed[1:]-proposed[:-1],tau).sum()
                    -sampler.kernel.log_kernel(x[1:]-x[:-1],tau).sum())
    assert ratio==pytest.approx(sampler.log_weight(proposed)-sampler.log_weight(x)-kin_delta,abs=1e-10)


def test_bead_metropolis_ratio_matches_full_weight_with_exchange():
    sampler=make(interacting=True)
    sampler.permutation[:]=[1,0]
    initial=sampler.path.copy()
    for j in (0,1,sampler.config.n_beads-1):
        captured=[]
        sampler._accept=lambda kind,path,p,log_ratio:captured.append((path.copy(),log_ratio)) or True
        sampler.move_bead(bead=j,particle=1)
        proposed,ratio=captured[0]
        assert ratio==pytest.approx(sampler.log_weight(proposed)-sampler.log_weight(initial),abs=1e-11)


def test_global_relabeling_conjugates_permutation():
    sampler=make(particles=3,interacting=True)
    sampler.permutation[:]=[1,2,0]
    order=np.array([2,0,1])
    inverse=np.argsort(order)
    perm_new=inverse[sampler.permutation[order]]
    assert sampler.log_weight(sampler.path[:,order],perm_new)==pytest.approx(sampler.log_weight())


def test_free_refresh_exact_acceptance_and_sector_weights():
    sampler=make()
    assert sampler._free_probabilities.sum()==pytest.approx(1.)
    observed=[]
    for _ in range(1400):
        assert sampler.move_refresh() is True
        observed.append(float(permutation_cycles(sampler.permutation)[0]>1))
    assert sampler._counts['refresh']==[1400,1400]
    expected=free_two_boson_exchange_probability(sampler.cell,sampler.config.beta)
    assert abs(np.mean(observed)-expected)<.045


def test_free_updates_staging_and_translation_accept_exactly():
    sampler=make()
    for _ in range(15):
        assert sampler.move_bridge(length=4) is True
        assert sampler.move_translate() is True
    assert sampler._counts['bridge']==[15,15]
    assert sampler._counts['translate']==[15,15]


def test_free_refresh_interacting_ratio_is_only_action_difference():
    sampler=make(interacting=True)
    old=sampler.path.copy()
    capture=[]
    sampler._accept=lambda kind,path,p,delta:capture.append((path.copy(),delta)) or True
    sampler.move_refresh()
    new,ratio=capture[0]
    expected=-sampler.config.time_step*(np.sum(sampler.potential(new))-np.sum(sampler.potential(old)))
    assert ratio==pytest.approx(expected,abs=1e-12)


def test_run_reproducibility_and_sample_shape():
    a=make(seed=5).run()
    b=make(seed=5).run()
    np.testing.assert_array_equal(a.exchange_indicator,b.exchange_indicator)
    np.testing.assert_array_equal(a.final_path,b.final_path)
    assert len(a.exchange_indicator)==40
    assert a.longest_cycle_fraction.min()>=.5
    assert a.longest_cycle_fraction.max()<=1.


def test_small_interacting_run_and_exchange_cycles():
    a=make(interacting=True).run()
    assert np.all(np.isfinite(a.midpoint_potential))
    assert np.all(a.midpoint_potential>=0)
    assert 0<=float(a.exchange_indicator.mean())<=1
    assert all(0<=v<=1 for v in a.acceptance.values() if np.isfinite(v))


@pytest.mark.parametrize('change', [
    {'bridge_probabilities':(.1,.5)}, {'bridge_probabilities':(0.,0.)},
    {'pilot_replicas':1}, {'validation_steps':1}, {'n_links':5},
    {'bridge_lengths':(8,)}, {'burn_in':1000},
    {'refresh_probability':.9},
])
def test_efficiency_config_rejects_bad_input(change):
    with pytest.raises(ValueError):
        replace(EfficiencyConfig(),**change).validate()


def efficiency_setup():
    cell=CubicCell(3.5)
    potential=SoftCorePairPotential(cell,height=3.0,cutoff=1.4)
    trial=SoftCoreJastrowTrial(cell,amplitude=.5,cutoff=1.4)
    cfg=EfficiencyConfig(bridge_probabilities=(0.,.6),pilot_steps=65,
                         validation_steps=80,burn_in=10,thin=3,
                         pilot_replicas=2,validation_replicas=2)
    return cell,potential,trial,cfg


def test_benchmark_chain_output_and_cpu_diagnostics():
    c,p,t,cfg=efficiency_setup()
    row=benchmark_chain(c,p,t,n_particles=2,cfg=cfg,
                        bridge_probability=.6,seed=111,n_steps=65,
                        phase='pilot',replica=0)
    assert row['n_samples']==19
    assert row['cpu_seconds']>0
    assert isfinite(row['effective_samples_per_cpu_second'])
    assert 0<row['effective_samples']<=row['n_samples']


def test_mixture_selection_uses_independent_holdout_seeds():
    c,p,t,cfg=efficiency_setup()
    pilot,valid,winner=compare_bridge_probabilities(c,p,t,n_particles=2,cfg=cfg)
    assert len(pilot)==4
    assert len(valid) in (2,4)  # if baseline wins, it is validated once
    assert winner in (0.,.6)
    assert set(x['seed'] for x in pilot).isdisjoint(x['seed'] for x in valid)
    assert all(x['phase']=='validation' for x in valid)


def test_quick_example_writes_distinct_ensemble_outputs(tmp_path):
    from examples.bosons.run_boson_permutations import run_experiment
    reps,summaries,pilots,valid,winner=run_experiment(tmp_path,quick=True)
    assert len(summaries)==2
    assert len(reps)==4
    assert len(pilots)==6
    assert winner in (0.,.4,.8)
    for name in ('permutation_replicas.csv','permutation_summary.csv',
                 'efficiency_pilot.csv','efficiency_validation.csv','protocol.json',
                 'cpu_efficiency.png','exchange_probability.png'):
        assert (tmp_path/name).stat().st_size>100
