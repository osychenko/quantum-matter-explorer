"""Patch 19: time-cut-normalized worm and exact image-integrated winding."""
from dataclasses import replace
import numpy as np
import pytest
from qmc.bosons import SoftCorePairPotential
from qmc.cells import CubicCell, OrthorhombicCell
from qmc.boson_permutations import BosonPermutationPaths, PermutationPathConfig
from qmc.winding import (winding_moments,draw_winding,canonical_free_winding,
                         _image_moments,_link_deltas,_free_one_winding_moment)
from qmc.mobile_worm import (MobileCutWorm,MobileWormConfig,_deltas)
from qmc.worm import canonical_free_occupations


def setup_model(seed=9):
    c=CubicCell(3.6);V=SoftCorePairPotential(c,height=0.0,cutoff=1.5)
    cfg=MobileWormConfig(time_step=.35,n_beads=6,n_steps=380,burn_in=60,thin=3,
                         seed=seed,bridge_lengths=(2,3),collapsed_refresh_probability=.13,
                         refresh_probability=.1,sector_fugacity=1.,gibbs_probability=.9)
    return MobileCutWorm(c,V,2,cfg)


def test_link_deltas_telescope_including_permutation():
    w=setup_model();w.base.permutation[:]=[1,0]
    np.testing.assert_allclose(_link_deltas(w.base.path,w.base.permutation).sum(axis=0),0,atol=1e-13)
    np.testing.assert_allclose(_deltas(w.base.path,w.base.permutation).sum(axis=(0,1)),0,atol=1e-13)


def test_winding_discrete_integer_and_finite():
    w=setup_model();rng=np.random.default_rng(3)
    draw=draw_winding(w.base,rng)
    assert draw.shape==(3,)
    assert np.issubdtype(draw.dtype,np.integer)
    est=winding_moments(w.base)
    assert np.all(np.isfinite(est.winding_squared)) and np.all(est.winding_squared>=0)


def test_winding_image_moments_against_manual_weights():
    delta=np.array([-2.,-.3,.2,1.8]);L=3.6;sigma=.7
    mean,var,images,p=_image_moments(delta,L,sigma)
    np.testing.assert_allclose(p.sum(axis=1),1)
    np.testing.assert_allclose(mean,np.sum(p*images,axis=1))
    np.testing.assert_allclose(var,np.sum(p*(images-mean[:,None])**2,axis=1))


def test_winding_conditional_matches_draws():
    w=setup_model();w.base.permutation[:]=[1,0]
    expected=winding_moments(w.base).winding_squared
    rng=np.random.default_rng(23)
    actual=np.asarray([draw_winding(w.base,rng)**2 for i in range(3000)]).mean(axis=0)
    np.testing.assert_allclose(actual,expected,atol=.12)


def test_winding_translation_invariant():
    w=setup_model();w.base.permutation[:]=[1,0]
    before=winding_moments(w.base)
    w.base.path=w.cell.wrap(w.base.path+np.array([2.5,-1.9,3.7]))
    after=winding_moments(w.base)
    np.testing.assert_allclose(before.winding_squared,after.winding_squared,atol=1e-11)
    np.testing.assert_allclose(before.fraction_axes,after.fraction_axes,atol=1e-11)


def test_winding_particle_relabeling_invariant():
    w=setup_model();w.base.permutation[:]=[1,0]
    before=winding_moments(w.base).winding_squared
    w.base.path=w.base.path[:,::-1,:].copy()
    w.base.permutation[:]=[1,0]
    np.testing.assert_allclose(before,winding_moments(w.base).winding_squared,atol=1e-11)


def test_one_particle_exact_winding_from_winding_sum():
    c=CubicCell(3.6);beta=2.1;D=.5
    ref=canonical_free_winding(c,1,beta,D)
    z,q=_free_one_winding_moment(c.length,beta,D)
    assert z>0
    np.testing.assert_allclose(ref.winding_squared,np.full(3,q/z),rtol=1e-13)


def test_two_particle_twisted_partition_reference():
    c=CubicCell(3.6);beta=2.1;D=.5
    expected=canonical_free_winding(c,2,beta,D).winding_squared[0]
    wave=np.arange(-10,11)
    def z_one(t,theta):
        return np.prod([np.exp(-t*D*((2*np.pi*wave+(theta if axis==0 else 0))/c.length)**2).sum()
                        for axis in range(3)])
    def z_two(theta):
        return (z_one(beta,theta)**2+z_one(2*beta,theta))/2
    h=1e-3
    second=-(np.log(z_two(h))-2*np.log(z_two(0))+np.log(z_two(-h)))/h**2
    assert second==pytest.approx(expected,rel=3e-5,abs=2e-7)


def test_zero_kinetic_or_bad_inputs():
    c=CubicCell(3.6)
    with pytest.raises(ValueError):canonical_free_winding(c,0,1.)
    with pytest.raises(ValueError):canonical_free_winding(c,1,-1.)
    with pytest.raises(ValueError):canonical_free_winding(c,1,1.,0)
    with pytest.raises(TypeError):canonical_free_winding(object(),1,1.)
    with pytest.raises(TypeError):winding_moments(object())
    with pytest.raises(ValueError):_link_deltas(np.zeros((1,2,3)),[0,1])


def test_mobile_probabilities_exact_with_M_factor():
    w=setup_model()
    prob=w.conditional_sector_probabilities()
    M,N=w.config.n_beads,w.n_particles
    assert prob.shape==(1+M*N,)
    np.testing.assert_allclose(prob.sum(),1,atol=1e-14)
    delta=_deltas(w.base.path,w.base.permutation)
    K=np.exp(w.kernel.log_kernel(delta,w.config.time_step))
    ratios=w.config.sector_fugacity/(M*N*w.cell.volume*K)
    np.testing.assert_allclose(prob[1:]/prob[0],ratios.ravel(),rtol=1e-13)


def test_mobile_sector_ratio_normalized():
    w=setup_model()
    s=np.array([.2,.15,-.3]);t=2;i=1
    expected=(np.log(w.config.sector_fugacity)
              -np.log(w.config.n_beads*w.n_particles*w.cell.volume)
              +float(w.kernel.log_kernel(w._delta_at(t,i)+s,w.config.time_step))
              -float(w.kernel.log_kernel(w._delta_at(t,i),w.config.time_step)))
    assert w.log_sector_ratio(i,s,cut=t)==pytest.approx(expected)


def test_cut_gibbs_matches_conditional_probabilities():
    w=setup_model();probs=w.conditional_sector_probabilities()
    counts=np.zeros(len(probs))
    for it in range(3500):
        w.move_sector_gibbs()
        idx=1+w.cut*w.n_particles+w.particle if w.open else 0
        counts[idx]+=1
    np.testing.assert_allclose(counts/len(range(3500)),probs,atol=.035)


def test_open_cut_relocation_preserves_underlying_paths():
    w=setup_model();w.open=True;w.cut=0;w.particle=0
    w.shift=w._heatbath_shift_at(0,0)
    old_path=w.base.path.copy();old_perm=w.base.permutation.copy()
    for _ in range(30):
        assert w.move_relocate_cut()
        assert 0<=w.cut<w.config.n_beads
    np.testing.assert_array_equal(w.base.path,old_path)
    np.testing.assert_array_equal(w.base.permutation,old_perm)
    assert w._counts['relocate_cut']==[30,30]


def test_mobile_collapsed_normalizer_matches_independent_formula():
    w=setup_model();p=w.base.permutation;x=w.base.path
    delta=_deltas(x,p)
    K=np.exp(w.kernel.log_kernel(delta,w.config.time_step))
    F=1+w.config.sector_fugacity*np.sum(1/K)/(w.config.n_beads*w.n_particles*w.cell.volume)
    assert np.exp(w._collapsed_log_normalizer(x,p))==pytest.approx(F,rel=1e-12)


def test_mobile_collapsed_refresh_remains_valid():
    w=setup_model()
    for i in range(12):
        w.move_collapsed_refresh()
        assert 0<=w.conditional_sector_probabilities().sum()<=1.000000000001
        if w.open:
            assert w.cut is not None and w.particle is not None
        else:
            assert w.cut is None


def test_mobile_result_consistent_and_reproducible():
    r1=setup_model(123).run();r2=setup_model(123).run()
    np.testing.assert_array_equal(r1.occupation_fractions,r2.occupation_fractions)
    np.testing.assert_array_equal(r1.winding_squared,r2.winding_squared)
    assert r1.n_closed+r1.n_open==len(np.arange(60,380,3))
    assert r1.time_cut_counts.sum()==r1.n_open
    assert r1.n0_fraction==pytest.approx(r1.occupation_fractions[0])
    assert r1.sector_effective_samples>0


def test_config_rejects_bad_relocation():
    for p in (-.1,1.1,np.nan):
        with pytest.raises(ValueError):replace(setup_model().config,relocate_probability=p).validate()


def test_mobile_invalid_cut_or_particle():
    w=setup_model()
    with pytest.raises(IndexError):w._delta_at(12,1)
    with pytest.raises(IndexError):w._delta_at(0,3)
    with pytest.raises(RuntimeError):w.move_relocate_cut()
    with pytest.raises(ValueError):w.log_sector_ratio(0,[0,0],cut=0)


def test_free_ideal_reference_occupation_bounded():
    w=setup_model()
    exact=canonical_free_occupations(w.cell,2,w.config.beta,w.indices)
    assert 0<exact[0]<1
    assert exact[1]>=0


def test_nonzero_interaction_winding_finite():
    w=setup_model()
    w.base.potential=SoftCorePairPotential(w.cell,height=4.,cutoff=1.5)
    for _ in range(15):w.base.step()
    res=winding_moments(w.base)
    assert np.all(np.isfinite(res.winding_squared))


def test_mobile_cut_count_all_indices_reachable():
    w=setup_model()
    for _ in range(150):
        w.move_sector_gibbs()
        if w.open:w.move_relocate_cut()
    assert w._counts['relocate_cut'][1]>0
