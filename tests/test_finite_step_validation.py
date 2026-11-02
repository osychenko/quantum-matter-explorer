import numpy as np
import pytest
from qmc.finite_step_validation import ProjectedPrimitiveReference, compare_replicas, continuum_fit


def test_projected_free_limit():
    from qmc.green_spectral import TwoBosonAdditionReference
    ref=ProjectedPrimitiveReference(height=0,k_max=1)
    exact=TwoBosonAdditionReference(height=0,k_max=1).measure(2.4,[1.2],((0,0,0),)).green[0,0]
    assert np.isclose(ref.green_midpoint(beta=2.4,links=4),exact,rtol=1e-9)
    assert np.isclose(ref.green_midpoint(beta=2.4,links=8),exact,rtol=1e-9)


def test_projected_converges_tau_at_fixed_basis():
    from qmc.green_spectral import TwoBosonAdditionReference
    ref=ProjectedPrimitiveReference(k_max=1)
    exact=TwoBosonAdditionReference(k_max=1).measure(2.4,[1.2],((0,0,0),)).green[0,0]
    a=ref.green_midpoint(beta=2.4,links=4)
    b=ref.green_midpoint(beta=2.4,links=12)
    assert np.isfinite(a) and np.isfinite(b)
    assert abs(b-exact)<abs(a-exact)+1e-7


def test_replica_statistics():
    rows=[dict(method='baseline',M=6,seed=i,green=x,block_sem_4=.12,block_sem_8=.13,block_sem_16=.15)
          for i,x in enumerate([1.7,1.8,1.75])]
    r=compare_replicas(rows,reference=1.69,method='baseline',links=6)
    assert r['replicas']==3 and np.isclose(r['mean_green'],1.75)
    assert r['block_16_over_8_median']>1
    with pytest.raises(ValueError):
        compare_replicas(rows[:1],reference=1.69)


def test_continuum_fit_model():
    rows=[dict(M=m,beta=2.4,method='baseline',mean_green=1.7+.2*(2.4/m)**2,replica_sem=.02)
          for m in (4,6,8,12)]
    r=continuum_fit(rows,method='baseline',power=2)
    assert np.isclose(r['continuum'],1.7)
    assert r['dof']==2
