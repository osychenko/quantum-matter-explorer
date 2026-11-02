"""Finite-step *projected* primitive transfer and conservative replica analysis.

The transfer P exp(-tau K/2) P exp(-tau V) P exp(-tau K/2) P
uses a truncated symmetric two-boson momentum basis and FFT quadrature.
Its projection at every slice differs from an untruncated coordinate-space
primitive path integral.  Compare k_max=1/2; do NOT label this exact tau reference.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import sqrt
import numpy as np

from .green_spectral import TwoBosonAdditionReference


def _matrix_from_fourier(ref, pairs, coefficients):
    first = np.asarray([ref.modes[p] for p, q in pairs], int)
    second = np.asarray([ref.modes[q] for p, q in pairs], int)
    repeat = np.asarray([p == q for p, q in pairs], float)
    d1 = first[:, None, :] - first[None, :, :]
    d2 = first[:, None, :] - second[None, :, :]
    v1 = coefficients[tuple(np.mod(d1[:, :, axis], ref.n_grid) for axis in range(3))]
    v2 = coefficients[tuple(np.mod(d2[:, :, axis], ref.n_grid) for axis in range(3))]
    matrix = (v1 + v2) / np.sqrt((1 + repeat[:, None]) * (1 + repeat[None, :]))
    return (matrix + matrix.conj().T) * .5


class ProjectedPrimitiveReference:
    """Independent finite-step transfer, with explicit basis truncation caveat."""

    def __init__(self, *, cell_length=2.8, kinetic_coefficient=.5, height=3.,
                 cutoff=1.2, k_max=1, n_grid=16):
        self.ref = TwoBosonAdditionReference(cell_length=cell_length,
            kinetic_coefficient=kinetic_coefficient, height=height,
            cutoff=cutoff, k_max=k_max, n_grid=n_grid)
        self._cache = {}

    @lru_cache(maxsize=8)
    def _fourier_boltzmann(self, tau):
        ref = self.ref
        n, L = ref.n_grid, ref.length
        c = np.arange(n) * (L/n)
        xyz = np.meshgrid(c, c, c, indexing='ij')
        radius2 = sum((x - L*np.rint(x/L))**2 for x in xyz)
        v = ref.height * np.maximum(1 - radius2/ref.cutoff**2, 0)**3
        return np.fft.fftn(np.exp(-tau*v)) / n**3

    def _sector(self, total, tau):
        key = (tuple(int(i) for i in total), tau)
        if key in self._cache:
            return self._cache[key]
        ref = self.ref
        pairs, lookup, _, _ = ref._block(total)
        u = _matrix_from_fourier(ref, pairs, self._fourier_boltzmann(tau))
        kinetic = np.array([ref.eps[p] + ref.eps[q] for p, q in pairs])
        half = np.exp(-tau * kinetic / 2)
        transfer = half[:, None] * u * half[None, :]
        eigenvalues, eigenvectors = np.linalg.eigh(transfer)
        if eigenvalues.min() < -1e-9:
            raise FloatingPointError('projected primitive transfer not positive')
        eigenvalues = np.maximum(eigenvalues, 0)
        self._cache[key] = (lookup, eigenvalues, eigenvectors)
        return self._cache[key]

    def green_midpoint(self, *, beta, links, mode=(0,0,0)):
        if type(links) is not int or links < 4 or links % 2:
            raise ValueError('links must be an even integer >=4')
        if not np.isfinite(beta) or beta <= 0:
            raise ValueError('beta must be finite and >0')
        ref = self.ref
        m = tuple(mode)
        if m not in ref.mode_lookup:
            raise ValueError('mode not present in finite basis')
        im = ref.mode_lookup[m]
        tau = beta/links
        ell = links//2
        Z = np.exp(-beta*ref.eps).sum()
        answer = 0.
        for p in range(len(ref.modes)):
            lookup, values, vectors = self._sector(ref.modes[p]+ref.modes[im], tau)
            pair = tuple(sorted((p,im)))
            row = lookup[pair]
            strength = 2. if p == im else 1.
            answer += np.exp(-(beta-ell*tau)*ref.eps[p]) * strength * np.sum(
                np.abs(vectors[row])**2 * values**ell)
        result = float(np.real(answer/Z))
        if not np.isfinite(result) or result <= 0:
            raise FloatingPointError('invalid projected primitive Green function')
        return result


def compare_replicas(rows, *, reference, method=None, links=None):
    """Replicate-based SEM and blocking diagnostics, not a convergence certificate."""
    data = [r for r in rows if (method is None or r['method'] == method)
                                  and (links is None or int(r['M']) == links)]
    if len(data) < 2 or len({int(r['seed']) for r in data}) != len(data):
        raise ValueError('require at least two independently seeded replicas')
    values = np.array([float(r['green']) for r in data])
    errors = np.array([[float(r[f'block_sem_{k}']) for k in (4,8,16)] for r in data])
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(errors)) or np.any(errors <= 0):
        raise ValueError('nonfinite or nonpositive data')
    mean = float(values.mean())
    sem = float(values.std(ddof=1)/sqrt(len(data)))
    internal = float(np.sqrt(np.mean(errors[:,1]**2)/len(data)))
    return dict(method=data[0]['method'], M=int(data[0]['M']), replicas=len(data),
                mean_green=mean, replica_sem=sem, reference=float(reference),
                discrepancy_sigma=abs(mean-reference)/max(sem,1e-15),
                internal_sem_8=internal,
                between_to_internal=sem/max(internal,1e-15),
                block_16_over_8_median=float(np.median(errors[:,2]/errors[:,1])),
                block_4_over_8_median=float(np.median(errors[:,0]/errors[:,1])))


def continuum_fit(rows, *, method, power=2, exclude=(), reference=None):
    """Independent-replica weighted fit; sensitivity and diagnostics mandatory.

    Model y = intercept + slope*tau**power. Assumes independent *grid* data;
    reused seeds/correlated runs invalidate standard fit covariance.
    """
    if power not in (1, 2):
        raise ValueError('power must be 1 or 2')
    select = [r for r in rows if r['method'] == method and int(r['M']) not in exclude]
    select.sort(key=lambda r:int(r['M']))
    if len(select)<3 or len({int(r['M']) for r in select}) != len(select):
        raise ValueError('at least 3 distinct grids required')
    tau=np.array([float(r['beta'])/int(r['M']) for r in select])
    y=np.array([float(r['mean_green']) for r in select])
    se=np.array([float(r['replica_sem']) for r in select])
    if not np.all(np.isfinite(y)) or not np.all(np.isfinite(se)) or np.any(se<=0):
        raise ValueError('invalid fit data')
    X=np.column_stack((np.ones(len(y)),tau**power))
    A=X/se[:,None]
    if np.linalg.cond(A)>1e12:
        raise ValueError('ill-conditioned fit')
    coef=np.linalg.lstsq(A,y/se,rcond=None)[0]
    cov=np.linalg.inv(A.T@A)
    residual=(y-X@coef)/se
    return dict(method=method, power=power, grids=','.join(str(int(r['M'])) for r in select),
                continuum=float(coef[0]), continuum_sem_formal=float(sqrt(cov[0,0])),
                slope=float(coef[1]), chi2=float(np.dot(residual,residual)), dof=len(y)-2,
                spectral_reference=reference if reference is not None else float('nan'),
                warning='Exploratory fit; fixed-power model and independent-grid SEM assumed')
