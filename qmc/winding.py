"""Periodic-image winding measurements for canonical bosonic ring polymers.

The wrapped free heat kernel is a SUM over integer image translations.  Given
wrapped beads and a permutation, those image variables have independent
conditional Gaussian-discrete distributions on every link.  Integrating their
first/second moments exactly (Rao--Blackwellization) gives <W_a**2> without
noisy draws of rare image changes.  The interaction action depends only on
wrapped bead positions, so the conditional law is unchanged by V(R).

This is a *finite-temperature*, finite-size helicity-response diagnostic; it
is not a direct measurement of BEC or a proof of thermodynamic superfluidity.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .boson_permutations import (BosonPermutationPaths,
                                 free_one_particle_partition)
from .cells import OrthorhombicCell
from .interacting_path import _integer, _positive


@dataclass(frozen=True)
class WindingEstimate:
    winding_mean: np.ndarray
    winding_squared: np.ndarray
    fraction_axes: np.ndarray
    fraction_isotropic: float


def _image_moments(delta, length, sigma):
    """Return E[integer image], Var[image] for the wrapped Gaussian kernel."""
    delta = np.asarray(delta, dtype=float)
    center = -np.rint(delta / length).astype(np.int64)
    residual = delta + center*length
    radius = max(1, int(np.ceil(0.5 + 8.0*sigma/length)))
    offsets = np.arange(-radius, radius+1, dtype=np.int64)
    images = center[..., None] + offsets
    exponent = -0.5*((residual[..., None]+offsets*length)/sigma)**2
    exponent -= np.max(exponent, axis=-1, keepdims=True)
    prob = np.exp(exponent)
    prob /= np.sum(prob, axis=-1, keepdims=True)
    mean = np.sum(prob * images, axis=-1)
    var = np.sum(prob * (images-mean[...,None])**2, axis=-1)
    return mean, np.maximum(0.0,var), images, prob


def _link_deltas(path, permutation):
    x = np.asarray(path, dtype=float)
    p = np.asarray(permutation)
    if x.ndim != 3 or x.shape[-1] != 3 or x.shape[0] < 2:
        raise ValueError('path must have (beads,particles,3) shape')
    n = x.shape[1]
    if (p.shape != (n,) or not np.issubdtype(p.dtype,np.integer)
            or not np.array_equal(np.sort(p),np.arange(n))
            or not np.all(np.isfinite(x))):
        raise ValueError('invalid permutation or path')
    # The raw sum of these deltas vanishes EXACTLY, even for exchanges;
    # therefore global W is the integer sum of image shifts.
    return np.concatenate((x[1:]-x[:-1],
                           (x[0,p]-x[-1])[None]),axis=0).reshape(-1,3)


def winding_moments(sampler: BosonPermutationPaths) -> WindingEstimate:
    """Conditional E[W] and E[W²] given one equilibrium closed configuration."""
    if not isinstance(sampler,BosonPermutationPaths):
        raise TypeError('expected a BosonPermutationPaths sampler')
    delta = _link_deltas(sampler.path,sampler.permutation)
    D = sampler.kernel.kinetic_coefficient
    tau = sampler.config.time_step
    sigma = np.sqrt(2*D*tau)
    means = np.zeros(3)
    second = np.zeros(3)
    for axis,L in enumerate(sampler.cell.lengths):
        mu,var,_,_ = _image_moments(delta[:,axis], L, sigma)
        means[axis] = float(np.sum(mu))
        second[axis] = float(np.sum(var)+means[axis]**2)
    beta = sampler.config.beta
    fractions = np.square(sampler.cell.lengths)*second/(2*D*beta*sampler.n_particles)
    return WindingEstimate(means,second,fractions,float(np.mean(fractions)))


def draw_winding(sampler: BosonPermutationPaths, rng: np.random.Generator):
    """Draw one integer winding vector conditional on the wrapped closed path."""
    if not isinstance(sampler,BosonPermutationPaths):
        raise TypeError('expected a BosonPermutationPaths sampler')
    delta = _link_deltas(sampler.path,sampler.permutation)
    sigma = np.sqrt(2*sampler.kernel.kinetic_coefficient*sampler.config.time_step)
    out = np.zeros(3,dtype=np.int64)
    for axis,L in enumerate(sampler.cell.lengths):
        _,_,images,weights = _image_moments(delta[:,axis],L,sigma)
        u = rng.random(len(delta))
        ix = np.sum(u[:,None]>np.cumsum(weights,axis=-1),axis=-1)
        ix=np.minimum(ix,images.shape[-1]-1)
        out[axis]=int(np.sum(images[np.arange(len(delta)),ix]))
    return out


def _free_one_winding_moment(length, beta, D):
    """Return trace Z1 and its UNNORMALIZED second winding moment (1D)."""
    sigma=np.sqrt(2*D*beta)
    radius=max(1,int(np.ceil(8.0*sigma/length)))
    w=np.arange(-radius,radius+1,dtype=float)
    ex=-0.5*(w*length/sigma)**2
    probs=np.exp(ex-np.max(ex))
    z=float(np.sum(probs))
    # Real-space heat-kernel trace for the periodic line: L/sqrt(4 pi D t)
    # *sum_w exp(-(wL)^2/(4Dt)).  Recover unshifted weights for Z1.
    pref=length/(np.sqrt(2*np.pi)*sigma)
    z=pref*np.exp(np.max(ex))*z
    q=pref*np.exp(np.max(ex))*float(np.sum(w*w*probs))
    return z,q


def canonical_free_winding(cell: OrthorhombicCell,n_particles:int,beta:float,
                           kinetic_coefficient:float=0.5) -> WindingEstimate:
    """Exact canonical ideal-Bose <W_a²> from the cycle partition recursion.

    A length-l exchange cycle is one periodic free Brownian loop of duration
    l*beta.  Its winding characteristic function generates the second moment.
    """
    if not isinstance(cell,OrthorhombicCell):
        raise TypeError('cell must be orthorhombic')
    N=_integer(n_particles,'n_particles',1)
    beta,D=_positive(beta,'beta'),_positive(kinetic_coefficient,'kinetic_coefficient')
    z=np.zeros(N+1);q=np.zeros((N+1,3));z[0]=1
    for n in range(1,N+1):
        for l in range(1,n+1):
            values=[_free_one_winding_moment(L,l*beta,D) for L in cell.lengths]
            z1=float(np.prod([x[0] for x in values]))
            q1=np.array([values[i][1]*np.prod([values[j][0] for j in range(3) if j!=i])
                         for i in range(3)])
            z[n]+=z1*z[n-l]/n
            q[n]+=(q1*z[n-l]+z1*q[n-l])/n
    second=q[N]/z[N]
    frac=np.square(cell.lengths)*second/(2*D*beta*N)
    return WindingEstimate(np.zeros(3),second,frac,float(np.mean(frac)))
