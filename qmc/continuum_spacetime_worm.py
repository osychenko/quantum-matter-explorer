"""Canonical 3D continuum N->N+1 *particle-addition* spacetime worm.

A finite-size, interacting two-boson benchmark, NOT general-N production QMC:
N_background is 0 (vacuum) or 1 (one ideal background boson).  The N+1
sector uses the fully Bose-symmetrized two-particle primitive propagator
at EVERY imaginary-time link, including exchange, and the extra-host pair
potential with symmetric Trotter endpoint weights.

The closed host is a free N=0/1 periodic loop.  An open interval of ell links
carries N+1 particles, with a free Gaussian reference proposal for the extra
strand.  The Metropolis ratios compare the *known normalized proposal laws*
so the relative sector normalization eta is not fitted.  Integration over
extra endpoints gives the canonical particle-addition Green function
G_k(tau)=<a_k(tau) a_k^dagger(0)>_N, not equal-time n_k or a pure projector.

The N=1 interaction is a true two-boson soft-core potential; extrapolate
primitive tau, convergence in chain length and finite box separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log

import numpy as np

from .cells import OrthorhombicCell
from .continuum_path import PeriodicFreePropagator
from .interacting_path import _integer, _positive
from .path_staging import periodic_bridge, autocorrelation_estimate
from .boson_permutations import free_one_particle_partition


@dataclass(frozen=True)
class ContinuumSpacetimeConfig:
    time_step: float = 0.55
    n_links: int = 4  # imaginary-time period beta = M*tau
    n_steps: int = 6500
    burn_in: int = 1200
    thin: int = 4
    sector_fugacity: float = 1.0
    toggle_probability: float = 0.42
    global_refresh_probability: float = 0.27
    time_move_probability: float = 0.45  # among local open-sector moves
    endpoint_move_probability: float = 0.20
    exchange_bridge_probability: float = 0.25
    move_sigma: float = 0.5
    seed: int = 2026

    def validate(self):
        for name in ("time_step", "sector_fugacity", "move_sigma"):
            _positive(getattr(self, name), name)
        for name, least in (("n_links", 3), ("n_steps", 2),
                            ("burn_in", 0), ("thin", 1), ("seed", 0)):
            _integer(getattr(self, name), name, least)
        if self.burn_in >= self.n_steps:
            raise ValueError("burn_in must be less than n_steps")
        if (self.n_steps-self.burn_in+self.thin-1)//self.thin < 4:
            raise ValueError("at least four measurements required")
        for name in ("toggle_probability", "global_refresh_probability",
                     "time_move_probability", "endpoint_move_probability",
                     "exchange_bridge_probability"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be finite and in [0,1]")
        if self.toggle_probability + self.global_refresh_probability > 1:
            raise ValueError("toggle + global refresh probabilities must sum <=1")
        if (self.time_move_probability+self.endpoint_move_probability
                +self.exchange_bridge_probability)>1:
            raise ValueError("time + endpoint + exchange bridge probabilities must sum <=1")
        return self

    @property
    def beta(self):
        return self.time_step*self.n_links


@dataclass(frozen=True)
class ContinuumGreenResult:
    beta: float
    imaginary_times: np.ndarray
    indices: np.ndarray
    green: np.ndarray             # (n_lengths, n_modes); G_k(tau), NOT occupation
    n_closed: int
    n_open_by_length: np.ndarray
    sector_effective_samples: float
    acceptances: dict[str, float]
    attempted: dict[str, int]


class ContinuumSpacetimeWorm:
    """Reversible N=0 or N=1 canonical insertion worm in a periodic 3D cell.

    For N=1, the two-body free kernel is
      K_2^S((r,e),(r',e')) = [K(r,r')K(e,e')+K(r,e')K(e,r')]/2.
    The product of these kernels is the correct symmetrized primitive
    N+1 projector, while the closed N=1 path uses K_1.  We sample a
    uniform origin and a uniform interval length 1..M-1; the extra strand
    is proposed under (1/V) prod K_1.  All selection densities cancel in
    open/close and time-advance ratios, leaving the explicit log factor.
    """

    def __init__(self, cell, potential, n_background: int,
                 config: ContinuumSpacetimeConfig = ContinuumSpacetimeConfig(),
                 *, kinetic_coefficient: float = 0.5,
                 indices=((0, 0, 0), (1, 0, 0))):
        if not isinstance(cell, OrthorhombicCell):
            raise TypeError("an orthorhombic periodic cell is required")
        self.n_background = _integer(n_background, "n_background", 0)
        if self.n_background > 1:
            raise ValueError("prototype currently supports N_background=0 or 1")
        self.config = config.validate()
        self.cell = cell
        self.kernel = PeriodicFreePropagator(cell, kinetic_coefficient)
        self.potential = potential
        if getattr(potential, "cell", None) is not None and not np.allclose(
                potential.cell.vectors, cell.vectors, rtol=1e-12, atol=1e-12):
            raise ValueError("potential uses a different cell")
        idx = np.asarray(indices)
        if (idx.ndim != 2 or idx.shape[1] != 3 or len(idx) < 1
                or not np.issubdtype(idx.dtype, np.number)
                or not np.all(np.isfinite(idx))
                or not np.all(idx == np.rint(idx))
                or np.max(np.abs(idx)) > 100):
            raise ValueError("indices must be finite integer triples")
        self.indices = idx.astype(int)
        if len(set(map(tuple, self.indices.tolist()))) != len(self.indices):
            raise ValueError("duplicate reciprocal modes")
        if not np.any(np.all(self.indices == 0, axis=1)):
            raise ValueError("include zero reciprocal mode")
        self.wavevectors = self.cell.wavevectors(self.indices)
        self.rng = np.random.default_rng(self.config.seed)
        self.host = self._new_host()
        self.open = False
        self.cut = None
        self.length = None
        self.extra = None
        self._counts = {k: [0, 0] for k in
                        ("open", "close", "global_refresh", "host_refresh",
                         "extra_refresh", "endpoint", "time", "exchange_bridge")}

    def _new_host(self):
        M = self.config.n_links
        if self.n_background == 0:
            return np.empty((M, 0, 3), dtype=float)
        start = self.cell.to_cartesian(self.rng.random((1, 3)))
        return periodic_bridge(self.kernel, self.rng, start, start, M,
                               self.config.time_step)[0:M]

    def _new_extra(self, length):
        start = self.cell.to_cartesian(self.rng.random(3))
        steps = np.sqrt(2*self.kernel.kinetic_coefficient*self.config.time_step) * (
            self.rng.normal(size=(length, 3)))
        return self.cell.wrap(start[None]+np.vstack((np.zeros(3), np.cumsum(steps, axis=0))))

    def _validate_state(self, host, cut, length, extra):
        M=self.config.n_links
        h=np.asarray(host, dtype=float)
        e=np.asarray(extra, dtype=float)
        if (h.shape != (M,self.n_background,3) or not np.all(np.isfinite(h))):
            raise ValueError("host path has invalid shape or values")
        if (not isinstance(cut, (int,np.integer)) or not 0<=cut<M
                or not isinstance(length,(int,np.integer)) or not 1<=length<M
                or e.shape!=(length+1,3) or not np.all(np.isfinite(e))):
            raise ValueError("worm cut/length/extra path invalid")
        return h,e

    def _log_extra_reference(self, extra):
        return (-log(self.cell.volume)
                +float(np.sum(self.kernel.log_kernel(extra[1:]-extra[:-1],self.config.time_step))))

    def log_relative_weight(self, host, cut, length, extra):
        """log R with pi_G=pi_Z*q(cut,length,extra)*R.

        q(cut,length,extra)=1/[M(M-1)V] prod K_tau(extra_s,extra_{s+1}).
        Thus R integrates to eta times the mean canonical addition G_0,
        averaged over lengths.  This is a density relative to the *exactly
        normalized* free proposal, not a fit to simulation occupancies.
        """
        h,e=self._validate_state(host,cut,length,extra)
        tau=self.config.time_step
        log_r=log(self.config.sector_fugacity) + log(self.n_background+1)
        if self.n_background == 1:
            for j in range(length):
                left=h[(cut+j)%len(h),0]
                right=h[(cut+j+1)%len(h),0]
                a=float(self.kernel.log_kernel(right-left,tau))
                b=float(self.kernel.log_kernel(e[j+1]-e[j],tau))
                c=float(self.kernel.log_kernel(e[j+1]-left,tau))
                d=float(self.kernel.log_kernel(right-e[j],tau))
                log_r+=float(np.logaddexp(a+b,c+d)-log(2)-a-b)
            host_interval=h[(cut+np.arange(length+1))%len(h),0]
            bead_positions=np.stack((host_interval,e),axis=1)
        else:
            bead_positions=e[:,None,:]
        vals=np.asarray(self.potential(bead_positions),dtype=float)
        if vals.shape!=(length+1,) or not np.all(np.isfinite(vals)):
            raise ValueError("potential must return finite one-energy-per-bead array")
        log_r-=tau*(0.5*(float(vals[0])+float(vals[-1]))
                    +float(np.sum(vals[1:-1])))
        if not np.isfinite(log_r):
            raise FloatingPointError("nonfinite worm relative weight")
        return float(log_r)

    def current_log_relative_weight(self):
        if not self.open:
            raise RuntimeError("not in G sector")
        return self.log_relative_weight(self.host,self.cut,self.length,self.extra)

    def _mh(self,name,log_ratio):
        self._counts[name][1]+=1
        if not np.isfinite(log_ratio):
            raise FloatingPointError("invalid Metropolis ratio")
        accepted=bool(np.log(self.rng.random())<min(0.,float(log_ratio)))
        if accepted:
            self._counts[name][0]+=1
        return accepted

    def move_toggle(self):
        if not self.open:
            cut=int(self.rng.integers(self.config.n_links))
            length=int(self.rng.integers(1,self.config.n_links))
            extra=self._new_extra(length)
            lr=self.log_relative_weight(self.host,cut,length,extra)
            if self._mh("open",lr):
                self.open,self.cut,self.length,self.extra=True,cut,length,extra
                return True
            return False
        if self._mh("close",-self.current_log_relative_weight()):
            self.open,self.cut,self.length,self.extra=False,None,None,None
            return True
        return False

    def move_global_refresh(self):
        """Independent q_Z*q(cut,length,extra) MH refresh in G, exact free refresh in Z."""
        host=self._new_host()
        if not self.open:
            self._counts["global_refresh"][0]+=1
            self._counts["global_refresh"][1]+=1
            self.host=host
            return True
        cut=int(self.rng.integers(self.config.n_links))
        length=int(self.rng.integers(1,self.config.n_links))
        extra=self._new_extra(length)
        lr=(self.log_relative_weight(host,cut,length,extra)
            -self.current_log_relative_weight())
        if self._mh("global_refresh",lr):
            self.host,self.cut,self.length,self.extra=host,cut,length,extra
            return True
        return False

    def move_host_refresh(self):
        """Exact refresh in closed sector, not a G-sector move."""
        if self.open:
            raise RuntimeError("host-only refresh requires closed sector")
        self._counts["host_refresh"][0]+=1
        self._counts["host_refresh"][1]+=1
        self.host=self._new_host()
        return True

    def move_extra_refresh(self):
        if not self.open:
            raise RuntimeError("extra refresh requires open sector")
        extra=self._new_extra(self.length)
        lr=(self.log_relative_weight(self.host,self.cut,self.length,extra)
            -self.current_log_relative_weight())
        if self._mh("extra_refresh",lr):
            self.extra=extra
            return True
        return False

    def move_endpoint(self, which="head"):
        """Symmetric wrapped Gaussian endpoint move with free-reference correction."""
        if not self.open:
            raise RuntimeError("endpoint move requires G sector")
        if which not in ("head","tail"):
            raise ValueError("which must be head or tail")
        index=-1 if which=="head" else 0
        extra=self.extra.copy()
        extra[index]=self.cell.wrap(extra[index]+self.config.move_sigma*self.rng.normal(3))
        log_q=(self._log_extra_reference(extra)-self._log_extra_reference(self.extra))
        log_r=(self.log_relative_weight(self.host,self.cut,self.length,extra)
               -self.current_log_relative_weight())
        if self._mh("endpoint",log_q+log_r):
            self.extra=extra
            return True
        return False

    def move_time(self, which="head", direction="extend"):
        """Reversible imaginary-time move; the free-link proposal cancels q.

        The four (head/tail,extend/retract) possibilities must be selected
        with STATE-INDEPENDENT probability, including invalid boundary moves
        as self-loops.  Otherwise Hastings selection factors are required.
        """
        if not self.open:
            raise RuntimeError("time move requires G sector")
        if which not in ("head","tail") or direction not in ("extend","retract"):
            raise ValueError("invalid endpoint or time direction")
        length=self.length
        self._counts["time"][1]+=1
        if direction=="extend" and length==self.config.n_links-1:
            return False
        if direction=="retract" and length==1:
            return False
        extra=self.extra.copy()
        cut=self.cut
        if direction=="extend":
            if which=="head":
                new=self.kernel.sample_step(extra[-1],self.config.time_step,self.rng)
                extra=np.vstack((extra,new))
            else:
                new=self.kernel.sample_step(extra[0],self.config.time_step,self.rng)
                extra=np.vstack((new,extra))
                cut=(cut-1)%self.config.n_links
            length+=1
        else:
            if which=="head":
                extra=extra[:-1]
            else:
                extra=extra[1:]
                cut=(cut+1)%self.config.n_links
            length-=1
        log_ratio=(self.log_relative_weight(self.host,cut,length,extra)
                   -self.current_log_relative_weight())
        # The q(extra) change equals the heatbath density of the added bead;
        # that factor cancels the forward/reverse time proposal exactly.
        if not np.isfinite(log_ratio):
            raise FloatingPointError("nonfinite time-move acceptance")
        if np.log(self.rng.random())<min(0.,float(log_ratio)):
            self._counts["time"][0]+=1
            self.cut,self.length,self.extra=cut,length,extra
            return True
        return False

    def move_exchange_bridge(self):
        """Exact free *Bose-symmetrized* conditional bridge of both particles.

        For ell links with fixed endpoints, K^S_2(ell*tau) has the direct
        and exchanged endpoint pairings, weighted by their heat kernels.
        Draw one pairing, two exact periodic Brownian bridges, then flip
        the intermediate particle labels independently with probability 1/2.
        This samples the full product of symmetrized free-link kernels,
        including all intermediate exchange possibilities.  With fixed
        endpoints only the **interior pair potential action** remains in MH.
        """
        if not self.open:
            raise RuntimeError("exchange bridge requires open sector")
        self._counts["exchange_bridge"][1] += 1
        if self.n_background != 1 or self.length < 2:
            return False
        ell=self.length
        idx=(self.cut+np.arange(ell+1))%self.config.n_links
        left=np.stack((self.host[idx[0],0],self.extra[0]),axis=0)
        right=np.stack((self.host[idx[-1],0],self.extra[-1]),axis=0)
        time=ell*self.config.time_step
        direct=float(np.sum(self.kernel.log_kernel(right-left,time)))
        exchanged=float(np.sum(self.kernel.log_kernel(right[::-1]-left,time)))
        p_exchange=float(np.exp(exchanged-np.logaddexp(direct,exchanged)))
        swap=bool(self.rng.random()<p_exchange)
        bridge=periodic_bridge(self.kernel,self.rng,left,
                               right[::-1] if swap else right,ell,
                               self.config.time_step,method="staging")
        # Average over the 2^(ell-1) equally weighted intermediate
        # label permutations of the symmetrized free-particle kernel.
        swaps=self.rng.random(ell-1)<0.5
        for j,do_swap in enumerate(swaps,start=1):
            if do_swap:
                bridge[j]=bridge[j,::-1].copy()
        host_new=self.host.copy()
        extra_new=self.extra.copy()
        host_new[idx[1:-1],0]=bridge[1:-1,0]
        extra_new[1:-1]=bridge[1:-1,1]
        before=np.asarray(self.potential(np.stack((self.host[idx[1:-1],0],
                                                    self.extra[1:-1]),axis=1)),dtype=float)
        after=np.asarray(self.potential(np.stack((host_new[idx[1:-1],0],
                                                   extra_new[1:-1]),axis=1)),dtype=float)
        if before.shape!=(ell-1,) or after.shape!=before.shape or not np.all(np.isfinite(after)):
            raise ValueError("invalid pair potential in exchange bridge")
        logratio=-self.config.time_step*float(np.sum(after-before))
        if np.log(self.rng.random())<min(0.,logratio):
            self.host,self.extra=host_new,extra_new
            self._counts["exchange_bridge"][0] += 1
            return True
        return False

    def step(self):
        u=self.rng.random()
        if u<self.config.toggle_probability:
            return self.move_toggle()
        if u<self.config.toggle_probability+self.config.global_refresh_probability:
            return self.move_global_refresh()
        if not self.open:
            return self.move_host_refresh()
        v=self.rng.random()
        if v<self.config.time_move_probability:
            return self.move_time(
                which="head" if self.rng.random()<0.5 else "tail",
                direction="extend" if self.rng.random()<0.5 else "retract")
        if v<self.config.time_move_probability+self.config.endpoint_move_probability:
            return self.move_endpoint("head" if self.rng.random()<0.5 else "tail")
        if v<(self.config.time_move_probability+self.config.endpoint_move_probability
              +self.config.exchange_bridge_probability):
            return self.move_exchange_bridge()
        return self.move_extra_refresh()

    def run(self):
        """Estimate G_k(ell*tau), with exact eta/(M-1) sector normalization."""
        M=self.config.n_links
        closed=open_counts=0
        counts=np.zeros(M-1,dtype=int)
        numerators=np.zeros((M-1,len(self.indices)),dtype=float)
        sectors=[]
        for step in range(self.config.n_steps):
            self.step()
            if step<self.config.burn_in or (step-self.config.burn_in)%self.config.thin:
                continue
            if not self.open:
                closed+=1
                sectors.append(0.)
            else:
                open_counts+=1
                counts[self.length-1]+=1
                shift=self.extra[-1]-self.extra[0]
                numerators[self.length-1]+=np.cos(self.wavevectors@shift)
                sectors.append(1.)
        if closed<4 or open_counts<4:
            raise RuntimeError("insufficient Z/G visits; increase steps or tune eta")
        if np.any(counts==0):
            raise RuntimeError("at least one imaginary-time interval was never sampled")
        # The proposal prior pi_length=1/(M-1) is part of the known G density.
        green=numerators*(M-1)/(self.config.sector_fugacity*closed)
        ess=autocorrelation_estimate(sectors).effective_samples
        return ContinuumGreenResult(
            beta=self.config.beta,
            imaginary_times=np.arange(1,M)*self.config.time_step,
            indices=self.indices.copy(),green=green,
            n_closed=closed,n_open_by_length=counts,
            sector_effective_samples=ess,
            acceptances={k:a/n if n else float("nan") for k,(a,n) in self._counts.items()},
            attempted={k:n for k,(_,n) in self._counts.items()},
        )


def exact_free_addition_green(cell, n_background, beta, times, indices,
                              kinetic_coefficient=0.5):
    """Exact canonical ideal-gas G_k(t) for N=0 or 1.

    For N=1, occupation n_k=exp(-beta*epsilon_k)/Z_1(beta).
    G_k(t)=exp(-t*epsilon_k)*(1+n_k).  Free vacuum has n_k=0.
    """
    if not isinstance(cell, OrthorhombicCell):
        raise TypeError("orthorhombic cell required")
    n=_integer(n_background,"n_background",0)
    if n>1:
        raise ValueError("exact benchmark supports only vacuum or one background boson")
    beta=_positive(beta,"beta")
    D=_positive(kinetic_coefficient,"kinetic_coefficient")
    times=np.asarray(times,dtype=float)
    if times.ndim!=1 or np.any(~np.isfinite(times)) or np.any(times<0) or np.any(times>=beta):
        raise ValueError("times must be a 1D array in [0,beta)")
    idx=np.asarray(indices)
    if idx.ndim!=2 or idx.shape[1]!=3:
        raise ValueError("indices must be (n_modes,3)")
    eps=D*np.sum(cell.wavevectors(idx)**2,axis=1)
    n_k=np.exp(-beta*eps)/free_one_particle_partition(cell,beta,D) if n else np.zeros_like(eps)
    return np.exp(-times[:,None]*eps[None,:])*(1+n_k)[None,:]
