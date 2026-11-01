"""Time-slice-mobile, equal-time canonical boundary-link worm.

Extends Patch 18 from a single closure cut to one cut at ANY of M time links.
Conditional weights include the required 1/M factor; Gibbs moves can relocate
that cut in imaginary time without violating detailed balance. The two open
endpoints are still at the SAME imaginary-time cut (not a separated-time
Green-function worm). Winding is measured only in the closed sector.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .worm import CanonicalWorm, WormConfig
from .path_staging import autocorrelation_estimate
from .interacting_path import _integer
from .winding import winding_moments


@dataclass(frozen=True)
class MobileWormConfig(WormConfig):
    relocate_probability: float = 0.5

    def validate(self):
        super().validate()
        if not np.isfinite(self.relocate_probability) or not 0 <= self.relocate_probability <= 1:
            raise ValueError('relocate_probability must be in [0,1]')
        return self


@dataclass(frozen=True)
class MobileWormResult:
    beta: float
    indices: np.ndarray
    occupation_fractions: np.ndarray
    closed_occupation_fractions: np.ndarray
    n_closed: int
    n_open: int
    n0_fraction: float
    winding_squared: np.ndarray
    superfluid_fraction_axes: np.ndarray
    superfluid_fraction_isotropic: float
    time_cut_counts: np.ndarray
    sector_effective_samples: float
    acceptance: dict[str,float]
    attempted: dict[str,int]


def _deltas(path,permutation):
    """(M,N,3) displacement from bead t to next bead, including closure P."""
    x=np.asarray(path,dtype=float)
    M,N=x.shape[:2]
    result=np.empty_like(x)
    result[:-1]=x[1:]-x[:-1]
    result[-1]=x[0,permutation]-x[-1]
    return result


class MobileCutWorm(CanonicalWorm):
    """Patch-18 Z/G sector with an explicitly sampled imaginary-time cut.

    For M links and N strands, pi_G(t,i,s)/pi_Z is
        eta / (M*N*V) * K_tau(Delta_(t,i)+s)/K_tau(Delta_(t,i)).
    This normalizes the *time-averaged*, finite-tau one-body insertion.
    """

    def __init__(self,cell,potential,n_particles,config:MobileWormConfig=MobileWormConfig(),
                 *,kinetic_coefficient=0.5,indices=((0,0,0),(1,0,0))):
        if not isinstance(config,MobileWormConfig):
            raise TypeError('MobileCutWorm requires MobileWormConfig')
        super().__init__(cell,potential,n_particles,config,
                         kinetic_coefficient=kinetic_coefficient,indices=indices)
        self.cut=None
        self._counts['relocate_cut']=[0,0]

    def _delta_at(self,cut,particle,path=None,permutation=None):
        x=self.base.path if path is None else path
        perm=self.base.permutation if permutation is None else permutation
        M=self.config.n_beads
        if not 0 <= cut < M or not 0 <= particle < self.n_particles:
            raise IndexError('cut/particle out of range')
        if cut==M-1:
            return x[0,perm[particle]]-x[-1,particle]
        return x[cut+1,particle]-x[cut,particle]

    def _delta(self,i):
        return self._delta_at(self.config.n_beads-1 if self.cut is None else self.cut,i)

    def log_sector_ratio(self,particle,shift,*,cut=None):
        i=_integer(particle,'particle',0)
        if i>=self.n_particles:
            raise IndexError('particle out of range')
        t=self.config.n_beads-1 if cut is None else _integer(cut,'cut',0)
        delta=self._delta_at(t,i)
        s=np.asarray(shift,dtype=float)
        if s.shape!=(3,) or not np.all(np.isfinite(s)):
            raise ValueError('shift must be finite (3,)')
        return float(np.log(self.config.sector_fugacity)
                     -np.log(self.config.n_beads*self.n_particles*self.cell.volume)
                     +self.kernel.log_kernel(delta+s,self.config.time_step)
                     -self.kernel.log_kernel(delta,self.config.time_step))

    def _heatbath_shift_at(self,cut,particle):
        y=self.kernel.sample_step(np.zeros(3),self.config.time_step,self.rng)
        return self.cell.wrap(y-self._delta_at(cut,particle))

    def _heatbath_shift(self,particle):
        return self._heatbath_shift_at(self.cut,particle)

    def conditional_sector_probabilities(self):
        delta=_deltas(self.base.path,self.base.permutation)
        logk=np.asarray(self.kernel.log_kernel(delta,self.config.time_step))
        logg=np.log(self.config.sector_fugacity)-np.log(
            self.config.n_beads*self.n_particles*self.cell.volume)-logk
        z=np.r_[0.,logg.ravel()]
        z-=z.max()
        weights=np.exp(z)
        return weights/weights.sum()

    def _collapsed_log_normalizer(self,path,perm):
        logk=self.kernel.log_kernel(_deltas(path,perm),self.config.time_step)
        z=np.r_[0.,(np.log(self.config.sector_fugacity)
                   -np.log(self.config.n_beads*self.n_particles*self.cell.volume)
                   -logk).ravel()]
        m=float(z.max())
        return m+np.log(float(np.exp(z-m).sum()))

    def move_sector_gibbs(self):
        probabilities=self.conditional_sector_probabilities()
        choice=int(self.rng.choice(len(probabilities),p=probabilities))
        if choice==0:
            self.open,self.particle,self.cut,self.shift=False,None,None,None
        else:
            self.open=True
            self.cut,self.particle=divmod(choice-1,self.n_particles)
            self.shift=self._heatbath_shift_at(self.cut,self.particle)
        self._counts['sector_gibbs'][0]+=1
        self._counts['sector_gibbs'][1]+=1
        return True

    def move_toggle(self):
        if not self.open:
            t=int(self.rng.integers(self.config.n_beads))
            i=int(self.rng.integers(self.n_particles))
            logk=float(self.kernel.log_kernel(self._delta_at(t,i),self.config.time_step))
            logratio=np.log(self.config.sector_fugacity)-np.log(self.cell.volume)-logk
            if self._accept('open',logratio):
                self.open,self.cut,self.particle,self.shift=True,t,i,self._heatbath_shift_at(t,i)
                return True
            return False
        logk=float(self.kernel.log_kernel(self._delta_at(self.cut,self.particle),self.config.time_step))
        logratio=np.log(self.cell.volume)+logk-np.log(self.config.sector_fugacity)
        if self._accept('close',logratio):
            self.open,self.particle,self.cut,self.shift=False,None,None,None
            return True
        return False

    def move_relocate_cut(self):
        """Conditional Gibbs relocation across all imaginary-time cuts and strands."""
        if not self.open:
            raise RuntimeError('relocation requires the worm sector')
        probabilities=self.conditional_sector_probabilities()[1:]
        probabilities=probabilities/probabilities.sum()
        choice=int(self.rng.choice(len(probabilities),p=probabilities))
        self.cut,self.particle=divmod(choice,self.n_particles)
        self.shift=self._heatbath_shift_at(self.cut,self.particle)
        self._counts['relocate_cut'][0]+=1
        self._counts['relocate_cut'][1]+=1
        return True

    def step(self):
        u=self.rng.random()
        if u<self.config.collapsed_refresh_probability:
            return self.move_collapsed_refresh()
        if u<self.config.collapsed_refresh_probability+self.config.toggle_probability:
            if self.rng.random()<self.config.gibbs_probability:
                return self.move_sector_gibbs()
            return self.move_toggle()
        if self.open:
            if self.rng.random()<self.config.relocate_probability:
                return self.move_relocate_cut()
            method='local' if self.rng.random()<self.config.local_shift_probability else 'heatbath'
            return self.move_shift(method=method)
        return self.base.step()

    def _closed_occupation(self):
        delta=_deltas(self.base.path,self.base.permutation)
        k2=np.sum(self.wavevectors**2,axis=1)
        k=self.kernel.log_kernel(delta,self.config.time_step)
        factor=np.exp(-k)/(self.cell.volume)
        phase=np.cos(delta@self.wavevectors.T)
        return ((phase*factor[...,None]).mean(axis=(0,1))
                *np.exp(-self.config.time_step*self.kernel.kinetic_coefficient*k2))

    def run(self) -> MobileWormResult:
        cfg=self.config
        numerator=np.zeros(len(self.indices))
        closed_occ=np.zeros(len(self.indices))
        w2=np.zeros(3)
        counts=np.zeros(cfg.n_beads,dtype=int)
        nZ=nG=0
        sectors=[]
        for step in range(cfg.n_steps):
            self.step()
            if step<cfg.burn_in or (step-cfg.burn_in)%cfg.thin:
                continue
            if self.open:
                nG+=1
                counts[self.cut]+=1
                numerator+=np.cos(self.wavevectors@self.shift)
                sectors.append(1.)
            else:
                nZ+=1
                closed_occ+=self._closed_occupation()
                w2+=winding_moments(self.base).winding_squared
                sectors.append(0.)
        if min(nZ,nG)<max(10,int(.01*(nZ+nG))):
            raise RuntimeError('insufficient worm sectors; extend run or tune sector_fugacity')
        second=w2/nZ
        L=np.asarray(self.cell.lengths)
        fractions=L**2*second/(2*self.kernel.kinetic_coefficient*cfg.beta*self.n_particles)
        acc={k:a/n if n else float('nan') for k,(a,n) in
             {**self.base._counts,**self._counts}.items()}
        attempts={k:n for k,(a,n) in {**self.base._counts,**self._counts}.items()}
        occupied=numerator/(cfg.sector_fugacity*nZ)
        return MobileWormResult(
            beta=cfg.beta,indices=self.indices.copy(),occupation_fractions=occupied,
            closed_occupation_fractions=closed_occ/nZ,n_closed=nZ,n_open=nG,
            n0_fraction=float(occupied[np.flatnonzero(np.all(self.indices==0,axis=1))[0]]),
            winding_squared=second,superfluid_fraction_axes=fractions,
            superfluid_fraction_isotropic=float(np.mean(fractions)),
            time_cut_counts=counts,sector_effective_samples=autocorrelation_estimate(sectors).effective_samples,
            acceptance=acc,attempted=attempts)
