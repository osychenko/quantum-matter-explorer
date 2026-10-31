"""Independent Fourier-Galerkin reference for TWO 3D periodic soft-core bosons.

Uniform center-of-mass sector and relative coordinate r=r2-r1 reduce
H=-D(grad_1^2+grad_2^2)+V(r) to H_rel=-2D grad_r^2+V(r).
Truncating relative plane waves to integer momenta [-K,K]^3 yields a small
Hermitian eigenproblem. Fourier coefficients of the soft-core potential and
Jastrow trial are computed by separate real-space quadrature on a 3D grid.
Errors from basis truncation and quadrature must be checked by increasing
K and n_grid; the reference is NOT an exact finite-basis continuum answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np


@dataclass(frozen=True)
class RelativeReferenceResult:
    beta: float
    energy: float
    midpoint_potential: float
    gap: float
    n_modes: int
    grid_points_per_axis: int


class TwoBosonRelativeReference:
    def __init__(self, *, cell_length: float, kinetic_coefficient: float,
                 height: float, cutoff: float, jastrow_amplitude: float = 0.0,
                 k_max: int = 2, n_grid: int = 32):
        self.length = float(cell_length)
        self.D = float(kinetic_coefficient)
        self.height = float(height)
        self.cutoff = float(cutoff)
        self.amplitude = float(jastrow_amplitude)
        for label, value in (("cell_length", self.length), ("kinetic_coefficient", self.D),
                             ("cutoff", self.cutoff)):
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{label} must be positive and finite")
        if (not np.isfinite(self.height) or self.height < 0
                or not np.isfinite(self.amplitude) or self.amplitude < 0):
            raise ValueError("height and jastrow_amplitude must be finite and nonnegative")
        if self.cutoff > self.length / 2:
            raise ValueError("cutoff must not exceed the cell inradius")
        for label, x, minimum in (("k_max", k_max, 1), ("n_grid", n_grid, 8)):
            if isinstance(x, (bool, np.bool_)) or not isinstance(x, (int, np.integer)) or x < minimum:
                raise ValueError(f"{label} must be an integer >= {minimum}")
        if n_grid < 4*k_max + 1:
            raise ValueError("n_grid too small to resolve basis Fourier differences")
        if (2*k_max + 1)**3 > 343:
            raise ValueError("dense Fourier benchmark limited to 343 modes")
        self.k_max = int(k_max)
        self.n_grid = int(n_grid)
        self.modes = np.array(list(product(range(-k_max,k_max+1),repeat=3)),dtype=int)
        self._setup()

    def _setup(self):
        L, n = self.length, self.n_grid
        grid = np.arange(n) * L / n
        gx, gy, gz = np.meshgrid(grid,grid,grid,indexing="ij")
        squared = np.zeros((n,n,n),dtype=float)
        for axis in (gx,gy,gz):
            dr = axis - L*np.rint(axis/L)
            squared += dr**2
        shape = np.maximum(1-squared/self.cutoff**2, 0.0)**3
        potential = self.height * shape
        trial = np.exp(-self.amplitude*shape)
        vhat = np.fft.fftn(potential) / n**3
        phihat = np.fft.fftn(trial) / n**3
        dk = self.modes[:,None,:] - self.modes[None,:,:]
        index = tuple(np.mod(dk[:,:,a],n) for a in range(3))
        self.v_matrix = np.asarray(vhat[index],dtype=complex)
        ksq = np.sum(self.modes**2,axis=1)
        self.hamiltonian = self.v_matrix + np.diag(2*self.D*(2*np.pi/L)**2*ksq)
        if not np.allclose(self.hamiltonian,self.hamiltonian.conj().T,atol=1e-12):
            raise RuntimeError("Fourier Hamiltonian is not Hermitian")
        self.energies, self.eigenvectors = np.linalg.eigh(self.hamiltonian)
        seed = np.array([phihat[tuple(np.mod(k,n))] for k in self.modes], dtype=complex)
        self.initial = seed / np.linalg.norm(seed)
        self.coefficients = self.eigenvectors.conj().T @ self.initial

    def measure(self,beta: float) -> RelativeReferenceResult:
        beta = float(beta)
        if np.isnan(beta) or beta < 0:
            raise ValueError("beta must be >=0")
        if np.isinf(beta):
            if abs(self.coefficients[0]) < 1e-13:
                raise ValueError("trial has zero ground-state overlap")
            vec = self.eigenvectors[:,0]
        else:
            weights = np.exp(-beta*(self.energies-self.energies[0])) * self.coefficients
            vec = self.eigenvectors @ weights
            vec /= np.linalg.norm(vec)
        energy = float(np.real(np.vdot(vec,self.hamiltonian @ vec)))
        potential = float(np.real(np.vdot(vec,self.v_matrix @ vec)))
        return RelativeReferenceResult(beta, energy, potential,
                                       float(self.energies[1]-self.energies[0]),
                                       len(self.modes),self.n_grid)
