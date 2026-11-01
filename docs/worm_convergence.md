# Patch 22 — canonical continuum worm reliability and independent interacting reference

Patch 21 introduced a canonical finite-temperature particle-addition worm for
zero or one background boson. The **additional** Patch 22 modules address two
limitations: its open/closed sector noise, and the absence of an *interacting*
reference Green function. The existing QMC solvers are unchanged.

## Precisely which Green function?

For an ideal one-particle background, with an interacting **two-boson** added
sector, the observable is

\[
G_k^{(+)}(t)=\frac{1}{Z_1}\mathrm{Tr}_{\mathcal H_1}
[e^{-(\beta-t)H_1}a_k e^{-tH_2}a_k^\dagger],
\qquad 0\le t<\beta .
\]

`N_background=1`, not a many-particle thermodynamic limit. No chemical
potential is introduced. This is a **thermal particle-addition correlator**, not
`n_k`, a condensate fraction or a ground-state projector OBDM. Its physical
value depends on inverse temperature `beta`, unlike a simple projection
parameter whose reference energy can be arbitrarily shifted.

## Independent plane-wave reference

`TwoBosonAdditionReference` constructs a finite symmetric *momentum-Fock*
basis from periodic single-particle modes `n ∈ [-K,K]^3`. States
`|p,q⟩_S` are properly normalized, including the `p=q` case. It forms the
soft-core interaction matrix from a real-space periodic FFT quadrature,
and diagonalizes each fixed-total-momentum block separately.

The independently computed Green function is

\[
G_k^{(+)}(t)=\frac{1}{Z_1}\sum_p e^{-(\beta-t)\epsilon_p}
(1+\delta_{pk})\sum_\alpha
|\langle p,k|_S\alpha,p+k\rangle|^2 e^{-t E_{\alpha,p+k}}.
\]

The `1+δ` factor is **Bose enhancement**. For a zero potential, the
implementation reproduces the exact canonical ideal-boson formula
`exp(-t*epsilon_k)*(1+exp(-beta*epsilon_k)/Z1)` at both basis cutoffs.

The Fourier **Galerkin** Hamiltonian is an *independent* approximation to the
continuum two-body problem. Increase `k_max` and `n_grid` to assess its
basis/quadrature error. The experiment writes both `K=1` and `K=2` results;
their difference is a diagnostic, **not** a bound on the remaining error.
The time-step error in the primitive-action worm is a different systematic
uncertainty; the reference uses no imaginary-time Trotter discretization.

## Exact augmented sector law

The Patch 21 normalized free extra-path proposal is `q(cut,length,extra)`.
Its relative open-sector weight is `R(X,cut,length,extra)` from
`ContinuumSpacetimeWorm.log_relative_weight`. Retain an *auxiliary* extra
path even in the closed Z sector and use

\[
\pi(X,u,z)\propto q_{\mathrm{host}}(X)q(u)
\begin{cases}1,&z=Z,\\ R(X,u),&z=G.\end{cases}
\]

Integrating over `u` in Z recovers the original closed ensemble; the G
sector remains exactly the same as Patch 21. The conditional probability
`p(G|X,u)=R/(1+R)` permits a rejection-free Gibbs sector update.

Closed-sector ghost-refresh draws `u~q` with acceptance one. Open-sector
updates use Patch 21's valid heatbath/reptation/endpoint/exchange proposals;
independent all-path proposals remain Metropolis corrected by `R_new/R_old`.
This is a **different Markov chain with the same extended target**, not a
change in Hamiltonian or fitted normalization.

Replacing raw sector indicators by their conditional expectations gives

\[
\widehat G_k^{\rm RB}(t_\ell)=
\frac{M-1}{\eta}\frac{\sum_s p_s(G)\,1_{\ell_s=\ell}
\cos(k\cdot\Delta r_s)}{\sum_s[1-p_s(G)]}.
\]

The original raw sector-count estimator is reported alongside it. This
Rao–Blackwellization reduces *conditional Bernoulli noise*. It cannot fix
long correlation times of ghost paths, omitted two-body basis modes, or
primitive-action discretization bias. Use independent replica SEMs, assess
within-chain integrated autocorrelation, and do not clip negative/large noisy
estimates. ESS diagnostics are approximate for metastable chains.

## Run

From the repository root:

```bash
python3 -m pytest -q
python3 -m examples.bosons.run_worm_convergence --quick
python3 -m examples.bosons.run_worm_convergence
python3 -m examples.bosons.run_worm_convergence --steps 20000 --replicas 6
```

The standard experiment uses `beta=1.6,2.4`, `M=4,6` imaginary-time links,
`tau=beta/M`, and both free and repulsive inserted two-boson sectors at the
same finite box size. The same physical `t=beta/2` is available in both grids.
The short quick test does *not* demonstrate convergence.

Outputs under `outputs/bosons/worm_convergence/`:

- `spectral_reference.csv` — K=1/K=2 independently diagonalized Green curves;
- `replicas.csv` — raw and Gibbs estimates, seeds, sector counts, ESS and CPU time;
- `summary.csv` — independent-replica mean/SEM and separate spectral errors;
- `protocol.json` — inputs, interpretation and stated limits;
- `green_spectral.png` and `bias_diagnostic.png` — comparisons and residuals.

## Remaining work

Even an apparent agreement of two short curves is **not proof of**
convergence. The next validation should systematically vary reference basis,
primitive `tau`, `beta`, chain lengths, move mixtures and initial states.
Only then enlarge the background particle count and implement reconnections
of more than two interacting bosons. This thermal addition-Green method does
not yet yield the pure ground-state condensate fraction.
