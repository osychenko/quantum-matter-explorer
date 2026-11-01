# Patch 21 — Canonical interacting 3D continuum spacetime-worm benchmark

Patch 21 transfers the **time-separated creation/annihilation endpoints** of the
Patch 20 lattice benchmark to **continuous 3D periodic coordinates**. The first
version is deliberately restricted to **zero or one background boson**; the
insertion sector then has **one or two particles**. This makes full bosonic
symmetrization of each short-time propagator explicit and inexpensive and
allows nontrivial checks against an independently calculated free-boson Green
function. It is not yet a general-N production worm algorithm.

## Physical observable (not a condensate fraction)

For a *canonical* background of N particles at inverse temperature beta,
we measure the **particle-addition** imaginary-time Green function:

\[
G_{ij}^{(+)}(t) = \frac{1}{Z_N}\,\mathrm{Tr}_{\mathcal H_N}
 \left[e^{-(\beta-t)H_N}a_i e^{-tH_{N+1}}a_j^\dagger\right],
\qquad 0<t<\beta.
\]

Its momentum-mode analogue is the thermal expectation
\(G_k^{(+)}(t)=\langle a_k(t)a_k^\dagger(0)\rangle_N\),
with \(a_k=V^{-1/2}\int_V e^{-ik\cdot r}\psi(r)\,dr\).
At **zero positive time**, \(G_k^{(+)}(0^+)=1+\langle n_k\rangle_N\).
At *finite* time \(G_k^{(+)}(t)\) is **not** \(n_k\), a superfluid
fraction, a condensate fraction, or a pure ground-state estimator.

The length of the full imaginary-time circle is \(\beta=M\tau\).
Each open interval lasts \(\ell\tau\) with \(\ell=1,\ldots,M-1\).
The tail and head have independent spatial positions, and both endpoint
times can advance or recede by one discretized link.

## Path weight and particle-number sectors

The **Z sector** contains a closed free path for N=0 or 1 boson. With one
particle, the correctly normalized closed host distribution is the periodic
Brownian loop. Its free-partition normalization is known analytically.

For one background boson, the **G sector** has an extra strand between time
slices `cut` and `cut+length`. Two particles propagate on those links with the
fully symmetrized free-boson kernel:

\[
K_2^S((r,e),(r',e');\tau)
 =\tfrac12\left[K_\tau(r,r')K_\tau(e,e')
 +K_\tau(r,e')K_\tau(e,r')\right].
\]

The exchanged term is **necessary**: using only a distinguishable extra strand
would return \(G_0=1\) for ideal bosons rather than the Bose-enhanced
\(1+1/Z_1(\beta)\) in the one-background-particle reference.

On every link inside the worm interval we replace the one-particle host kernel
with this two-boson kernel. The soft-core pair action is integrated by the
symmetric (trapezoidal) primitive rule:

\[
S_{\mathrm{extra}}=\tau\left[\frac{V_0+V_\ell}{2}
  +\sum_{j=1}^{\ell-1}V_j\right].
\]

The background one-particle potential is zero. The two-body potential is the
existing `SoftCorePairPotential`. **No additional interaction approximation**
is introduced beyond the symmetric primitive Trotter action, whose continuum
extrapolation must be checked. The N=0 vacuum uses the same construction with
just one particle in the worm interval.

### Explicit sector normalization

Choose the open interval origin and length uniformly:
\(p(cut,\ell)=1/[M(M-1)]\). Propose a free extra strand under
\(q(e)=V^{-1}\prod_{j=0}^{\ell-1}K_\tau(e_j,e_{j+1})\).
The extended target is specified by the **dimensionless** ratio

\[
\pi_G(X,cut,\ell,e)
=\pi_Z(X)\,\frac{q(e)}{M(M-1)}\,R(X,cut,\ell,e),
\]

where, for N=1,

\[
R=\eta (N+1) e^{-S_{\rm extra}}
\prod_{j=0}^{\ell-1}
\frac{K^S_{N+1}(R_j^+,R_{j+1}^+)}
     {K^S_N(R_j,R_{j+1}) K_1(e_j,e_{j+1})}.
\]

Here \(R_j^+=(R_j,e_j)\). The \(N+1\) prefactor arises from the
bosonic field creation/annihilation matrix elements, *not* from an adjustable
normalization. `sector_fugacity=eta` is the known (user-selected) relative
sector weight. In the vacuum, `R=eta` exactly for the free Hamiltonian.

For each length, the **known** normalization gives

\[
G_k^{(+)}(\ell\tau)
= \frac{1}{\eta/(M-1)}
\frac{\left\langle \mathbf1_{G,\ell}
\cos(k\cdot[e_\ell-e_0])\right\rangle}
{\left\langle \mathbf1_Z\right\rangle}.
\]

No post-hoc calibration against free reference results is performed.
Measurements at different lengths are correlated; independent full-chain
replicas determine error bars. A free-reference mismatch larger than twice
the between-replica SEM is **printed as a warning**, not suppressed. Such
replica SEMs can be overoptimistic for trapped or metastable chains. The per-link two-particle permanent may
increase variance as `M` grows. The output should **not** be treated as
converged at small timestep without substantially longer chains.

### Reversible Monte Carlo moves

- **Open/close:** choose the length, origin and free extra-strand proposal
  with their normalized densities; MH acceptance is `min(1,R)` or
  `min(1,1/R)`.
- **Global refresh:** independently propose a new free background loop,
  uniform origin/length, and free extra strand. In G its acceptance is
  `min(1,R_new/R_old)`; in Z it is an exact free heatbath.
- **Head/tail spatial displacement:** a symmetric wrapped Gaussian changes
  one endpoint. MH includes both the free extra-strand proposal-density ratio
  and `R_new/R_old`.
- **Head/tail time advance/recede:** add/remove a Gaussian free link at either
  endpoint. The added link's proposal density *exactly cancels* its factor in
  the extra-strand reference density. The acceptance is therefore
  `min(1,R_new/R_old)`. The four directions are selected with equal
  state-independent probabilities, and disallowed moves at interval bounds
  are self-loops. Never silently exclude invalid directions from the menu;
  doing so requires extra Hastings factors.
- **Exchange bridge:** at fixed open interval endpoints, choose a direct or
  exchanged two-boson endpoint pairing with its exact finite-time free heat
  kernel weight, draw two independent periodic Brownian bridges, and randomly
  relabel the intermediate two-particle configurations. This is the **exact
  conditional symmetric free two-body bridge**; all free kinetic and
  background-path proposal factors cancel. The acceptance depends *only* on
  the change of the interior pair-potential action. It is always accepted
  for free bosons. The move does not alter endpoint locations or times.
- **Extra-strand heatbath:** resample the whole extra free strand, with the
  `R_new/R_old` acceptance.

The host free loop is sampled with the exact periodic Brownian bridge, including
winding images. Every move preserves the normalized, **primitive-discretized**
extended target. This is a **finite-temperature** N to N+1 particle-addition
estimator, not a pure two-sided zero-temperature OBDM.

## Independent free reference

For N=0 and N=1 noninteracting bosons,

\[
G_k^{(+)}(t)=e^{-tD|k|^2}(1+n_k),\qquad
n_k=\begin{cases}0,&N=0,\\
e^{-\beta D|k|^2}/Z_1(\beta),&N=1.\end{cases}
\]

`Z_1(beta)` is evaluated by the independent reciprocal/Poisson Gaussian sum
from Patch 17. For the zero mode, vacuum gives 1; one boson gives
`1+1/Z_1(beta)`, independent of `t`. This is **Bose enhancement**, absent
from a distinguishable tagged-particle insertion.

## Run

```bash
python3 -m pytest -q
python3 -m examples.bosons.run_continuum_spacetime_worm --quick
python3 -m examples.bosons.run_continuum_spacetime_worm
# For more reliable reference comparisons:
python3 -m examples.bosons.run_continuum_spacetime_worm --steps 30000 --replicas 6
```

The runnable example compares vacuum and free one-background-particle references
against an interacting `N=1 -> N+1=2` soft-core insertion. It holds beta fixed
while changing `M` (full mode). Results are stored in
`outputs/bosons/continuum_spacetime_worm/`:

- `replicas.csv`: raw replica and mode/length results, sector visits, acceptance;
- `summary.csv`: means, SEM **across independent replicas**, and
  standardized differences from the exact free reference (not formal p-values);
- `protocol.json`: model parameters and explicit estimator normalization;
- `green_vs_time.png`: zero-mode time dependence;
- `nonzero_momentum.png`: nonzero-mode comparison with the free reference.

## Scientific limitations and next steps

1. **Background N<=1 only**. The interacting case has two bosons *inside*
   the insertion interval, not a general interacting many-body background.
2. **Finite temperature**, fixed canonical background, and finite primitive
   timestep. No pure ground-state extrapolation is made.
3. The **fully symmetrized two-body kernel** is evaluated explicitly at every
   time link. This is a correct finite-N target, but grows factorially beyond
   two particles; a production implementation needs **explicit sampled
   particle-line connections and exchange/reconnection moves**.
4. The sampler's **time-step and long-interval mixing can be poor** because
   the product of Bose-symmetrized short-time kernels creates rare-event
   weights relative to independent free-strand proposals. The two-particle
   exchange-bridge update improves access to these configurations but does
   not guarantee short sector residence times. Increased acceptance
   alone is not evidence of adequate effective sample size.
5. Analytic comparisons are available only in the **free** cases. Interacting
   estimates are exploratory, and independent two-body spectral reference
   calculations should be added before interpreting their values.
6. This Green function describes **addition** at finite time; recovering
   equal-time occupations requires controlled `tau -> 0+` analysis and a
   proper treatment of discretization and statistical uncertainty. A
   positive-time G should not be labeled condensate fraction or superfluidity.

The subsequent milestone is to generalize to N>1 with explicit exchange-line
reconnections and improve worm-sector mixing, then benchmark interacting
Green functions against an independent few-body solver.
