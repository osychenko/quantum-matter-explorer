# Patch 19 — imaginary-time cut mobility and winding-number response

This is a **finite-temperature, canonical, finite-system** PIMC extension. It
adds a time-slice-mobile **equal-time** closure-link worm to the normalized
Z/G construction of Patch 18 and a variance-reduced winding-number estimator
for Patch 17's exchange-connected closed worldlines.

**Scope boundary:** the worm's two endpoints are still located at the *same*
imaginary-time cut. Changing the cut time is not the same as independently
propagating the worm head and tail in spacetime, and is not a measurement of a
nonzero-imaginary-time Green function. This patch does not supply a pure
zero-temperature one-body density matrix.

## Mobile-cut off-diagonal sector

The closed bosonic path has `M` links, `N` particle strands, and fixed
permutation closure `P`. Each link has free periodic heat kernel `K_tau` and
one displacement `Delta[t,i]` (interior links or the permutation closure).
For any time-link `t`, strand `i`, and uniform displacement `s` in the
primitive cell, the joint *density* of the open sector relative to the
closed sector is

\[
\frac{\pi_G(X,P,t,i,s)}{\pi_Z(X,P)}
  =\frac{\eta}{MN V}
   \frac{K_\tau(\Delta_{t,i}+s)}{K_\tau(\Delta_{t,i})}.
\]

The factor `1/M` is essential. The path weight is integrated over the
uniform Cartesian measure `ds` and summed over the discrete time-link and
particle indices. For the finite-step kinetic-link insertion, the resulting
sector normalization is

\[
 Z_G/Z_Z=\eta\,\overline{n_0/N}^{\mathrm{link}},\qquad
 \widehat{n_0/N}=N_G/(\eta N_Z).
\]

The overbar is the average of **link-insertion** estimates over all `M`
imaginary-time slices; at nonzero primitive timestep these are only a
specified discretized off-diagonal convention, not an exact continuous-time
operator insertion for interacting particles.

The same sector also gives finite-cell reciprocal-mode occupation estimates:

\[
 \widehat{n_k/N}=\frac{1}{\eta N_Z}
    \sum_{\mathrm{G\ measurements}}\cos(k\cdot s).
\]

The independent closed-link reference analytically integrates over `s`
conditional on each sampled closed path and averages across all time links:

\[
 \widehat{n_k/N}\big|_Z = \frac{1}{MN}\sum_{t,i}
 \frac{e^{-\tau Dk^2}\cos(k\cdot\Delta_{t,i})}
 {V K_\tau(\Delta_{t,i})}.
\]

This second estimator may have very large variance when the periodic kernel
is tiny; agreement between estimators at short run lengths is not guaranteed.

### Moves

- A conditional Gibbs sector update chooses **Z or any `(t,i)` G state** with
  normalized probabilities `1 : eta/(MN V K_tau(Delta[t,i]))`. If G is
  selected, `s` is sampled from its exact conditional periodic free kernel.
- A conditional *G-only* Gibbs relocation redraws `(t,i,s)` given the G
  sector. It changes the time-cut index without touching the worldlines;
  acceptance is exactly one, with no empirical normalization.
- Reversible open/close Metropolis proposals use a uniformly proposed `(t,i)`
  and its exact conditional displacement. Their acceptance ratio contains
  `eta/(V K_tau(Delta))`: the `1/(MN)` target factor cancels the uniform
  proposal choice.
- The sector-collapsed free-boson refresh from Patch 18 now integrates over
  **all** possible `(t,i)` G sectors when evaluating its acceptance ratio.
- When closed, the sampler retains all Patch 17 closed-worldline moves.

### Winding and finite-size response

A wrapped periodic link is a mixture over integer image shifts:

\[
 K_\tau(\Delta)=\sum_{n\in\mathbb Z^3}
 (4\pi D\tau)^{-3/2}
 \exp[-|\Delta+nL|^2/(4D\tau)].
\]

For each axis `a`, conditional on the wrapped path, its *integer* image
shift has probability proportional to the corresponding Gaussian term.
The **global winding** is the sum over all links, including the permutation
closure. The raw wrapped link displacements telescope around the full
permuted collection of paths, so the sum of image integers is the global
winding `W_a`.

The conditional image shifts on different links are independent because the
primitive interaction action depends only on wrapped positions. Instead of
drawing a single noisy winding value, we integrate image moments exactly:

\[
 \mathrm E[W_a^2\mid X,P] =
  \sum_e \mathrm{Var}(n_{e,a}\mid X,P)
  + \left(\sum_e \mathrm E[n_{e,a}\mid X,P]\right)^2.
\]

The corresponding **finite-size winding response** for a rectangular torus is

\[
 R_a=\frac{L_a^2\langle W_a^2\rangle}{2 D\beta N},
 \qquad R_{\mathrm{iso}}=\tfrac13\sum_{a=1}^3 R_a.
\]

For isotropic cubic systems, this is the familiar winding-number estimator
of the superfluid *helicity response* (`rho_s/rho`) at fixed finite size and
inverse temperature. It does not establish thermodynamic superfluidity.
Only **closed Z-sector configurations** contribute winding measurements.
The separate `draw_winding` function can produce an integer topological
sample for independent checks, but the Rao–Blackwellized estimator normally
has lower noise.

### Independent free-boson reference

For an exchange cycle of length `l`, the canonical ideal gas contributes the
single-particle free trace `Z1(l beta)`. The trace's second derivative with
respect to a boundary twist gives its unnormalized winding-square moment
`Q1_a(l beta)`. The canonical recurrences are

\[
 Z_N=\frac1N\sum_{l=1}^N Z_1(l\beta) Z_{N-l},\quad Z_0=1,
\]
\[
 Q_{N,a}=\frac1N\sum_{l=1}^N
 \big[Q_{1,a}(l\beta)Z_{N-l}+Z_1(l\beta)Q_{N-l,a}\big],
 \quad Q_{0,a}=0,
\]

and `E[W_a²]=Q_N,a/Z_N`. This is an independent analytical check, and the
new tests also differentiate a reciprocal-space partition function under
a small boundary twist.

## Run

```bash
python3 -m pytest -q
python3 -m examples.bosons.run_mobile_worm --quick
python3 -m examples.bosons.run_mobile_worm
```

You may customize the output directory with `--output DIRECTORY`.
The example compares two particles with and without repulsive soft-core
interactions at fixed `beta=2.1`. Full mode compares `tau=0.35` and
`tau=0.2625` with independent seeds. It is a **benchmark**, not a completed
continuum/time-step extrapolation.

Output files:

- `replicas.csv`: raw per-seed worm, closed-link and winding estimates,
  sector visits and acceptance rates;
- `summary.csv`: means/replica-based standard errors; free reference;
- `time_cuts.csv`: how often each imaginary-time cut was selected;
- `protocol.json`: exact configuration and scope;
- `winding_response.png`, `coherence.png`, `cut_mobility.png`.

All error bars in the example are standard errors across **independently
seeded Markov chains**, not naive independent-bead uncertainties. In small
runs a finite-sample estimate of `n0/N` may exceed one; raw estimates are not
clipped. Strong autocorrelation, sector trapping, and rare small-kernel
configurations can make replica SEM optimistic.

## Next scientific steps

1. Independently separated imaginary-time head and tail (`G(r,tau)`), with
   explicit particle-number sectors and balanced advance/recede moves.
2. More efficient sector-balanced and cycle-changing updates; replicate
   convergence checks at fixed CPU budget.
3. Systematic `tau -> 0`, size/density and temperature studies, including
   independent winding/reference comparisons.
4. Only then consider joint characterization of spatial order, off-diagonal
   coherence, and superfluid response in the interacting continuum model.
