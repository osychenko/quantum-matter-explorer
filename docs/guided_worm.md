# Patch 25 — Interaction-guided insertion with **known proposal density**

Patch 25 adds an **experimental sampler**, not a new Hamiltonian or a change to the
Green-function definition. It builds on Patches 21–24 (`qmc/continuum_spacetime_worm.py`,
`qmc/reliable_spacetime_worm.py`, `qmc/worm_audit.py`, and `qmc/rare_worm.py`).
The supported physics remains canonical **one noninteracting background boson** in a 3D
periodic orthorhombic box, with up to two bosons in the open interval and the
existing symmetric primitive pair action. There is no larger-N continuum worm yet.

## Motivation

The normalized G-sector auxiliary proposal used previously samples an initial
inserted position uniformly from the simulation cell and each subsequent
position from the periodic free heat kernel. Interaction weights can then be
concentrated in relatively rare paths. Patch 24's multiple-try refresh respects
reversibility but pays for several full path evaluations.

The new proposal changes **where to propose**, while preserving the original
underlying probability distribution and the `G(k,t)` normalization.

## Normalized guide and mixture

For a fixed, independently sampled host path `h`, cut `c`, interval `ell`,
and inserted path `e_0,...,e_ell`, the free insertion proposal is

```
q0(e|h,c,ell) = (1/V) * product_{j=0}^{ell-1} K_tau(e_{j+1}-e_j).
```

The guide uses a weak repulsive tilt for the initial bead:

```
qg(e_0|h_c) = exp[-a * tau * V_pair(e_0-h_c)/2] / (V * Z_init),
Z_init = (1/V) int_cell exp[-a * tau * V_pair(r)/2] dr.
```

For the finite-support radial soft-core potential (cutoff no greater than the
cell inradius), translation invariance makes `Z_init` independent of `h_c`.
With `V(r)=height*(1-r²/rc²)^3_+`, it is a 1D radial integral:

```
Z_init = 1 - (4*pi/V) * int_0^rc r² [1-exp(-a*tau*V(r)/2)] dr.
```

We compute this with 64-point Gauss–Legendre quadrature. It should be checked
against higher-order quadrature for extremely sharp user-selected potentials.
The first coordinate is sampled *exactly from the normalized tilted law* by
rejection sampling from a uniform cell position, with acceptance
`exp[-a*tau*V/2]`. A finite iteration guard reports excessive rejection.

For every later bead, the guide uses a wrapped Gaussian kernel centered on
`e_j + d_j`, where `d_j` is a bounded outward force from the host at the
next time slice. The force uses the analytic soft-core pair gradient, evaluated
at the **previous** inserted coordinate. Every conditional transition is a
properly normalized periodic heat kernel:

```
qg(e|h,c,ell) = qg(e_0|h_c)
                 * product_j K_tau(e_{j+1}-e_j-d_j).
```

The final proposal is the *mixture*

```
qmix = (1-p) q0 + p qg,
```

with `0 <= p < 1`. For a sampled inserted path we evaluate **the full mixture
probability**, not just the conditional density of the component that happened
to generate it. Thus `q0/qmix <= 1/(1-p)` and the reference has full support.

## Correct acceptance and independent estimator

The original open target is `piG(h,c,ell,e) proportional to q_host*q_cut*q_length*q0(e)*R(h,c,ell,e)`.
A whole open-state proposal draws `q_host*q_cut*q_length*qmix(e)`.
Define `W = R * q0/qmix`, where `log R` is Patch 21's original
`log_relative_weight` and its existing `eta` normalization. The *exact*
independence-MH acceptance is

```
min(1, W(new)/W(old)).
```

Both forward and reverse proposal densities are included. **Omitting
`q0/qmix` would bias the worm-sector statistics.** The guided and existing
moves are mixed with state-independent probabilities, preserving reversibility.
The original conditional Gibbs-sector updates are unchanged.

An independent stratified check, for every separation `ell*tau`, evaluates

```
G_k(ell*tau) = E_qmix[(R/eta)*(q0/qmix)
                        * cos(k*(e_ell-e_0))].
```

This is an independent expectation (not a ratio of correlated Markov-chain
sums); it measures the same finite-`tau` Green function as previous patches.
Its standard errors can nevertheless miss rare configurations. `weight_ess`
reflects degeneracy among **positive importance weights**, whereas the
Markov-chain ratio ESS (Patch 23) reflects **temporal autocorrelation**.
Neither one alone proves convergence.

## Running

```bash
python3 -m pytest -q tests/test_guided_worm.py
python3 -m examples.bosons.run_guided_worm --quick
python3 -m examples.bosons.run_guided_worm
python3 -m examples.bosons.run_guided_worm --steps 3000 --replicas 6 --links 4 6 --iid-draws 1000
```

Under `outputs/bosons/guided_worm/` the example saves:

- `replicas.csv` — independent MCMC runs, block errors, ratio ESS, and CPU seconds;
- `summary.csv` — between-replica SEM and cautious comparison to the spectral reference;
- `independent_reference.csv` — free and guided importance-sampling estimates,
  positive-weight ESS, tail concentration, and CPU time;
- `spectral_reference.csv` — free analytic or independent truncated K=1/K=2
  interacting references, with basis difference;
- `green_accuracy.png`, `cpu_efficiency.png`, `importance_tails.png`,
  and `protocol.json` — figures and reproducible parameters.

The benchmark compares baseline, guided, and Patch 24's 4-candidate
multiple-try proposals on independently seeded chains. The guide's mixture
fraction is **fixed before comparison**, not optimized on the validation runs.
A quick run is a *smoke test*. No method is automatically called converged.

## Limits and caveats

- The guide is specialized to a **nonnegative soft-core pair potential** and
  exactly one background boson, matching the existing prototype's scope.
- The guide affects only a whole G-sector independent refresh; this is not a
  new general space-time worm implementation or a new thermodynamic ensemble.
- The symmetric primitive propagator has timestep bias. The spectral
  reference has plane-wave basis and quadrature bias. Both require separate
  convergence studies.
- Short correlated worm chains may have large and underestimated errors.
  Use independent replicas, jackknife block-length sensitivity, split-Rhat,
  and direct reference discrepancies as in Patch 23.
- A lower importance-weight tail concentration **does not imply** higher
  effective MCMC samples per CPU second. Additional guided density evaluations
  and rejection sampling can outweigh any mixing gains.
- The guide proposal is exactly reweighted *for the code's finite-action
  target*. It does not cure possible inconsistencies in the physical-model
  approximation or the original worldline-sector definition.
