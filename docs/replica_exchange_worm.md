# Patch 27 — Interaction-replica exchange for the continuum spacetime worm

This is an **experimental, exactly corrected mixing move**, not a new physical observable and not a guarantee of faster sampling. It builds on the normalized auxiliary-path worm in Patches 21–23 for **one free background boson** with an inserted interacting two-boson interval. It does not change the primitive action, the boson symmetry, the finite box, or the canonical particle-addition Green function.

## Why an interaction ladder?

The previous one-chain proposal families occasionally spend many iterations in high-weight worm configurations, especially at finer time discretization. Patch 27 adds a second chain with interaction amplitude `f × height`, where `0 <= f <= 1`, alongside the physical `height` chain. Each evolves with the same normalized free-worldline reference, fugacity, imaginary-time grid, and base sampler. The weaker replica can navigate different regions of configuration space, and the two chains exchange complete auxiliary configurations using a Metropolis swap. **Only the full-strength chain contributes to measurements.**

## Joint measure and exact swap acceptance

Let `s = (host, cut, length, extra, sector)` include the auxiliary path **even if the sector is closed**. The reference measure `q(s)` is shared by both replicas. Its sector-dependent target can be written as

\[
\pi_\lambda(s)\propto q(s)\begin{cases}
1, & s\in Z,\\
R_0(s)\exp[-\lambda U(s)], & s\in G.
\end{cases}
\]

Here `R_0` includes all the free-particle and Bose-exchange factors, as well as the known sector fugacity; `U(s)` is the *full-strength* primitive interaction action of the inserted pair, with half endpoint weights. Define `U(s)=0` in the closed sector. We simulate independent factors `pi_1(s_c) pi_f(s_h)`. Proposing to exchange the complete `s_c` and `s_h` is symmetric, and all shared `q` and `R_0` factors cancel:

\[
\log A=(1-f)\,[U(s_c)-U(s_h)],\qquad
P_{\rm accept}=\min(1,e^{\log A}).
\]

The physical replica and the auxiliary replica each first take their usual invariant worm step. With a *fixed, state-independent* probability, a complete state swap is then attempted. Every subkernel preserves the joint target. Swapping only the open/closed flags, the head/tail positions, or the extra path without the host/ghost variables would be **incorrect**; this patch swaps the complete state. Separate RNG instances are retained by each replica.

The parameter `hot_interaction_fraction` is **an interaction multiplier**, not a physical temperature. A value of 1 gives a control experiment in which every swap is accepted. With a zero pair potential, every swap is accepted as well.

## Estimators, diagnostics and validity

The physical Green function is measured using the *unchanged* Patch 23 ratio-block-jackknife audit on the **physical chain only**, for `G_(0,0,0)(beta/2)`. The other chain contributes only to computational expense and movement. The reported process CPU time includes **both worm chains and all swap attempts**.

Per-replica outputs include: Green estimate, time-block jackknife error, ratio-influence effective sample size, ratio ESS per CPU second, physical-chain closed/open visits, full-strength refresh acceptance, total swap acceptance, open-relevant swap acceptance, and swap counts. Between-independent-replica SEM is reported separately. `status=inconclusive` for quick runs and any detected low ESS or spectral discrepancy; `requires_long_chain_validation` otherwise. Neither label certifies convergence.

The `K=1` and `K=2` symmetric plane-wave spectral results are a separate independent **finite-basis** reference. Their difference is a basis-sensitivity diagnostic, not Monte Carlo uncertainty or an exact continuum error bound. Finite Trotter timestep, finite box, and finite simulation time remain independent systematic errors.

## Run

```bash
python3 -m pytest -q tests/test_replica_exchange_worm.py tests/test_replica_exchange_example.py
python3 -m examples.bosons.run_replica_exchange_worm --quick
python3 -m examples.bosons.run_replica_exchange_worm
python3 -m examples.bosons.run_replica_exchange_worm --links 4 6 --steps 5000 --replicas 6
python3 -m examples.bosons.run_replica_exchange_worm --links 4 6 --steps 5000 --replicas 6 --hot-fraction 0.5
```

The default output directory is `outputs/bosons/replica_exchange_worm/` with:

- `replicas.csv`: independent chains, timing and swap diagnostics
- `summary.csv`: across-chain means, SEM, ESS per CPU, warnings
- `spectral_reference.csv`: K=1 / K=2 benchmarks
- `protocol.json`: fixed experimental parameters
- `green_accuracy.png`, `cpu_efficiency.png`: numerical comparison

The optional `--height 0` control switches off the interaction and should yield 100% replica-swap acceptance. A priori tuning of `--hot-fraction` must be confined to a pilot; report separate **fresh-seed** validation for the final selection. Changing it after inspecting the test runs invalidates a held-out-performance claim.

## Results from the small development study

The quick 4-link comparison (420 steps, two independent replicas per method) estimated about 182 ESS/process-CPU-second for the single-chain baseline and about 84 for the two-chain replica-exchange method. This is a **loss of CPU efficiency**, not evidence of improvement.

In a separate 900-step smoke study, the two replica-exchange values at 6 links happened to cluster closer to the finite-basis Green reference than the two baseline values. With only two short chains, this observation is *inconclusive*: it could reflect chance and/or metastability in either method. A meaningful comparison needs longer independent replicas, stable block lengths, and accuracy measured at equal CPU cost.

## Limitations

- Only the current **N=1 -> N=2** physical worm, the isotropic finite-range soft-core repulsion, and a free background host are supported.
- This is a **finite-temperature canonical particle-addition** correlator, **not** a pure ground-state OBDM or direct superfluid fraction.
- Replica exchange does not solve rare-event bias, finite-time-step errors, or the problem of drawing enough independent samples from the worm sector.
- **Two replicas cost more CPU.** Exchange moves can improve decorrelation in some regimes and fail in others; numerical speedups have not been established.
