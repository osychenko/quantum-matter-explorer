# Patch 38 — Two live, bounded physics laboratories

Patch 38 expands the **Patch 37 Science Map** without upgrading any research-only
many-body algorithm to a certified general-purpose engine. The two new results
are actual runs of `qmc.vmc.run_vmc`; their plots use `qmc.observables` and
analytically justified reference quantities. They are **not** recorded outputs.

## 1. Ground-state energy — 3D harmonic oscillator

**Question.** What is the cost of choosing an imperfect trial wavefunction?

Physical model (reduced units):

    H = −(1/2) ∇² + (1/2) ω² r²   (one particle, three dimensions)
    Ψ_α(r) ∝ exp(−α r²/2)

The exact ground-state energy is `E0 = 3 ω / 2`. For the *normalized Gaussian
trial*, the exact expectation is `E_trial(α,ω) = 3(α + ω²/α)/4`. This obeys
`E_trial >= E0` and equals the ground energy at `α=ω`. These are two different
analytical references: the first is exact ground-state physics, the second is
an exact integral of the chosen trial wavefunction.

For the numerical point, Metropolis VMC samples `|Ψ_α|²`, computes the local
energy and averages it across three independently seeded runs. The plotted
vertical error bar is the across-replica SEM, which estimates *sampling* noise
but not the trial-state bias or an upper-bound confidence interval. A finite
VMC estimate can fluctuate below `E0`; the theoretical inequality applies to
an exact expectation, not each Monte Carlo sample.

**Interactive parameters:** `trial_alpha` (0.5–1.8), `omega` (0.7–1.5).
**Fixed numerical settings:** `D=0.5`, 3 replicas (seeds 380010–380012), 128
walkers, 260 steps, 60 burn-in, stride 2, proposal σ=0.75, adaptation off.
The run is short and meant to show the variational principle, not optimized
uncertainty estimation. No DMC or oscillator thermodynamics are implied.

## 2. Structural correlations — six bosons, periodic box

**Question.** How do trial-state spatial correlations appear in real space and
Fourier space?

Physical model (fixed for this lab):

    H = −D Σᵢ ∇ᵢ² + Σᵢ<ⱼ V0 (max(1−rij²/rc²,0))³
    N=6; D=0.5; L=5; V0=3; rc=1.2
    Ψ_a(R) = exp(−a Σᵢ<ⱼ (max(1−rij²/rc²,0))³)

All pair distances use the periodic minimum image. **The slider changes the
*trial Jastrow amplitude* `a` (0–1.2); it does not change the Hamiltonian or
interaction height.** At `a=0` the trial distribution is uniform and has
`g(r)=1` for full spherical shells and `S(k)=1` at nonzero reciprocal vectors
in expectation. These are positional *ideal-reference benchmarks*, not an
exact ground-state solution for the interacting Hamiltonian.

`PairCorrelation` measures the full-shell radial `g(r)` on 12 bins through
`r=1.5` (inside `L/2`). `StaticStructureFactor` measures *directional* modes
`(1,0,0)`, `(0,1,0)`, `(0,0,1)`, `(1,1,0)`. The interface shows distinct
points for `S(k)`; they are not connected or radially averaged. Three
independent seeded runs (380110–380112) give pointwise replica SEM. These
short, finite-N trial-state results must not be interpreted as a phase boundary,
true condensation, superfluidity or exact interacting ground-state correlations.

**Fixed sampler settings:** 48 walkers, 150 particle sweeps (40 warm-up),
measurement stride 5, proposal σ=0.55, non-adaptive, independent replicas.
The numerical engine uses the existing particle-move and structural accumulator
code and reports mean variational energy per particle.

## Safe API boundaries

The same `POST /api/v1/experiments/{id}/run` endpoint now allows:

- `harmonic-oscillator-vmc-v1` with only `trial_alpha` and `omega`.
- `softcore-structure-vmc-v1` with only `jastrow_amplitude`.

All other numerical settings and seed lists are fixed, not user-controllable.
New results have execution mode `bounded_live_vmc` and status
`educational_variational`; they are not marked as certified ground-state data.
The structure result also has a small, bounded `distributions` array with
per-point SEM and explicit reciprocal indices.

A process-local single-slot semaphore rejects overlapping live VMC requests
with HTTP 429 rather than queueing unlimited expensive work. **The API is
still a localhost developer tool.** This semaphore is not a multi-worker or
multi-process production scheduler and should not be relied on to secure a
public deployment. Arbitrary research solvers are not exposed.

## Run locally

```bash
python3 -m pip install -e '.[web-test]'
python3 -m pytest -q tests/test_educational_labs38.py \
    tests/test_educational_labs38_integration.py \
    tests/test_educational_contract.py \
    tests/test_educational_science.py \
    tests/test_educational_api.py
python3 -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
cd web
npm install
npm run test:missions
npm run build
npm run dev
```

Visit `http://127.0.0.1:5173`, open **Science Map**, and choose the ground-state
or structural-observables problem. The new labs require clicking *Run Monte
Carlo experiment* rather than launching jobs on every slider movement.

### API quick checks

```bash
curl -s http://127.0.0.1:8000/api/v1/science-map | python3 -m json.tool | head
curl -s -X POST 'http://127.0.0.1:8000/api/v1/experiments/harmonic-oscillator-vmc-v1/run' \
  -H 'Content-Type: application/json' \
  -d '{"parameters":{"trial_alpha":1.3,"omega":1.0}}' | python3 -m json.tool
curl -s -X POST 'http://127.0.0.1:8000/api/v1/experiments/softcore-structure-vmc-v1/run' \
  -H 'Content-Type: application/json' \
  -d '{"parameters":{"jastrow_amplitude":0.8}}' | python3 -m json.tool
```

## Starlette warning noted in Patch 37

The warning `Using httpx with starlette.testclient is deprecated; install
httpx2 instead` concerns **test tooling**, not the numerical engine. Starlette
now prefers `httpx2` for its `TestClient`. Patch 38 includes `httpx2` in the
`web-test` optional dependency group; reinstalling that extra in the project's
same active Python environment should remove the specific warning. There is no
blanket warning suppression or forced change to production HTTP clients.

## Next steps

Patch 39 will build a **Comparison Laboratory** that persists pairs of runs
with explicit fields for *held fixed*, *changed*, *reference*, *uncertainty*
and *what conclusion is justified*; it should not treat unlike Hamiltonians
as a simple interaction-only difference.
