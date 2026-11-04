# Patch 37 — Science Map and transparent experiment explanations

## Why this patch exists

The first prototype could run a small set of trusted calculations and gave the
player missions and a 3D illustration, but it did not make the physics research
question, Hamiltonian, numerical method, recorded settings, observable, and
valid comparison explicit. Patch 37 makes those relationships first-class,
**without adding another experimental solver to the public API**.

## Navigation

The app now opens on **Science Map**, where each card starts with a question.
It presents two curated exercises (a live analytical ideal-boson control and a
fixed, historical M=8 worm benchmark), alongside three clearly labeled
research directions (ground-state energy, spatial order and boson exchange).
Research-only cards **have no executable preset ID**, and the API continues to
reject unrestricted QMC jobs. The Science Map also includes expandable method
explanations and physics at three depths: Concept, Physics, Mathematics.

The laboratory includes the same five-part structure for the selected preset:

1. Physical question and intended model;
2. Calculational method and its limitations;
3. Parameters grouped as physical, numerical, statistical and measurement;
4. Precisely defined Green-function observable;
5. Comparison protocol: held fixed, changed, and what a difference means.

All numeric values in the inspector are read from that run's **actual JSON
response**. `qmc/educational_science.py` describes names and interpretations
through `parameters.*`, `provenance.*` and `observables.*` paths. It does not
copy user-adjustable numeric values into separate frontend constants.
`validate_science_map()` checks that these paths resolve in both curated
responses and that research-only cards cannot expose runnable presets.

## Three comparisons, three different scientific questions

**Ideal analytical parameter sweep:** Same N=1 free-boson Hamiltonian, same
zero-momentum addition observable, one control (`beta` or `cell_length`)
varied while the other is fixed. The partition function is a finite momentum
sum, with a last-shell sensitivity diagnostic; it is not a Monte Carlo SEM.

**Interacting worm vs spectral K=2:** Same interacting Hamiltonian, beta, cell
and target observable, but different numerical approximations. Worm uses M=8
primitive time links; the spectrum is truncated to a finite plane-wave basis.
Their numerical agreement is a useful cross-check, **not** proof of zero
time-step or basis truncation errors.

**Matched ideal vs recorded interacting case:** The application separately
calculates the ideal control with the recorded `beta` and `cell_length`, so
physical D, N and the zero-momentum observable are matched. Interaction
strength differs, **and** numerical method and discretization differ.
Therefore, this comparison alone cannot isolate the quantitative effect of
repulsion. The UI says this explicitly.

## Reproducibility details (historical Patch 30 M=8 baseline)

The educational recording now exposes fixed settings already used in the
previously supplied experiment:

- beta=2.4, cubic L=2.8, kinetic D=0.5;
- background N=1 and added N+1=2 sector, zero momentum at beta/2;
- soft-core potential height 3.0 and radial cutoff 1.2;
- M=8 links, 30,689 steps per chain, 400 warm-up steps, stride 2;
- worm-sector fugacity eta=0.9 (a sampling convention, **not** chemical potential);
- 8 held-out seed IDs 2380000...2380007;
- per-chain CPU target 45 s; observed mean CPU ~52.65 s;
- Green function 1.685053810648142, independent-replica SEM 0.04352458506961887;
- K=2 finite-basis spectral result 1.6910255956087066.

Those are historical inputs and measurements, not a live regenerated result.
The ideal analytical JSON includes the evaluated mode cutoff (10) and the
comparison cutoff (8); shell sensitivity is not a rigorous upper bound.

## Files and entry points

- `qmc/educational_science.py`: curated read-only science questions, methods,
  three-layer theory, profiles and parameter-path validation.
- `app/api.py`: new `GET /api/v1/science-map` endpoint (no new POST execution).
- `qmc/educational_contract.py`: additive fixed-parameter/provenance metadata.
- `web/src/ScienceMap.tsx`: Science Map, five-step experiment summary, comparison
  panel, method steps and live parameter inspector.
- `web/src/App.tsx`, `web/src/style.css`: integration, navigation and styling.
- `tests/test_educational_science.py`: scope, paths, provenance and regression
  tests.

## Run

From repository root:

```bash
python3 -m pytest -q tests/test_educational_science.py tests/test_educational_contract.py tests/test_educational_api.py
python3 -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

In a second terminal:

```bash
cd web
npm install
npm run test:missions
npm run build
npm run dev
```

Open `http://127.0.0.1:5173`. Science Map is the landing page. The dev
servers remain localhost-only; use SSH/VS Code port forwarding if remote.

## Status and next patch

This is a navigation/interpretation milestone, **not** physics certification.
Patch 38 can add curated, independently tested harmonic-oscillator and
structural-observable experiments. Until validation is complete, their cards
remain research-only and inaccessible through the educational execution API.
