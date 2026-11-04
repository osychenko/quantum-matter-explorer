# Patch 35 — Quantum Field School (educational missions)

Patch 35 turns the local-first React interface from Patch 34 into a **guided scientific expedition**, without expanding or altering the physics engine. The mission logic is separate from the numerical API, and all evidence is collected from the **existing Patch 32/33 allowlisted endpoints**.

## First expedition: four missions

| Chapter | Mission | Evidence | Learning outcome |
| --- | --- | --- | --- |
| 1 | The Thermal Detective | Two **live analytical** ideal-boson evaluations with different `beta`, fixed box | Thermal occupation and the imaginary-time particle-addition Green function |
| 2 | The Expanding Box | Two **live analytical** ideal-boson evaluations with different `cell_length`, fixed `beta` | Allowed momenta and the one-particle thermal partition function |
| 3 | The Repulsion Case File | Ideal analytical control plus **historical recorded** M=8 interacting result | A numerical difference does not alone isolate the effect of interaction or establish a phase transition |
| 4 | The Uncertainty Investigator | **Historical recorded** M=8 interacting result with independent-replica SEM and finite-basis spectral reference | The meaning and limitations of agreement within two reported SEMs |

### Science-first game loop

1. Select a prediction **before** revealing the experiment.
2. Collect evidence from the curated API. Analytical controls are computed live; recorded benchmarks are retrieved, not simulated again.
3. Interpret the observation and optionally write a research notebook note.
4. Receive the physically grounded explanation, including caveats. Both confirmed and contradicted hypotheses receive the **same 40 XP** for completing the process. Answers are explained, not scored for correctness.
5. Unlock the next chapter and accumulate educational badges.

The available badges are *Hypothesis Maker*, *Evidence Collector*, *Model Guardian*, *Careful Skeptic* and *Quantum Explorer*. Four completed chapters yield 160 XP. Replaying a mission **never awards XP twice**.

The progression is **not scientific certification**. In particular:
- The ideal-gas midpoint addition correlator is a canonical ideal **N=1→2** analytical control with finite momentum-sum truncation.
- The interacting **M=8** baseline observation remains a *candidate historical recording*, with independent-replica uncertainty and finite-time-step limitations. No live interacting sampler is run.
- The worldline animation from Patch 34 is a graphic explanatory schematic, **not a physical trajectory or measured QMC path**.
- The fact that the M=8 result lies near the K=2 spectral reference does **not** prove continuum convergence.
- Problematic research-only presets (e.g. M=12 exchange) are not exposed in missions.

## Start

Requires **Patch 34**, including its 3D interface, and the Patch 33 backend.

```bash
cd ~/quantum-matter-explorer
# Backend — in terminal 1
python3 -m pip install -e '.[web-test]'
python3 -m uvicorn app.api:app --host 127.0.0.1 --port 8000

# Frontend — in terminal 2
cd ~/quantum-matter-explorer/web
npm install
npm run test:missions
npm run build
npm run dev
```

Open http://127.0.0.1:5173 and select **Research missions** in the sidebar. The original laboratory, 3D views, notebook and API remain accessible. For VS Code Remote-SSH forward ports **5173 and 8000**; neither server should be public without a separate production deployment.

## Storage, reproducibility and privacy

Completion records are stored in this browser's `localStorage` under `qme-missions-v1` (separate from the Patch 33 notebook). The store is versioned, accepts only recognized mission IDs, and is bounded to the number of missions. Each record retains prediction/interpretation IDs, whether the observation supported the prediction, a short optional research note, and **engine-generated experiment fingerprints** for provenance. It does not retain full result objects or send progress to the backend.

Click **Export expedition journal** to save a JSON copy of your progress. This preview has **no import feature, cloud accounts, synchronization or anti-cheating system**. Clearing browser storage clears progress; for unreliable/incognito storage the interface warns you to export. Research notes remain entirely local unless the user exports and shares them.

The physical output stays the source of truth: results and their status are validated against the expected allowlisted experiment ID, execution mode, parameter values, uncertainty kind and response schema before mission progression. This is a local educational consistency check, not authentication or an integrity proof against a modified API.

## Implementation

- `web/src/missionEngine.ts`: pure mission definitions, evidence verification, finding calculations, bounded progress validation, one-time XP, unlocks and badges.
- `web/src/MissionBoard.tsx`: accessible predict → observe → interpret → reflect UI, API evidence fetches, mission journal export.
- `web/src/missions.css`: responsive styling; integrates with the existing Patch 34 dark laboratory layout.
- `web/src/App.tsx`: a new sidebar destination and entry link; keeps laboratory and notebook unchanged.
- `web/tests/mission-engine.test.mjs`: Node built-in test runner with TypeScript compilation. No new dependencies.

Tests (run from `web/`):

```bash
npm run test:missions
npm run build
```

The mission tests check model restrictions, real-data-derived predictions, reference discrepancy in SEM units, duplicated XP, unlock rules, storage validation and bad input handling. The existing backend tests can be run separately:

```bash
python3 -m pytest -q tests/test_educational_api.py tests/test_educational_contract.py
```

## Suggested Patch 36: playtesting, not a new solver

Run the first expedition with 3–5 novice users. Observe whether they notice the difference between a calculated analytical model, a recorded numerical result and an illustrative visual. Improve explanations, navigation, keyboard accessibility, and error handling based on observations. Only then consider additional missions, user profiles, or live bounded many-body runs.
