# Patch 36 — Onboarding, learning feedback and MVP playtesting

Patch 36 is a *product usability* release. The curated physics API, numerical
samplers, mission evidence validator, completion scoring and browser-only
storage contract are unchanged.

## Experience changes

- Mission field guide explains prediction, numerical evidence, interpretation,
  and reflection. The glossary defines beta, Green function, periodic cell,
  recorded benchmark, replica SEM and finite-basis spectral reference.
- A four-stage indicator guides each chapter. The scientific result remains
  concealed until a prediction is submitted. Feedback is derived **only** from
  the checked API observations and never from the user's guess.
- Evidence-collection network failures preserve the prediction and offer a
  concrete recovery hint. Chapter changes focus the chapter heading, and
  scroll without animation when reduced motion is requested.
- Progressbar semantics, explicit screen-reader feedback, keyboard focus styles
  and mobile layout are preserved or improved.
- Optional in-browser playtest questionnaire exports a JSON file; answers are
  not transmitted or persisted by the application. Do not enter identifying
  information. The existing expedition journal is still stored only in the
  browser and exported separately.

## Run

```bash
# Terminal 1, from repository root
python3 -m uvicorn app.api:app --host 127.0.0.1 --port 8000
# Terminal 2
cd web
npm install
npm run test:missions
npm run build
npm run dev
```

Open http://127.0.0.1:5173 and select **Research missions**. In remote
VS Code SSH, forward the Vite and FastAPI ports.

## Guided playtest protocol (15–20 minutes)

Recruit 3–5 volunteers with different physics backgrounds, ideally including
at least two beginners. Ask each to complete the first two missions without
verbal guidance, then try the recorded-benchmark and uncertainty missions.
Observe whether they can:

1. Identify what changes when beta grows (temperature goes down).
2. Predict before revealing the analytical control.
3. Explain why a 3D imaginary-time visualization is not real-time motion.
4. Tell recorded Monte Carlo data from a live calculation.
5. Explain what the ± uncertainty means without calling it proof.
6. Find the next chapter, retry a failed network request, and export notes.

Record completion/blockers and qualitative comments. Do **not** interpret
self-reported learning as measured learning gain; an independent pre/post
assessment would be necessary. No automated analytics or third-party tracking
is added. Test with keyboard only, a narrow viewport, reduced motion and
an intentionally stopped backend. For the public release consider a proper
screen-reader audit and browser compatibility tests.

## Scope and non-goals

The recorded interacting M=8 result remains a candidate historical benchmark,
not a newly computed Monte Carlo run or proof of continuum convergence.
M=12 exchange remains research-only. A prettier interface does not certify the
scientific engine. Patch 36 intentionally does not add live QMC jobs,
accounts, centralized telemetry, score-based physics judgments or new samplers.
