# Patch 34 — Interactive periodic-cell laboratory (educational visualization)

Patch 34 upgrades the Patch 33 React interface to a Three.js scene without
modifying `qmc` solvers, the analytical result, or the curated API.

## Run

After applying the patch, from `web/`, run `npm install` then `npm run build`
(or `npm run dev` for development). In another terminal, start the same Patch 33
FastAPI backend (`python3 -m uvicorn app.api:app --host 127.0.0.1 --port 8000`).

## What is real and what is illustrative

* **Ideal one-particle host:** turquoise sphere is a seeded independent draw
  uniformly distributed over a periodic unit cell. That one-body position
  marginal is exact for the canonical ideal N=1 host. It is **not** a sampled
  interacting N=2 configuration or a Green-function path.
* **Recorded interacting benchmark:** the two sphere positions are strictly
  illustrative and **not** drawn from stored Monte Carlo coordinates. The
  recorded result/uncertainty in the main panel are unchanged.
* **Imaginary-time exchange:** colored curves constitute a hand-authored diagram
  of exchanging endpoints, **not** sampled paths. Vertical direction indicates
  an abstract imaginary-time parameter; the diagram is not a real-time movie.
* **Neighboring periodic images:** ghost particles are copies of the ideal host
  at unit lattice translations. They are not additional particles.
* **Physical cell length:** the wireframe displays a normalized cell. A displayed
  L updates with the selected ideal control, and the engine continues to compute
  the actual L-dependent observable; screen-space dimensions are intentionally
  normalized for legibility.

The scene responds to pointer drag (orbit), wheel/pinch (zoom), new seeded
uniform draw, and a checkbox for neighboring periodic images. It disposes GPU
resources and disconnects its resize observer on unmount; it gracefully reports
WebGL unavailability without preventing use of the rest of the application.

## Scientific scope and next milestones

No numerical physics is added or changed in Patch 34. Do **not** interpret the
scene as real-time evolution, evidence for Bose condensation, or a measured
interacting quantum configuration. If future patches add true configuration or
worldline sampling, introduce separately typed/provenanced scene data and
validate backend parameters before relabeling a visualization as a measurement.

Patch 35 can add hypothesis → simulation → interpretation learning missions,
using the existing structured experimental results and clearly labeled scenes.
