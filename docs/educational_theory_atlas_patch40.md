# Patch 40 — Interactive Quantum Theory Atlas

Patch 40 extends the Patch 37 Theory Atlas into a separate navigable workspace,
linked to the existing Science Map and to the curated laboratories from Patches 38–39.
It does **not** add any executable research solver endpoints or widen the set of
permitted Monte Carlo jobs.

## The three live mathematical demonstrations

1. **Imaginary-time projection:** A normalized two-level state with energy gap
   Δ and a nonzero initial ground-state overlap. The excited-state weight obeys
   `p(τ) = p₀ exp(-2 Δ τ) / (1-p₀+p₀ exp(-2 Δ τ))`. Its decrease is an exact
   property of this *ground-state projector*. It is distinct from the finite-
   temperature imaginary-time worm measurement (thermal period β), and neither
   is a real-time trajectory.
2. **Variational principle:** For the *3D* harmonic oscillator
   `H=-∇²/2+ω²r²/2` and Gaussian `ψα∝exp(-αr²/2)`, the analytically integrated
   energy is `3(α+ω²/α)/4 ≥ 3ω/2`. The independent controls expose the
   difference between changing the variational trial width α and changing the
   physical Hamiltonian via ω. The exact local-energy standard deviation is
   `sqrt(3/8) |ω²-α²|/α`; it is **not** a sampled uncertainty or standard error.
3. **Boson exchange:** For **two noninteracting bosons in one dimension**, the
   two terms in the diagonal symmetrized density matrix have an exchange/
   identity ratio `exp[-d²/(2Dβ)]`, with D fixed to 0.5 in the UI. Its
   normalized *term share* is not a physical swap probability, not a measured
   worldline trajectory, and not evidence from the interacting boson simulator.

Each panel is explicitly marked as analytical or schematic. Read-only theory
notes originate in the existing `GET /api/v1/science-map` endpoint; the new
`imaginary-time-projection` note is additive. The frontend associates curated
experiments using `profiles[*].theory_ids`, rather than implying that all
methods or research-only modules have executable educational presets.

## Interface and fixes

- Separate **Theory Atlas** navigation entry and per-topic deep links from
  the Science Map; three selectable explanation depths remain.
- Topic-related buttons open curated laboratories and inherit their real
  numerical provenance.
- The comparison workspace no longer inadvertently renders the notebook below
  the Comparison Laboratory (a Patch 39 navigation conditional bug).
- The earlier `learningSteps()` unused-parameter TypeScript fix is already in
  the baseline commit `34b5590`; no second copy is included.

## Validation

From repository root (Patch 39 applied):

```bash
python3 -m pytest -q tests/test_educational_theory40.py \
  tests/test_educational_science.py tests/test_educational_comparisons39.py \
  tests/test_educational_comparison_api39.py
cd web
npm install
npm run test:theory
npm run test:missions
npm run build
```

Launch FastAPI as before, then `npm run dev` in `web/`; open **Theory Atlas**.
Use the Science Map to jump straight to individual topics.

These demonstrations are pedagogical *small-model identities*. A real VMC
estimate and its uncertainty must still be obtained from the existing
laboratory; the fixed recorded worm result remains a historical benchmark.
