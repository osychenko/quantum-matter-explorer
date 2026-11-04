# Patch 39 — Scientific Comparison Laboratory

**Goal:** replace informal comparisons with curated two-arm investigations that
state the model, observables, numerical methods, controlled variables and
limitations. The Comparison Laboratory is available from the application sidebar
next to the Science Map. It does **not** add a new numerical sampler.

## Available studies

| ID | Arms | What is actually controlled | What it establishes |
| --- | --- | --- | --- |
| `oscillator-trial-choice-v1` | Two live Gaussian VMC trials | Same 3D harmonic Hamiltonian and numerical settings; Gaussian width α varies (ω shared) | A controlled *trial-state* change; VMC values can be checked against the exact Gaussian expectations and exact ground energy |
| `softcore-jastrow-choice-v1` | Two live N=6 boson VMC trials | Same fixed soft-core interaction, box, particle count, VMC configuration, radial bins and reciprocal directions; Jastrow amplitude varies | Trial-state-dependent energy, g(r) and directional S(k); **not** an interaction-strength scan |
| `ideal-vs-interacting-v1` | Analytical ideal Bose gas and recorded M=8 interacting worm | Same β, L, momentum mode, imaginary-time midpoint and N=1→2 observable | An **illustrative confounded contrast**: both the physical potential and numerical approach differ |
| `worm-vs-spectral-v1` | Recorded M=8 worm value and its stored K=2 spectral reference | Nominal interacting physical model and zero-momentum particle-addition observable | A numerical consistency screen, not a continuum/finite-basis convergence proof |

The two live studies run existing Patch 38 VMC implementations **twice**, in
sequence. They use the existing fixed reproducible seeds and a shared, bounded
server-side VMC semaphore. Do not interpret the two arm estimates as independent:
matching seed identifiers may induce nonzero covariance, which the current
Patch 38 result contract does not report. We therefore show **individual
replica SEMs** and a **descriptive difference** but no combined SEM, z-score,
confidence interval or claim of significance for the difference. Even apparent
agreement of a VMC estimate with an exact reference is not proof of general
algorithmic convergence.

Structural distributions are checked for matching radial bins and reciprocal
indices before plotting an overlay. The S(k) values belong to *distinct
reciprocal directions* (displayed as points, not as a continuous curve).
`g(r)=1` and `S(k)=1` are reference expectations for independent uniform
positions, not exact interacting ground-state predictions. The plotted
pointwise SEMs are not simultaneous confidence bands. Short three-replica
experiments cannot establish a liquid/solid transition.

For the worm/spectral comparison, K=2 is a finite-basis Galerkin approximation
and M=8 is a finite-time-step worm calculation. Their systematic errors are
not included in the replica SEM. The spectral value is read from the recorded
benchmark field and **not** recomputed or presented as a separate simulation.

## JSON API

`GET /api/v1/comparisons` lists the four curated study definitions, adjustable
settings, scientifically bounded ranges, defaults and metadata.

`POST /api/v1/comparisons/{id}/run` accepts:

```json
{"parameters":{"alpha_left":0.7,"alpha_right":1.3,"omega":1.0}}
```

The response contains:

- `comparison_schema_version`, `comparison_id`, `comparison_kind`, `question`;
- `arms` (two original, full versioned experiment responses and the primary
  comparable observable; a derived reference arm is explicitly labeled);
- `controls.held_fixed`, `.changed`, `.confounders`;
- `analysis.difference_b_minus_a`, `combined_sem: null`, `difference_z: null`,
  and the rationale for not reporting a significance test;
- `interpretation.supported_conclusions` and `.not_supported`;
- `fingerprint_sha256` of the full study definition and the two evidence hashes.

Unknown study IDs, arbitrary extra parameters, out-of-range/non-finite values,
identical trial-arm settings, mismatched physical controls or structural grids
are rejected. The endpoint does not provide an arbitrary experiment runner.
Concurrent live VMC jobs are rejected with HTTP 429 instead of oversubscribing
the local machine. Analytical and recorded comparisons are inexpensive and do
not occupy the live VMC slot. No accounts, persistent uploads or external
telemetry are added. The frontend exports the entire comparison as local JSON.

## Development checks

```bash
python3 -m pytest -q \
  tests/test_educational_comparisons39.py \
  tests/test_educational_comparison_api39.py \
  tests/test_educational_labs38_integration.py \
  tests/test_educational_contract.py \
  tests/test_educational_api.py
cd web
npm run test:missions
npm run build
```

The two original real-VMC integration tests are especially important because
they execute the research modules, rather than injected numerical results.
On the local development host, verify an oscillator comparison and structural
overlay by hand, check the "held fixed" panels against both result payloads,
and test changing parameters mid-calculation and running a second comparison
while the first is busy.
