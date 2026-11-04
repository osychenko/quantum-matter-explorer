# Patch 32 — Educational engine v0.3 (limited MVP contract)

**Status: limited-scope MVP foundation, not general many-body QMC certification.**
This patch adds a stable JSON-serializable **contract v1.0.0** and a curated
experiment registry. The research samplers, actions, and tests are untouched.

## Allowed presets and certification boundary

| Preset | Availability | Meaning |
| --- | --- | --- |
| `ideal-boson-addition-v1` | `supported` | Analytic canonical ideal Bose-gas control (3D periodic, N=1 host, zero mode, midpoint). |
| `softcore-m8-baseline-recorded-v1` | `candidate_recording` | Historical Patch 30 baseline M=8 measurement, with independent-replica SEM, finite-basis spectral comparison, and Patch 31 screening label. **No new simulation is executed.** |

The **M=12 exchange** method is **excluded** from the registry after the Patch
31 gate found a significant discrepancy (~5.1 independent-replica SEM) against
its K=2 finite-basis spectral reference. Its status is a *research issue*, not
an educational simulation option. Similarly, our other many-particle VMC/DMC
methods have not been automatically certified by this contract.

## Commands

```bash
python3 -m pytest -q tests/test_educational_contract.py
python3 -m examples.bosons.run_educational_experiment list
python3 -m examples.bosons.run_educational_experiment run ideal-boson-addition-v1 --beta 2.4 --cell-length 2.8 --output outputs/education/ideal.json
python3 -m examples.bosons.run_educational_experiment run softcore-m8-baseline-recorded-v1 --output outputs/education/recorded_m8.json
```

## Public API schema

`schema_version`, `engine_release`, `experiment_id`, `preset_status`,
`execution_mode`, `parameters`, `observables`, `provenance`, `cautions`,
`fingerprint_sha256` are top-level fields. Observable entries include `id`,
`value`, `units`, `uncertainty`, `uncertainty_kind` and where available a
numerical/basis sensitivity indicator. The SHA-256 fingerprint of canonical
parameters and provenance is an **identity of the request definition**, not a
cryptographic signature of scientific correctness or an immutable dataset.

The analytic preset allows only `beta` in [0.5,5] and `cell_length` in [1.5,5].
The boson kinetic coefficient is fixed to 0.5, host N=1 and zero momentum
are fixed. The Green function is `1+1/Z1`, where `Z1` is the ideal
one-boson periodic partition sum, evaluated with momentum cube `[-10,10]^3`.
The difference from `[-8,8]^3` is a truncation indicator, **not a rigorous
error bound**. It is imaginary-time propagation, not classical trajectories.

Recorded M=8 benchmark values are sourced from the user's Patch 30 calculation:
`G = 1.685053810648142 ± 0.04352458506961887` (8 independent replicas),
with K=2 spectral reference `1.6910255956087066`. This result passed the
Patch 31 *screening* tests, but remains a `candidate_recording` because a
finite basis and finite imaginary-time resolution are distinct systematics.

## What is deliberately out of scope

- No public knob for generic Hamiltonians, arbitrary particle count, exchange
  tuning, number of MC steps, or unbounded multiprocessing.
- No claim that an M=8 recorded benchmark is an online or interactive QMC run.
- No validated many-particle phase diagram, real-time trajectories, universal
  BEC or superfluidity certificate.
- No web server or front-end framework is added: the contract is designed for
  Patch 33 to expose via a bounded async simulation API.

## Exit criteria for Patch 33

1. Exact allowed preset IDs and parameter ranges are exposed via `list`.
2. Same inputs produce identical JSON and provenance fingerprints.
3. Invalid, dangerous or research-only configurations are rejected.
4. Output names retain observable meaning, uncertainty type and caution.
5. Separate research commands continue to exist unchanged.
6. UI must visibly distinguish **analytic**, **recorded**, and any future
   **live Monte Carlo** output; certification is per preset, not per module.

## Follow-up research issue

Investigate M=12 replica-exchange nonstationarity / target-action consistency
against the independent baseline. This is tracked separately from MVP delivery
and is a release blocker only if that particular preset is proposed for use.
