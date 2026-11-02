# Patch 28.1 — parallel independent worm benchmark replicas

**Scope:** Extend Patch 28's benchmark runner, not the physical model or the
worm transition kernels. New option `--workers` uses `ProcessPoolExecutor` and
one **separate process** per independent validation chain. The pilot still runs
**sequentially**; workers never reuse an earlier replica's RNG state. The
result order and Patch 28 seed formula are preserved exactly. The same fixed
calibration plan and seeds must give the same observables with 1 or 6 workers,
though measured CPU times will differ.

## Hardware: Ryzen 9 5900HX, 8 CPUs, 15 GiB RAM

Reserve capacity for the user's MinIO, Streamlit and VS Code services. Use
**6 workers** initially; scale down to 4 if the system becomes sluggish, or
test 8 with care. This is CPU/process parallelism, not GPU acceleration. Set
BLAS thread counts before Python starts to avoid oversubscription:

```bash
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

### Smoke test

```bash
python3 -m pytest -q tests/test_equal_cpu_parallel.py
python3 -m examples.bosons.run_equal_cpu_worm --quick --workers 2 \
  --output outputs/bosons/equal_cpu_smoke_281
```

### Main experiment

```bash
python3 -m examples.bosons.run_equal_cpu_worm \
  --links 4 6 8 --cpu-seconds 20 --replicas 12 \
  --workers 6 --pilot-steps 600 \
  --output outputs/bosons/equal_cpu_parallel_281
```

Every validation chain has the **same number of steps as its serial counterpart**
when given the same plan; the CPU calibration itself uses elapsed process CPU
and can vary across independent benchmark invocations. Thus comparisons of
`--workers 1` and `--workers 6` should ideally reuse the same precomputed plan
(not recalibrate separately) when testing numerical equivalence.

## Calibration improvement

Patch 28's default 40-step pilots only timed raw `step()` calls and sometimes
predicted validation cost inaccurately (`cpu_budget_mismatch`). Patch 28.1 uses
an independent **400-step production-style pilot** by default (`--pilot-steps
600` recommended for the longer experiment), including the same ratio audit
and thinning work as validation. For very short pilots (<160), such as `--quick`,
the original raw-step calibration is retained. This **may reduce** CPU-budget
mismatch but does not force CPU time to equal the requested target: chain
residence and CPU contention still vary, and minimum/maximum step caps apply.

## Output and interpretation

As before, `replicas.csv` contains per-chain `cpu_seconds`, `cpu_ratio`,
`ratio_ess`, `max_abs_influence_fraction` and `block_stability`.
`summary.csv` retains the `flags` and conservative `status` screening.
`protocol.json` now records `workers`, `pilot_steps`, total wall-clock seconds,
the sum of validation **process** CPU seconds, and the sum of pilot CPU seconds.
Wall time measures how quickly an experiment finishes, while sum-of-CPU-time
accounts for computational effort. They must not be interchanged.

Run the calibration and each validation experiment using its own output path;
**never concurrently write to the same output directory**. Independent replicas
are not shared memory or threads. The spectral reference remains the K=1
**truncated-basis** reference, not an exact 3D interacting continuum answer.
The complete benchmark still does not certify Monte Carlo or time-step
convergence.
