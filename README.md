# QMC prototype refactor

A small modular VMC/DMC prototype extracted from the original monolithic script.

## Layout

- `qmc/potentials.py` — physical potentials only.
- `qmc/trials.py` — trial wavefunctions only.
- `qmc/evaluation.py` — validated potential/local-energy/trial evaluation.
- `qmc/vmc.py` — VMC sampler and generic one-parameter scan.
- `qmc/dmc.py` — importance-sampled fixed-population DMC.
- `qmc/stats.py` — block statistics.
- `qmc/resampling.py` — systematic resampling.
- `qmc/plotting.py` — headless PNG plots.
- `run_yukawa.py` — one experiment: optimize alpha with VMC, then run DMC.

## Run

```bash
python3 run_yukawa.py
```

Plots are written under `outputs/yukawa/`.

## Important physics note

The one-parameter `exp(-alpha*r)` ansatz generally gives a VMC optimum with
`alpha != g`, which violates the Coulomb-like cusp condition of the Yukawa
potential at the origin. It is fine as a first pipeline test, but the next
upgrade should be a cusp-preserving multi-parameter trial such as
`YukawaCuspTrial`.
