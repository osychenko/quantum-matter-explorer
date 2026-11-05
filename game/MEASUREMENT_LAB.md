# G6 — Quantum Measurement Laboratory

The laboratory is a teaching activity tied to the two existing rectangular
quantum barriers. It is **not a new gameplay ability** and does not require a
relic or change any existing save format.

## Play

1. Move within tunneling range of either blue barrier and press **M**.
2. Choose sub-barrier incident energy `E` and barrier width `a` in the panel.
   `V0` (barrier height) and `D` (kinetic coefficient) come from the chosen barrier.
3. Run 1, 10, 100, or 1,000 independent transmission/reflection trials.
4. Compare the theoretical probability `T` with the observed frequency.
   The panel also displays theoretical sampling standard error `sqrt(T*(1-T)/N)`.
5. Change `E`, `a`, or the random seed to start a fresh experimental series.
   **Clear results** resets the counts, and **Barrier defaults** restores E/a.
6. Press **M**, **Esc**, or the **Close** button to return to the platformer.

While the panel is open, player physics controls are suspended. The experiment
spends **no Coherence** and does not mutate barrier attributes or saved progress.
Each batch uses the input seed plus a batch index; resetting results repeats
exactly the same seeded sequence for identical settings.

## Scientific scope

`TunnelingModel.gd` evaluates an exact stationary 1D rectangular potential
barrier for `0 < E < V0`. `MeasurementExperiment.gd` samples independent
Bernoulli trials with the calculated probability `T`. This is an instructional
Monte Carlo *sampling demonstration*, **not quantum Monte Carlo (QMC)**.
It does not simulate wavepacket propagation or collapse dynamics.

## Automated checks

```bash
godot --headless --path game --editor --import
for test in test_tunneling test_demo_scene test_progression test_echoes test_realm test_measurement test_measurement_lab; do
    godot --headless --path game --script "res://tests/${test}.gd" || exit 1
done
```

These tests use a fresh scene with persistence disabled. They never touch the
real user's saved progression.
