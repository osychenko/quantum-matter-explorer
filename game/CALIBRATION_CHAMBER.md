# Phase G7 — Calibration Chamber

This is the **second playable room**, reached by pressing **P** after clearing
Barrier Ruins (G5/G6). Its level is `scenes/CalibrationChamber.tscn`; you can
open that scene directly in the Godot editor to test it without completing G5.
Press **B** to return to Barrier Ruins.

## Puzzle

1. Walk to the blue calibration console and press **M**.
2. Adjust Energy **E** and Width **a** until the theoretical transmission
   probability is **40%–50%**. For the provided console, try `E=0.8`, `a=1.0`.
3. Click **Run 10** to record a small sample and its model-based standard error.
4. Without changing parameters, click **Run 1000**. The expected sampling error
   shrinks as `sqrt(T(1-T)/N)`, and the blue gate unlocks permanently.
5. Close the lab with **M** or **Esc**. Walk through the gate, touch the
   checkpoint, jump across the hazard, and reach the green exit.

The gate **does not judge the observed frequency**, which fluctuates randomly.
It judges the analytical model's parameters, reproducible experimental protocol,
and smaller theoretical sampling uncertainty. This prevents luck-based gating.

These trials are independent Bernoulli samples of a **stationary rectangular
barrier's analytical transmission coefficient**. This is a Monte Carlo teaching
experiment, **not** QMC, quantum measurement of an actual moving 2D particle,
or a numerical path-integral/DMC simulation.

## Editable geometry

In the Godot 2D editor, open `scenes/CalibrationChamber.tscn` and select `Ground`,
`RightLedge`, `CalibrationGate`, or `CalibrationConsole` in the Scene tree.
Drag with the move tool (W) or edit Inspector > Transform > Position.
`LevelPlatform.gd` makes physics collision and drawings follow the same
`Dimensions` property, as in the original room.

## Persistence

G7 gate/room completion are optional fields in the existing v1 save JSON.
G3–G6 saves load unchanged. Completed calibration is not reset by a parameter
change. Use the original room inventory's **Shift+N** to reset all progression.

## Tests

```bash
godot --headless --path game --editor --import
godot --headless --path game --script res://tests/test_calibration.gd
godot --headless --path game --script res://tests/test_progression.gd
```
