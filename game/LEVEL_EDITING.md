# Quantum Matter Explorer — G5: editing levels in Godot

Open `game/project.godot` using Godot 4.x, then open `res://scenes/DemoLevel.tscn`.

## Move the right ledge (the example that motivated this refactor)

1. In **Scene**, select `RightLedge` (a `StaticBody2D`).
2. Press **W** in the 2D view to use the Move tool, and drag it up/down; or edit **Inspector → Transform → Position**.
3. Larger **Y** means lower on the screen. G5 starts with `RightLedge` at `(1400, 585)` (50 px lower than G4).
4. **Ctrl+S**, then **F5** to play.

The visible ledge and its `CollisionShape2D` are children of the same transform, so both move together. Change **Inspector → Dimensions** on the `LevelPlatform.gd` node to resize its visible rectangle **and** collider simultaneously.

## What you can move directly

- `Ground`, `LeftBoundary`, `RightBoundary`, `LeftLedge`, `RightLedge`, `SpectralBridge`, `EchoGate`, `RealmGate`: `LevelPlatform.gd` (`@tool`, editable Dimensions/Fill Color/Edge Color).
- `AbilityRelic`, `EchoRelic`, `ImaginaryRelic`, `Checkpoint`, `RealmCheckpoint`, `EchoSwitch`, `EchoMilestone`, `Hazard`, `RealmHazard`, `Goal`: `WorldMarker.gd` (`@tool`, visual follows node transform).
- `Barrier` and `Barrier2`: already draw their artwork relative to their own physics node (`TunnelBarrier.gd`).

**Important:** Moving a marker changes its visible art and physics trigger together. For rescaling markers, edit the `CollisionShape2D` shape and check the visual style. `WorldMarker.gd` currently auto-reads the hazard collision width; other icons have fixed local sizes. Do **not** move `CollisionShape2D` separately from the parent unless you deliberately want an offset.

`DemoLevel.gd` now draws only decorative background. Player, HUD, camera, triggers, and gate logic are still handled by scripts.

## Level scripting vs. editing

The G5 initial coordinates and trigger order are set by the scene. Most trigger logic is spatially agnostic. A few gameplay rules still use X-coordinates (HUD hints and science-panel switching), while respawn destinations are named constants in `DemoLevel.gd`; update those if you relocate an entire section or a checkpoint. The camera right limit and world-width background remain 2880 until you expand the level.

## G5 gameplay

After tunneling (E) and the Echo switch puzzle (Q), collect the green prism to unlock the Imaginary-Time Realm. A second checkpoint is past the Echo gate. Press **R** near the green gate to open it and materialize the spectral bridge for 7 seconds; jump past the green instability hazard to the final goal. It costs 30 Coherence and has a 10-second cooldown. Inventory: **I**; science: **F**; reset: **Shift+N** while inventory is open.

### Scientific integrity

The name invokes imaginary-time propagation, mathematically `exp[-τ(H − E_ref)]`, which can suppress excited-state contributions relative to the ground state when there is initial ground-state overlap and appropriate spectral conditions. **The timed realm and bridge are a fictional puzzle mechanic; the game does not execute imaginary-time evolution, a quantum Monte Carlo calculation, or time travel.**

## Headless checks

```bash
godot --headless --path game --editor --import
for test in test_tunneling test_demo_scene test_progression test_echoes test_realm; do
  godot --headless --path game --script "res://tests/${test}.gd" || exit 1
done
```

Tests set `persistence_enabled = false` for instantiated levels and use a separate test-specific JSON save path; they do not overwrite the player's game save.
