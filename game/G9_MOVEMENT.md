# G9 — Responsive platformer movement and stomp combat

This phase replaces the G8 basic integration with tunable character motion. It changes **no** research/QMC code or saved progression.

## Controls and timing

- **A/D / arrow keys**: gradually accelerate to `max_run_speed`, or reverse with finite acceleration. Releasing gradually decelerates.
- **Space / ui_accept**: jump. Hold longer for the full jump, release for a short hop. The key state for press and release now comes from the same input pathway.
- **Apex**: a short, one-off reduction of gravity on long jumps. Falling has higher gravity than rising; no artificial freeze.
- **Coyote time**: jump within 0.12 s after stepping off an edge.
- **Jump buffer**: a jump input just before landing executes when the player reaches the floor.
- **Air inertia**: takeoff keeps horizontal velocity; weaker air control and air braking apply while airborne.
- **Stomp enemy**: land on a lime-striped sentinel **while moving downward** to defeat it and bounce. Side contact returns you to the room checkpoint and uses that room's existing hazard penalty.

## Tuning in Godot

Open either `DemoLevel.tscn` or `CalibrationChamber.tscn`, select **Player**, and tune the **G9 Movement** and **G9 Jump** Inspector groups. These export properties are independently editable per room. The defaults use 285 px/s horizontal speed, 1700 px/s² run acceleration, 2050 px/s² ground braking, 720 px/s² air acceleration, 155 px/s² air friction, a 560 px/s jump impulse, 1450 px/s² base gravity, 1.48× falling gravity, 0.085 s apex hang, and 0.12 s jump leniency.

## Collisions

`LevelPlatform.gd` has `corner_cut_px` (0–12). Ledges use a **convex eight-vertex chamfer**, not an ellipse: the middle of the top stays flat, so the character does not slide toward the edge. Walls, gates and ground retain rectangular collision (`corner_cut_px = 0`). The renderer uses the same polygon when chamfered. Resize via the existing Dimensions Inspector field.

## Tests

```bash
 godot --headless --path game --editor --import
 godot --headless --path game --script res://tests/test_movement.gd
 godot --headless --path game --script res://tests/test_realm.gd
```

Run the full G1–G8 suite before merging. The stomp enemies are intended as first-pass training targets, not a fully featured combat system. Combat immunity, AI, animation and reward balancing can follow as a dedicated phase.
