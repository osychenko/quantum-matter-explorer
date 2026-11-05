# G8 · Presentation, Game Feel, and Navigation

G8 adds a procedural title/room-selection screen, fade transitions, an animated geometric avatar, collision-independent ability VFX, and a shared cooldown strip. No new gameplay costs, probabilities, or save fields. Research QMC code remains untouched.

## Controls

- **Menu:** Play/Continue enters Barrier Ruins; Calibration Chamber unlocks when Barrier Ruins have been completed.
- **H:** return to the main menu from either room (not while a measurement modal is open).
- **P:** from completed Barrier Ruins into Calibration Chamber; **B:** return from Calibration to Barrier Ruins.
- The existing E, Q, R, M, F, I, movement, and jump controls remain unchanged.

## Visual changes

- Player body/limbs receive ground stride, jump stretch, and landing squash; physics collision remains the same.
- World-space burst rings clarify successful/failed tunneling and ability activation, checkpoints, calibration and exits.
- Platforms get engraved decoration, following their scene-node dimensions and positions automatically.
- The compact E/Q/R cooldown strip reflects actual ability cooldowns.
- Scene transitions fade via a temporary CanvasLayer attached to the tree root.

## Test

```sh
godot --headless --path game --editor --import
godot --headless --path game --script res://tests/test_presentation.gd
```

The G8 visuals are gameplay-inspired effects. They do not depict actual quantum Monte Carlo simulations, and all analytical probabilities remain computed by the pre-existing tunneling model.
