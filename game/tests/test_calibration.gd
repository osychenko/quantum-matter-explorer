extends SceneTree
## G7: experiment-driven gate, parameter consistency, editor geometry and save.

var failures: Array[String] = []


func _initialize() -> void:
    call_deferred("run_tests")


func check(value: bool, explanation: String) -> void:
    if not value:
        failures.append(explanation)
        push_error(explanation)


func run_tests() -> void:
    var packed: PackedScene = load("res://scenes/CalibrationChamber.tscn")
    check(packed != null, "G7 calibration scene loads")
    if packed == null:
        quit(1)
        return
    var level = packed.instantiate()
    level.persistence_enabled = false
    root.add_child(level)
    await physics_frame
    await physics_frame
    var player = level.get_node("Player")
    var console = level.get_node("CalibrationConsole")
    var gate = level.get_node("CalibrationGate")
    var gate_collider = gate.get_node("CollisionShape2D") as CollisionShape2D
    check(not level.gate_open and not gate_collider.disabled,
            "Calibration gate starts physically closed")
    check(level.get_node("RightLedge") is StaticBody2D,
            "Optional ledge is movable in editor")
    check(not level._open_lab(), "Lab cannot be opened far from console")
    player.global_position = console.global_position + Vector2(-80, 25)
    player.velocity = Vector2.ZERO
    await physics_frame
    check(level._open_lab(), "Console opens the G6 measurement lab")
    check(level.lab.visible and not player.is_physics_processing(),
            "Lab pauses player movement")
    var laboratory = level.lab
    check(laboratory.guidance_label.text.contains("PUZZLE"),
            "Measurement lab shows the G7 challenge inside its UI")
    var before_coherence: float = player.coherence
    laboratory.energy_spin.value = 0.8
    laboratory.width_spin.value = 1.0
    var prediction: Dictionary = laboratory.run_batch(1000)
    check(prediction.get("valid", false) and not level.gate_open,
            "Large batch alone is insufficient without a baseline")
    check(not level.small_batch_seen, "Small sample still required")
    laboratory.width_spin.value = 0.8
    laboratory.run_batch(10)
    check(not level.small_batch_seen,
            "Outside target transmission window, a small batch does not advance puzzle")
    laboratory.width_spin.value = 1.0
    var small: Dictionary = laboratory.run_batch(10)
    check(small.get("valid", false) and level.small_batch_seen,
            "Small baseline records its sampling uncertainty")
    check(laboratory.guidance_label.text.contains("STEP 2"),
            "Instructions advance to large-batch phase")
    laboratory.width_spin.value = 1.1
    check(not level.small_batch_seen, "Changing parameters invalidates previous baseline")
    laboratory.width_spin.value = 1.0
    laboratory.run_batch(10)
    var large: Dictionary = laboratory.run_batch(1000)
    check(large.get("valid", false), "Long measurement batch finishes")
    check(level.gate_open and level.progression.calibration_gate_cleared,
            "Correct two-stage experiment permanently unlocks gate")
    check(level.small_standard_error > float(large["standard_error"]),
            "Sampling error shrinks after increasing N")
    await physics_frame
    await physics_frame
    check(gate_collider.disabled, "Unlocked gate has no collision")
    check(console.completed, "Terminal shows the solved state")
    check(laboratory.guidance_label.text.contains("GATE UNLOCKED"),
            "Lab displays the result of a successful calibration")
    check(player.coherence == before_coherence, "Experiments do not spend Coherence")
    laboratory.width_spin.value = 1.25
    check(level.gate_open, "Changing parameters cannot relock solved gate")
    level._close_lab()
    check(not level.lab.visible and player.is_physics_processing(),
            "Closing lab resumes movement")
    level._on_checkpoint(player)
    check(level.checkpoint_active, "Checkpoint can activate after the gate")
    check(level.respawn_position == level.get_node("Checkpoint").global_position + Vector2(0, 10),
            "Checkpoint controls respawn")
    player.global_position = Vector2(1110, 595)
    level._on_hazard(player)
    check(player.global_position == level.respawn_position,
            "Hazard returns player to new checkpoint")
    level._on_goal(player)
    check(level.room_completed and level.progression.calibration_chamber_cleared,
            "Exit records complete G7 mastery")
    level.queue_free()
    await process_frame
    var reloaded = packed.instantiate()
    reloaded.persistence_enabled = false
    reloaded.progression.calibration_gate_cleared = true
    reloaded.progression.calibration_chamber_cleared = true
    root.add_child(reloaded)
    await physics_frame
    await physics_frame
    var restored_collider = reloaded.get_node("CalibrationGate/CollisionShape2D") as CollisionShape2D
    check(reloaded.gate_open and restored_collider.disabled and reloaded.room_completed,
            "Solved gate and room are restored after reentering the chamber")
    reloaded.queue_free()
    await process_frame
    if failures.is_empty():
        print("PASS test_calibration.gd — two batches, consistency, gate, checkpoint, completion")
    else:
        printerr("FAIL test_calibration.gd: %d failures" % failures.size())
    quit(0 if failures.is_empty() else 1)
