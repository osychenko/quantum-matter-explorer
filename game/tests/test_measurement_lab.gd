extends SceneTree
## G6 integration: modal HUD, two existing barriers, sample counts, no game mutation.

var failures: Array[String] = []


func _initialize() -> void:
    call_deferred("run_tests")


func check(condition: bool, description: String) -> void:
    if not condition:
        failures.append(description)
        push_error(description)


func run_tests() -> void:
    var packed: PackedScene = load("res://scenes/DemoLevel.tscn")
    check(packed != null, "Scene loads with measurement laboratory")
    if packed == null:
        quit(1)
        return
    var level = packed.instantiate()
    level.persistence_enabled = false
    root.add_child(level)
    await physics_frame
    await physics_frame
    var player = level.get_node("Player")
    var barrier = level.get_node("Barrier")
    var barrier2 = level.get_node("Barrier2")
    var lab = level.measurement_lab
    check(lab != null and not lab.visible, "Measurement lab starts hidden")
    check(not level._open_measurement_lab(), "Must approach a barrier before opening")
    player.global_position = Vector2(830, 595)
    player.velocity = Vector2.ZERO
    await physics_frame
    check(level._open_measurement_lab(), "Lab opens beside barrier one")
    check(lab.visible, "Lab becomes visible")
    check(not player.is_physics_processing(), "Lab freezes player controls")
    check(lab.barrier_name == "BARRIER 1", "Nearest barrier correctly selected")
    var before_energy: float = barrier.incident_energy
    var before_width: float = barrier.model_width
    var before_coherence: float = player.coherence
    lab.energy_spin.value = 0.95
    lab.width_spin.value = 1.3
    check(barrier.incident_energy == before_energy and barrier.model_width == before_width,
            "Lab parameters do not overwrite gameplay barriers")
    var one: Dictionary = lab.run_batch(10)
    check(one.get("valid", false) and one["trials"] == 10,
            "Interactive batch runs ten trials")
    lab.clear_results()
    check(lab.run_batch(10) == one, "Cleared series repeats exactly with the same seed")
    var two: Dictionary = lab.run_batch(100)
    check(two.get("valid", false), "Interactive second batch completes")
    check(lab.total_trials == 110 and lab.run_number == 2,
            "Cumulative counts combine identical settings")
    check(lab.total_transmitted >= 0 and lab.total_transmitted <= 110,
            "Accumulated transmitted count is bounded")
    check(player.coherence == before_coherence,
            "Measurements are free and do not change Coherence")
    lab.width_spin.value = 1.5
    check(lab.total_trials == 0 and lab.run_number == 0,
            "Changing model parameters clears incomparable experiment results")
    level._close_measurement_lab()
    check(not lab.visible and player.is_physics_processing(),
            "Closing lab restores player physics")
    player.global_position = Vector2(1365, 595)
    player.velocity = Vector2.ZERO
    await physics_frame
    check(level._open_measurement_lab(), "Lab opens beside barrier two")
    check(lab.barrier_name == "BARRIER 2", "Second barrier is selected")
    check(absf(lab.barrier_height - barrier2.barrier_height) < 0.0001,
            "Second barrier parameters are loaded")
    check(lab.run_batch(1000).get("valid", false), "One thousand trial batch completes")
    level._close_measurement_lab()
    level.queue_free()
    await process_frame
    if failures.is_empty():
        print("PASS test_measurement_lab.gd — HUD, selection, parameters, controls, free sampling")
    else:
        printerr("FAIL test_measurement_lab.gd: %d failures" % failures.size())
    quit(0 if failures.is_empty() else 1)
