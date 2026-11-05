extends SceneTree
## G4 integration: replay trace, gate switch, cooldown, reset and progress safety.

var failures: Array[String] = []


func _initialize() -> void:
    call_deferred("run_tests")


func check(value: bool, description: String) -> void:
    if not value:
        failures.append(description)
        push_error(description)


func run_tests() -> void:
    var scene: PackedScene = load("res://scenes/DemoLevel.tscn")
    check(scene != null, "G4 scene loads")
    if scene == null:
        quit(1)
        return
    var level = scene.instantiate()
    level.persistence_enabled = false
    root.add_child(level)
    await physics_frame
    await physics_frame
    var player = level.get_node("Player")
    var gate = level.get_node("EchoGate")
    var collider = gate.get_node("CollisionShape2D") as CollisionShape2D
    check(level.get_node("EchoRelic") is Area2D, "Echo shard pickup exists")
    check(level.get_node("EchoSwitch") is Area2D, "Echo switch exists")
    check(gate is StaticBody2D and not collider.disabled, "Gate starts solid")
    check(not player.echoes_unlocked, "Echo ability starts locked")
    var locked: Dictionary = player.attempt_echo()
    check(not locked["spawned"] and locked["cost"] == 0.0,
            "Uncollected echo ability cannot spend Coherence")
    level._on_echo_relic_collected(player)
    check(player.echoes_unlocked, "Violet relic unlocks QMC Echoes")
    check(level.progression.is_unlocked("echoes"), "Echo unlock is recorded")
    await process_frame  # The Area2D monitoring change is deferred by Godot.
    await physics_frame
    check(not level.echo_relic.monitoring, "Collected shard is no longer collectible")
    # Simulate a legitimate 1-second stationary recording on the violet switch.
    player.echo_history.clear()
    for _i in range(60):
        player.echo_history.append(Vector2(1610, 620))
    player.global_position = Vector2(1644, 618)
    player.velocity = Vector2.ZERO
    var before: float = player.coherence
    var launched: Dictionary = player.attempt_echo()
    check(launched["spawned"] and launched["samples"] >= 60,
            "A recorded trajectory launches a replay ghost")
    check(absf(before - player.coherence - player.ECHO_COST) < 0.0001,
            "Echo charges exactly the advertised Coherence")
    check(is_instance_valid(level.echo_ghost), "Echo ghost appears in scene")
    var remaining: float = player.coherence
    var cooling: Dictionary = player.attempt_echo()
    check(not cooling["spawned"] and cooling["cost"] == 0.0,
            "Cooldown prevents instant replay spam")
    check(absf(remaining - player.coherence) < 0.0001,
            "Rejected echo does not spend Coherence")
    await physics_frame
    await physics_frame
    await physics_frame
    check(level.echo_gate_open and level.gate_open_remaining > 0.0,
            "Historical replay on switch opens the gate")
    check(collider.disabled, "Open gate disables collision")
    player.global_position = Vector2(1100, 618)
    player.velocity = Vector2.ZERO
    if is_instance_valid(level.echo_ghost):
        level.echo_ghost.queue_free()
    level.echo_ghost = null
    level.gate_open_remaining = 0.0
    await physics_frame
    await physics_frame
    check(not level.echo_gate_open and not collider.disabled,
            "Gate closes after replay and timer end")
    level._reset_progression()
    await physics_frame
    check(not player.echoes_unlocked and not level.progression.is_unlocked("echoes"),
            "Reset removes echo unlock")
    check(not level.echo_gate_open and level.echo_ghost == null,
            "Reset clears the echo and closes the gate")
    check(player.global_position.distance_to(level.START_POSITION) < 50.0,
            "Reset returns player to the start")
    level.queue_free()
    await process_frame
    if failures.is_empty():
        print("PASS test_echoes.gd — unlock, playback, gate, cooldown, reset")
    else:
        printerr("FAIL test_echoes.gd: %d failure(s)" % failures.size())
    quit(0 if failures.is_empty() else 1)
