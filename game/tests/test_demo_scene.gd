extends SceneTree
## Integration: instantiate an actual physics scene, validate exit checks and currency.

var failures: Array[String] = []


func _initialize() -> void:
    call_deferred("run_tests")


func check(value: bool, description: String) -> void:
    if not value:
        failures.append(description)
        push_error(description)


func run_tests() -> void:
    var packed: PackedScene = load("res://scenes/DemoLevel.tscn")
    check(packed != null, "Playable scene exists")
    if packed == null:
        quit(1)
        return
    var level := packed.instantiate()
    level.persistence_enabled = false  # Never read or modify a real player save in tests.
    root.add_child(level)
    await physics_frame
    await physics_frame
    var player = level.get_node("Player")
    var barrier = level.get_node("Barrier")
    var floor = level.get_node("Ground")
    check(floor is StaticBody2D, "Demo has physical ground")
    check(barrier is StaticBody2D, "Barrier is a real collision body")
    check(player is CharacterBody2D, "Player uses a physics character body")
    var barrier2 = level.get_node("Barrier2")
    check(barrier2 is StaticBody2D, "Second barrier is a real collision body")
    check(barrier2.transmission()["probability"] < barrier.transmission()["probability"],
            "Second barrier is harder for the same incoming particle")
    check(level.get_node("Checkpoint") is Area2D, "Checkpoint trigger exists")
    check(level.get_node("Hazard") is Area2D, "Hazard trigger exists")
    check(level.get_node_or_null("HUD") != null, "A HUD is mounted")
    check(level.get_node_or_null("LeftBoundary") is StaticBody2D,
            "Left wall prevents leaving the viewport")
    check(level.get_node_or_null("RightBoundary") is StaticBody2D,
            "Right wall prevents leaving the viewport")
    check(level.get_node("AbilityRelic") is Area2D, "Unlock pickup uses a real physics trigger")
    check(level.inventory_panel != null and not level.inventory_panel.visible,
            "Ability inventory is created and initially hidden")
    check(not player.tunneling_unlocked, "Fresh game starts with tunneling locked")
    player.global_position = Vector2(830, 606)
    player.velocity = Vector2.ZERO
    await physics_frame
    check(not player.tunneling_unlocked, "Approaching a barrier does not auto-unlock tunneling")
    var locked_coherence: float = player.coherence
    var locked_result: Dictionary = player.attempt_tunneling()
    check(not locked_result["attempted"] and locked_result["cost"] == 0.0,
            "Tunneling cannot be attempted before collecting the core")
    check(absf(player.coherence - locked_coherence) < 0.0001,
            "Locked tunneling does not spend Coherence")
    level._on_relic_collected(player)
    check(player.tunneling_unlocked, "Collecting the core unlocks tunneling")
    check(level.progression.is_unlocked("tunneling"), "Unlock is recorded in progression")
    check(barrier.is_near_player(player), "Player can discover the barrier from outside")
    var destination: Vector2 = barrier.exit_position(player)
    check(destination.x > barrier.global_position.x + barrier.WIDTH_PIXELS / 2.0,
            "Exit is beyond the wall, not inside it")
    check(barrier.exit_is_clear(player, destination), "Open exit is initially clear")
    # Add a real obstacle at the target position to test safe landing.
    var blocker := StaticBody2D.new()
    blocker.collision_layer = 1
    blocker.position = destination
    var collider := CollisionShape2D.new()
    var shape := RectangleShape2D.new()
    shape.size = Vector2(70, 70)
    collider.shape = shape
    blocker.add_child(collider)
    level.add_child(blocker)
    await physics_frame
    check(not barrier.exit_is_clear(player, destination), "Occupied exit is refused")
    var before: float = player.coherence
    var prevented: Dictionary = barrier.try_tunnel(player, before, player.rng)
    check(not prevented["attempted"] and prevented["cost"] == 0.0,
            "Blocked exit never spends Coherence")
    blocker.queue_free()
    await physics_frame
    check(barrier.exit_is_clear(player, destination), "Removing obstacle restores destination")
    player.rng.seed = 42
    var result: Dictionary = player.attempt_tunneling()
    check(result["attempted"], "Within-range, sufficient-energy attempt is eligible")
    check(absf(player.coherence - (before - 18.0)) < 0.1,
            "Eligible attempt spends exactly 18 Coherence")
    var cost_after: float = player.coherence
    var cooling: Dictionary = player.attempt_tunneling()
    check(not cooling["attempted"] and cooling["cost"] == 0.0,
            "Cooldown rejects repeated attempt without charge")
    check(absf(player.coherence - cost_after) < 0.0001,
            "Cooldown never charges Coherence")
    if result["success"]:
        check(player.global_position.x > barrier.global_position.x,
                "Successful tunneling moves across the barrier")
        check(barrier.exit_is_clear(player, player.global_position),
                "Successful tunneling places player in an unoccupied location")
    else:
        check(player.global_position.x < barrier.global_position.x,
                "Reflected attempt leaves player on entrance side")
    # Checkpoint/respawn handlers: direct invocation tests the deterministic rule,
    # while the scene above verifies both real Area2D triggers are present.
    level._on_checkpoint(player)
    check(level.checkpoint_activated, "Checkpoint can be activated")
    check(level.respawn_position == Vector2(1160, 595), "Checkpoint records safe respawn")
    var before_hazard: float = player.coherence
    level._on_hazard(player)
    check(player.global_position == level.respawn_position, "Hazard returns player to checkpoint")
    check(absf(player.coherence - maxf(0.0, before_hazard - 15.0)) < 0.0001,
            "Hazard applies its documented resource penalty")
    player.global_position = Vector2(1365, 595)
    player.velocity = Vector2.ZERO
    await physics_frame
    check(barrier2.is_near_player(player), "Player can approach the second barrier")
    check(player.get_nearby_barrier() == barrier2, "Nearest barrier selects second barrier")
    level._reset_progression()
    check(not player.tunneling_unlocked and not level.checkpoint_activated,
            "Reset locks ability and clears checkpoint")
    check(player.global_position == Vector2(210, 595), "Reset returns to start")
    check(not level.progression.is_unlocked("tunneling"), "Reset clears progression state")
    level.queue_free()
    await process_frame
    if failures.is_empty():
        print("PASS test_demo_scene.gd — physics scene, occupancy and ability")
    else:
        printerr("FAIL test_demo_scene.gd: %d failure(s)" % failures.size())
    quit(0 if failures.is_empty() else 1)
