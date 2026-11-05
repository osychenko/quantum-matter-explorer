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
    root.add_child(level)
    await physics_frame
    await physics_frame
    var player = level.get_node("Player")
    var barrier = level.get_node("Barrier")
    var floor = level.get_node("Ground")
    check(floor is StaticBody2D, "Demo has physical ground")
    check(barrier is StaticBody2D, "Barrier is a real collision body")
    check(player is CharacterBody2D, "Player uses a physics character body")
    check(level.get_node_or_null("HUD") != null, "A HUD is mounted")
    player.global_position = Vector2(830, 606)
    player.velocity = Vector2.ZERO
    player.discovered = true
    await physics_frame
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
    if result["success"]:
        check(player.global_position.x > barrier.global_position.x,
                "Successful tunneling moves across the barrier")
        check(barrier.exit_is_clear(player, player.global_position),
                "Successful tunneling places player in an unoccupied location")
    else:
        check(player.global_position.x < barrier.global_position.x,
                "Reflected attempt leaves player on entrance side")
    level.queue_free()
    await process_frame
    if failures.is_empty():
        print("PASS test_demo_scene.gd — physics scene, occupancy and ability")
    else:
        printerr("FAIL test_demo_scene.gd: %d failure(s)" % failures.size())
    quit(0 if failures.is_empty() else 1)
