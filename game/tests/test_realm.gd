extends SceneTree
## G5: editor-placeable platforms, realm unlock/cost, gate/bridge and checkpoint.

var failures: Array[String] = []


func _initialize() -> void:
    call_deferred("run_tests")


func check(value: bool, description: String) -> void:
    if not value:
        failures.append(description)
        push_error(description)


func run_tests() -> void:
    var packed: PackedScene = load("res://scenes/DemoLevel.tscn")
    check(packed != null, "G5 scene loads")
    if packed == null:
        quit(1)
        return
    var level = packed.instantiate()
    level.persistence_enabled = false
    root.add_child(level)
    await physics_frame
    await physics_frame
    var player = level.get_node("Player")
    var right_ledge = level.get_node("RightLedge")
    var ground = level.get_node("Ground")
    var ledge_collider = right_ledge.get_node("CollisionShape2D") as CollisionShape2D
    check(ground is StaticBody2D, "Ground is a scene-owned physics node")
    check(right_ledge is StaticBody2D, "Right ledge is an editor-movable physics node")
    check(right_ledge.position == Vector2(1400, 585), "Right ledge lowered by 50 px")
    check(ledge_collider.shape is ConvexPolygonShape2D,
            "G9 ledge collision has a flat top and beveled corners")
    if ledge_collider.shape is ConvexPolygonShape2D:
        var points: PackedVector2Array = (ledge_collider.shape as ConvexPolygonShape2D).points
        check(points.size() == 8 and points[0].y == -right_ledge.dimensions.y * 0.5,
                "Beveled ledge keeps a horizontal top")
    right_ledge.dimensions = Vector2(200, 22)
    if ledge_collider.shape is ConvexPolygonShape2D:
        var resized: PackedVector2Array = (ledge_collider.shape as ConvexPolygonShape2D).points
        check(is_equal_approx(resized[1].x, 100.0 - right_ledge.corner_cut_px),
                "Resizing Inspector dimensions updates convex collision")
    var old_collider_pos: Vector2 = ledge_collider.global_position
    right_ledge.position.y += 25.0
    check(absf(ledge_collider.global_position.y - old_collider_pos.y - 25.0) < 0.01,
            "Moving ledge also moves its attached collision")

    var realm_gate = level.get_node("RealmGate")
    var bridge = level.get_node("SpectralBridge")
    var gate_collider = realm_gate.get_node("CollisionShape2D") as CollisionShape2D
    var bridge_collider = bridge.get_node("CollisionShape2D") as CollisionShape2D
    check(not gate_collider.disabled and bridge_collider.disabled,
            "Realm gate closed and spectral bridge intangible at start")
    check(not player.imaginary_time_unlocked, "Imaginary-Time Realm starts locked")
    var locked: Dictionary = player.attempt_imaginary_time()
    check(not locked["activated"] and locked["cost"] == 0.0,
            "Locked realm does not spend Coherence")
    level._on_imaginary_relic_collected(player)
    check(player.imaginary_time_unlocked, "Green prism unlocks the realm")
    check(level.progression.is_unlocked("imaginary_time"), "Unlock recorded in progression")
    player.global_position = Vector2(2150, 595)
    player.velocity = Vector2.ZERO
    player.coherence = 100.0
    var before: float = player.coherence
    var started: Dictionary = player.attempt_imaginary_time()
    check(started["activated"], "R activates the imaginary-time gameplay realm")
    check(absf(before - player.coherence - player.REALM_COST) < 0.0001,
            "Realm deducts exactly 30 Coherence")
    await physics_frame
    await physics_frame
    check(level.realm_environment_active and gate_collider.disabled and not bridge_collider.disabled,
            "Realm opens the gate and materializes the bridge")
    var used: float = player.coherence
    var repeated: Dictionary = player.attempt_imaginary_time()
    check(not repeated["activated"] and repeated["cost"] == 0.0,
            "Cannot activate an already-active realm twice")
    check(absf(used - player.coherence) < 0.0001,
            "Rejected second use is free")
    player.realm_remaining = 0.0
    player.global_position = Vector2(2150, 595)
    player.velocity = Vector2.ZERO
    await physics_frame
    await physics_frame
    check(not level.realm_environment_active and not gate_collider.disabled and bridge_collider.disabled,
            "Gate and bridge revert after realm duration")
    var cooling: Dictionary = player.attempt_imaginary_time()
    check(not cooling["activated"] and cooling["cost"] == 0.0,
            "Realm cooldown prevents immediate reuse without cost")
    level._on_realm_checkpoint(player)
    check(level.realm_checkpoint_activated and level.progression.realm_checkpoint_reached,
            "Second checkpoint is recorded")
    check(level.respawn_position == level.REALM_CHECKPOINT_POSITION,
            "Later checkpoint becomes active respawn location")
    player.global_position = Vector2(2485, 595)
    var before_hazard: float = player.coherence
    level._on_hazard(player)
    check(player.global_position == level.REALM_CHECKPOINT_POSITION,
            "Realm hazard returns the player to second checkpoint")
    check(absf(player.coherence - maxf(0.0, before_hazard - 15.0)) < 0.0001,
            "Realm hazard retains the resource penalty")
    level._reset_progression()
    await physics_frame
    await physics_frame
    check(not player.imaginary_time_unlocked and not level.progression.realm_checkpoint_reached,
            "Reset clears G5 ability and new checkpoint")
    check(not gate_collider.disabled and bridge_collider.disabled,
            "Reset closes realm gate and disables spectral bridge")
    level.queue_free()
    await process_frame
    if failures.is_empty():
        print("PASS test_realm.gd — editor geometry, unlock, gate, bridge, checkpoint, reset")
    else:
        printerr("FAIL test_realm.gd: %d failure(s)" % failures.size())
    quit(0 if failures.is_empty() else 1)
