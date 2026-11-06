extends SceneTree
## G9: deterministic movement integration, chamfer shape and stomp behavior.
const PLAYER = preload("res://scripts/Player.gd")
const PLATFORM = preload("res://scripts/LevelPlatform.gd")
const STOMP = preload("res://scripts/StompEnemy.gd")
var failures: Array[String] = []
var side_hit_count := 0


func _initialize() -> void:
    call_deferred("run_tests")


func check(condition: bool, explanation: String) -> void:
    if not condition:
        failures.append(explanation)
        push_error(explanation)


func _on_side_hit(_body: Node2D) -> void:
    side_hit_count += 1


func run_tests() -> void:
    var p = PLAYER.new()
    p.set_physics_process(false)
    root.add_child(p)
    var dt := 1.0 / 60.0
    p.advance_movement(dt, 1, false, false, true)
    check(p.velocity.x > 0.0 and p.velocity.x < p.max_run_speed,
            "Ground acceleration is progressive")
    for i in range(30):
        p.advance_movement(dt, 1, false, false, true)
    check(is_equal_approx(p.velocity.x, p.max_run_speed), "Run speed is capped")
    p.advance_movement(dt, 0, false, false, true)
    check(p.velocity.x > 0.0 and p.velocity.x < p.max_run_speed,
            "Releasing movement produces finite braking inertia")

    p.velocity = Vector2(260.0, -200.0)
    p.advance_movement(dt, 0, false, true, false)
    check(p.velocity.x > 0.0 and p.velocity.x < 260.0,
            "Takeoff momentum survives in the air")
    var long_jump = PLAYER.new()
    long_jump.set_physics_process(false)
    root.add_child(long_jump)
    var short_jump = PLAYER.new()
    short_jump.set_physics_process(false)
    root.add_child(short_jump)
    long_jump.advance_movement(dt, 0, true, true, true)
    short_jump.advance_movement(dt, 0, true, true, true)
    for i in range(12):
        long_jump.advance_movement(dt, 0, false, true, false)
        short_jump.advance_movement(dt, 0, false, false, false)
    check(short_jump.velocity.y > long_jump.velocity.y,
            "Releasing Space early yields a shorter jump")
    var cut_after_once: float = short_jump.velocity.y
    short_jump.advance_movement(dt, 0, false, false, false)
    check(short_jump.velocity.y > cut_after_once,
            "Jump cutting is not repeatedly multiplied each frame")

    var apex = PLAYER.new()
    apex.set_physics_process(false)
    root.add_child(apex)
    apex.velocity.y = -30.0
    apex.advance_movement(dt, 0, false, true, false)
    check(apex.apex_hang_remaining > 0.0 and
            apex.velocity.y < -30.0 + apex.base_gravity * dt,
            "Brief apex hang reduces gravity without freezing motion")
    var falling = PLAYER.new()
    falling.set_physics_process(false)
    root.add_child(falling)
    falling.velocity.y = 120.0
    falling.advance_movement(dt, 0, false, false, false)
    check(falling.velocity.y > 120.0 + falling.base_gravity * dt,
            "Fall gravity exceeds rising gravity")

    var coyote = PLAYER.new()
    coyote.set_physics_process(false)
    root.add_child(coyote)
    coyote.advance_movement(dt, 0, false, false, true)
    coyote.advance_movement(0.04, 0, true, true, false)
    check(coyote.velocity.y < 0.0, "Coyote jump remains possible after leaving edge")
    coyote.coyote_remaining = 0.0
    coyote.velocity.y = 70.0
    coyote.advance_movement(dt, 0, true, true, false)
    check(coyote.velocity.y > 0.0 and coyote.jump_buffer_remaining > 0.0,
            "Early jump press is buffered while falling")
    coyote.advance_movement(dt, 0, false, true, true)
    check(coyote.velocity.y < 0.0 and coyote.jump_buffer_remaining == 0.0,
            "Buffered jump fires on landing")

    var platform = PLATFORM.new()
    var platform_collider := CollisionShape2D.new()
    platform_collider.name = "CollisionShape2D"
    platform_collider.shape = RectangleShape2D.new()
    platform.add_child(platform_collider)
    root.add_child(platform)
    platform.corner_cut_px = 6.0
    platform.dimensions = Vector2(200, 22)
    check(platform_collider.shape is ConvexPolygonShape2D,
            "Chamfer is a convex collider, not a slippery ellipse")
    if platform_collider.shape is ConvexPolygonShape2D:
        var points: PackedVector2Array = (platform_collider.shape as ConvexPolygonShape2D).points
        check(points.size() == 8 and points[0].y == points[1].y,
                "Chamfer leaves a flat walkable upper surface")
    platform.corner_cut_px = 0.0
    check(platform_collider.shape is RectangleShape2D,
            "A zero corner cut preserves rectangular ground and gate collision")

    var enemy = STOMP.new()
    enemy.position = Vector2(400, 623)
    var enemy_collider := CollisionShape2D.new()
    enemy_collider.name = "CollisionShape2D"
    enemy_collider.shape = RectangleShape2D.new()
    enemy.add_child(enemy_collider)
    root.add_child(enemy)
    var player_collider := CollisionShape2D.new()
    player_collider.name = "CollisionShape2D"
    var shape := RectangleShape2D.new()
    shape.size = Vector2(26, 44)
    player_collider.shape = shape
    p.add_child(player_collider)
    p.position = Vector2(400, 581)
    p.velocity = Vector2(0, 170)
    enemy._on_body_entered(p)
    check(enemy.defeated and p.velocity.y < 0.0,
            "Descending from above defeats enemy and bounces the player")
    var side_enemy = STOMP.new()
    side_enemy.position = Vector2(600, 623)
    var side_collider := CollisionShape2D.new()
    side_collider.name = "CollisionShape2D"
    side_collider.shape = RectangleShape2D.new()
    side_enemy.add_child(side_collider)
    root.add_child(side_enemy)
    side_enemy.player_hit.connect(_on_side_hit)
    p.position = Vector2(575, 618)
    p.velocity = Vector2(180, 0)
    side_enemy._on_body_entered(p)
    check(not side_enemy.defeated and side_hit_count == 1,
            "Side impact is a hazard rather than a stomp")

    if failures.is_empty():
        print("PASS test_movement.gd — acceleration, jumps, inertia, coyote, chamfers, stomp")
    else:
        printerr("FAIL test_movement.gd: %d failure(s)" % failures.size())
    quit(0 if failures.is_empty() else 1)
