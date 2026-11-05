extends CharacterBody2D
## Deterministic platform movement plus a separately sampled tunneling action.

signal tunneling_result(outcome: Dictionary)

const WALK_SPEED := 285.0
const ACCELERATION := 2350.0
const GROUND_FRICTION := 2600.0
const AIR_ACCELERATION := 1350.0
const AIR_FRICTION := 650.0
const COYOTE_TIME := 0.11
const JUMP_BUFFER_TIME := 0.12
const JUMP_CUT_MULTIPLIER := 0.45
const TUNNEL_COOLDOWN := 0.32
const JUMP_SPEED := -530.0
const GRAVITY := 1400.0
const MAX_COHERENCE := 100.0
const REGEN_PER_SECOND := 8.0

var coherence := MAX_COHERENCE
var barrier: Variant = null
var additional_barriers: Array[StaticBody2D] = []
var coyote_remaining := 0.0
var jump_buffer_remaining := 0.0
var tunnel_cooldown_remaining := 0.0
var tunneling_unlocked := false
var rng := RandomNumberGenerator.new()
var rng_initial_seed: int = 0
var attempts := 0
var glow_time := 0.0
var facing := 1.0
var _e_was_pressed := false
var _space_was_pressed := false


func _ready() -> void:
    rng.randomize()
    rng_initial_seed = rng.seed


func _physics_process(delta: float) -> void:
    coherence = minf(MAX_COHERENCE, coherence + REGEN_PER_SECOND * delta)
    tunnel_cooldown_remaining = maxf(0.0, tunnel_cooldown_remaining - delta)
    var right_down := Input.is_physical_key_pressed(KEY_D) or Input.is_key_pressed(KEY_RIGHT)
    var left_down := Input.is_physical_key_pressed(KEY_A) or Input.is_key_pressed(KEY_LEFT)
    var axis := int(right_down) - int(left_down)
    var on_ground := is_on_floor()
    if on_ground:
        coyote_remaining = COYOTE_TIME
    else:
        coyote_remaining = maxf(0.0, coyote_remaining - delta)
    if Input.is_action_just_pressed("ui_accept") or (Input.is_physical_key_pressed(KEY_SPACE) and not _space_was_pressed):
        jump_buffer_remaining = JUMP_BUFFER_TIME
    else:
        jump_buffer_remaining = maxf(0.0, jump_buffer_remaining - delta)
    var space_pressed := Input.is_physical_key_pressed(KEY_SPACE)
    _space_was_pressed = space_pressed
    var acceleration := ACCELERATION if on_ground else AIR_ACCELERATION
    var friction := GROUND_FRICTION if on_ground else AIR_FRICTION
    if axis == 0:
        velocity.x = move_toward(velocity.x, 0.0, friction * delta)
    else:
        velocity.x = move_toward(velocity.x, float(axis) * WALK_SPEED, acceleration * delta)
        facing = float(axis)
    if not on_ground:
        velocity.y += GRAVITY * delta
    if jump_buffer_remaining > 0.0 and coyote_remaining > 0.0:
        velocity.y = JUMP_SPEED
        jump_buffer_remaining = 0.0
        coyote_remaining = 0.0
    if Input.is_action_just_released("ui_accept") and velocity.y < 0.0:
        velocity.y *= JUMP_CUT_MULTIPLIER
    move_and_slide()
    var e_pressed := Input.is_physical_key_pressed(KEY_E)
    if e_pressed and not _e_was_pressed:
        attempt_tunneling()
    _e_was_pressed = e_pressed
    if glow_time > 0.0:
        glow_time = maxf(0.0, glow_time - delta)
    queue_redraw()


func _nearby_barrier() -> Variant:
    var nearest: Variant = null
    var nearest_distance := INF
    for candidate in [barrier] + additional_barriers:
        if candidate != null and candidate.has_method("is_near_player") and candidate.is_near_player(self):
            var distance := absf(global_position.x - candidate.global_position.x)
            if distance < nearest_distance:
                nearest_distance = distance
                nearest = candidate
    return nearest


func get_nearby_barrier() -> Variant:
    return _nearby_barrier()


func attempt_tunneling() -> Dictionary:
    var active: Variant = _nearby_barrier()
    if not tunneling_unlocked:
        var locked := {"attempted": false, "success": false, "cost": 0.0,
                "reason": "Tunneling locked: collect the glowing quantum core first."}
        tunneling_result.emit(locked)
        return locked
    if active == null:
        var too_far := {"attempted": false, "success": false, "cost": 0.0,
                "reason": "Approach a quantum barrier to attempt tunneling."}
        tunneling_result.emit(too_far)
        return too_far
    if tunnel_cooldown_remaining > 0.0:
        var cooling := {"attempted": false, "success": false, "cost": 0.0,
                "reason": "Tunneling is recharging."}
        tunneling_result.emit(cooling)
        return cooling
    var outcome: Dictionary = active.try_tunnel(self, coherence, rng)
    if outcome.get("attempted", false):
        coherence -= float(outcome["cost"])
        attempts += 1
        glow_time = 0.42
        tunnel_cooldown_remaining = TUNNEL_COOLDOWN
        if outcome["success"]:
            global_position = outcome["destination"]
            velocity = Vector2.ZERO
    tunneling_result.emit(outcome)
    return outcome


func _draw() -> void:
    if glow_time > 0.0:
        for i in range(4):
            var alpha := glow_time * (0.13 + float(i) * 0.05)
            draw_arc(Vector2(-facing * float(i) * 11.0, -6), 24.0 + float(i) * 9.0,
                    0, TAU, 48, Color(0.40, 0.90, 1.0, alpha), 2.0)
        draw_circle(Vector2(0, -6), 31.0 + 8.0 * glow_time,
                Color(0.25, 0.89, 1.0, glow_time * 0.4))
    # Original simple geometric avatar; no external art assets or IP.
    draw_line(Vector2(-9, 12), Vector2(-15, 22), Color("#94c9d4"), 5.0)
    draw_line(Vector2(9, 12), Vector2(15, 22), Color("#94c9d4"), 5.0)
    draw_polygon(PackedVector2Array([Vector2(-13, -5), Vector2(13, -5),
            Vector2(11, 15), Vector2(-11, 15)]),
            PackedColorArray([Color("#4b9cae"), Color("#4b9cae"),
            Color("#255775"), Color("#255775")]))
    draw_circle(Vector2(0, -12), 14.0, Color("#e1f4ea"))
    draw_circle(Vector2(5.0 * facing, -13), 2.5, Color("#19344e"))
    draw_line(Vector2(-11, -20), Vector2(-7, -28), Color("#bff2eb"), 3.0)
    draw_line(Vector2(11, -20), Vector2(7, -28), Color("#bff2eb"), 3.0)
