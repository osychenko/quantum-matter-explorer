extends CharacterBody2D
## Deterministic platform movement plus a separately sampled tunneling action.

signal tunneling_result(outcome: Dictionary)
signal echo_requested(recording: Array)
signal echo_result(outcome: Dictionary)
signal realm_result(outcome: Dictionary)

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
const ECHO_COST := 22.0
const ECHO_COOLDOWN := 5.0
const ECHO_RECORD_FRAMES := 120  # ~2 seconds at Godot's default 60 Hz physics tick.
const ECHO_MIN_FRAMES := 12
const REALM_COST := 30.0
const REALM_DURATION := 7.0
const REALM_COOLDOWN := 10.0

var coherence := MAX_COHERENCE
var barrier: Variant = null
var additional_barriers: Array[StaticBody2D] = []
var coyote_remaining := 0.0
var jump_buffer_remaining := 0.0
var tunnel_cooldown_remaining := 0.0
var tunneling_unlocked := false
var echoes_unlocked := false
var imaginary_time_unlocked := false
var realm_remaining := 0.0
var realm_cooldown_remaining := 0.0
var realm_uses := 0
var echo_cooldown_remaining := 0.0
var echo_history: Array[Vector2] = []
var echo_uses := 0
var rng := RandomNumberGenerator.new()
var rng_initial_seed: int = 0
var attempts := 0
var glow_time := 0.0
var _stride_phase := 0.0
var _land_squash := 0.0
var _grounded_last_frame := false
var facing := 1.0
var _e_was_pressed := false
var _space_was_pressed := false
var _q_was_pressed := false
var _r_was_pressed := false


func _ready() -> void:
    rng.randomize()
    rng_initial_seed = rng.seed


func _physics_process(delta: float) -> void:
    coherence = minf(MAX_COHERENCE, coherence + REGEN_PER_SECOND * delta)
    tunnel_cooldown_remaining = maxf(0.0, tunnel_cooldown_remaining - delta)
    echo_cooldown_remaining = maxf(0.0, echo_cooldown_remaining - delta)
    realm_cooldown_remaining = maxf(0.0, realm_cooldown_remaining - delta)
    realm_remaining = maxf(0.0, realm_remaining - delta)
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
    # Animation is purely visual; the CharacterBody2D and collider never scale.
    var grounded_now := is_on_floor()
    if grounded_now and not _grounded_last_frame:
        _land_squash = 1.0
    _grounded_last_frame = grounded_now
    _land_squash = maxf(0.0, _land_squash - 4.0 * delta)
    _stride_phase += absf(velocity.x) / WALK_SPEED * 13.0 * delta
    echo_history.append(global_position)
    if echo_history.size() > ECHO_RECORD_FRAMES:
        echo_history.pop_front()
    var q_pressed := Input.is_physical_key_pressed(KEY_Q)
    if q_pressed and not _q_was_pressed:
        attempt_echo()
    _q_was_pressed = q_pressed
    var r_pressed := Input.is_physical_key_pressed(KEY_R)
    if r_pressed and not _r_was_pressed:
        attempt_imaginary_time()
    _r_was_pressed = r_pressed
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


func attempt_echo() -> Dictionary:
    var outcome: Dictionary
    if not echoes_unlocked:
        outcome = {"spawned": false, "cost": 0.0,
                "reason": "Echoes locked: collect the violet shard beyond barrier two."}
    elif echo_cooldown_remaining > 0.0:
        outcome = {"spawned": false, "cost": 0.0,
                "reason": "Echoes are recharging."}
    elif coherence < ECHO_COST:
        outcome = {"spawned": false, "cost": 0.0,
                "reason": "Not enough Coherence for an echo (22 required)."}
    elif echo_history.size() < ECHO_MIN_FRAMES:
        outcome = {"spawned": false, "cost": 0.0,
                "reason": "Move for a moment to record a trajectory."}
    else:
        # Keep the typed array intact: GDScript rejects assigning untyped Array literals.
        var trace: Array[Vector2] = []
        for point in echo_history:
            trace.append(point)
        coherence -= ECHO_COST
        echo_cooldown_remaining = ECHO_COOLDOWN
        echo_uses += 1
        outcome = {"spawned": true, "cost": ECHO_COST, "samples": trace.size()}
        echo_requested.emit(trace)
    echo_result.emit(outcome)
    return outcome


func is_realm_active() -> bool:
    return realm_remaining > 0.0


func attempt_imaginary_time() -> Dictionary:
    # Gameplay metaphor for imaginary-time projection; not time travel or simulation.
    var outcome: Dictionary
    if not imaginary_time_unlocked:
        outcome = {"activated": false, "cost": 0.0,
                "reason": "Imaginary-Time Realm locked: collect the green prism."}
    elif is_realm_active():
        outcome = {"activated": false, "cost": 0.0,
                "reason": "The realm is already active."}
    elif realm_cooldown_remaining > 0.0:
        outcome = {"activated": false, "cost": 0.0,
                "reason": "Imaginary-Time Realm is recharging."}
    elif coherence < REALM_COST:
        outcome = {"activated": false, "cost": 0.0,
                "reason": "Not enough Coherence (30 required)."}
    else:
        coherence -= REALM_COST
        realm_remaining = REALM_DURATION
        realm_cooldown_remaining = REALM_COOLDOWN
        realm_uses += 1
        outcome = {"activated": true, "cost": REALM_COST,
                "duration": REALM_DURATION}
    realm_result.emit(outcome)
    return outcome


func _draw() -> void:
    var running := is_on_floor() and absf(velocity.x) > 45.0
    var stride := sin(_stride_phase) if running else 0.0
    var bob := absf(stride) * 2.0 if running else 0.0
    var stretch := 0.08 if not is_on_floor() and velocity.y < 0.0 else 0.0
    var squat := _land_squash * 0.13

    # Shadow is outside the animated body transform.
    draw_circle(Vector2(0, 23), 17.0, Color(0.02, 0.05, 0.10, 0.30))
    if glow_time > 0.0:
        for i in range(4):
            var alpha := glow_time * (0.13 + float(i) * 0.05)
            draw_arc(Vector2(-facing * float(i) * 11.0, -6), 24.0 + float(i) * 9.0,
                    0, TAU, 48, Color(0.40, 0.90, 1.0, alpha), 2.0)
        draw_circle(Vector2(0, -6), 31.0 + 8.0 * glow_time,
                Color(0.25, 0.89, 1.0, glow_time * 0.4))
    if is_realm_active():
        draw_arc(Vector2(0, -8), 35.0, 0.0, TAU, 48,
                Color(0.42, 1.0, 0.77, 0.8), 3.0)
    if echo_cooldown_remaining > 0.0 and echoes_unlocked:
        draw_arc(Vector2(0, -8), 30.0, 0.0, TAU, 42,
                Color(0.75, 0.53, 1.0, 0.25), 2.0)

    # Apply squash, stretch, and running bob to artwork only.
    draw_set_transform(Vector2(0.0, -bob), 0.0,
            Vector2(1.0 + squat * 0.55 - stretch * 0.4,
                    1.0 - squat + stretch))
    var left_step := stride * 8.0
    var right_step := -stride * 8.0
    draw_line(Vector2(-8, 9), Vector2(-12 + left_step, 23), Color("#a2d4e0"), 5.0)
    draw_line(Vector2(8, 9), Vector2(12 + right_step, 23), Color("#a2d4e0"), 5.0)
    draw_line(Vector2(-10 + left_step, 23), Vector2(-2 + left_step, 23),
            Color("#e3f8f1"), 4.0)
    draw_line(Vector2(10 + right_step, 23), Vector2(18 + right_step, 23),
            Color("#e3f8f1"), 4.0)
    draw_colored_polygon(PackedVector2Array([Vector2(-13, -5), Vector2(13, -5),
            Vector2(11, 15), Vector2(-11, 15)]), Color("#398caa"))
    draw_line(Vector2(-13, -4), Vector2(-13 - 3.5 * stride, 11),
            Color("#b2e4de"), 3.5)
    draw_line(Vector2(13, -4), Vector2(13 + 3.5 * stride, 11),
            Color("#b2e4de"), 3.5)
    draw_circle(Vector2(0, -12), 14.0, Color("#e5fff6"))
    draw_arc(Vector2(0, -12), 14.0, PI, TAU, 32, Color("#83cfdc"), 2.0)
    draw_circle(Vector2(5.0 * facing, -13), 2.6, Color("#183c55"))
    draw_line(Vector2(-11, -20), Vector2(-7, -28), Color("#bff2eb"), 3.0)
    draw_line(Vector2(11, -20), Vector2(7, -28), Color("#bff2eb"), 3.0)
    draw_set_transform(Vector2.ZERO)
