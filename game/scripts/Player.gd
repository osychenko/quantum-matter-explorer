extends CharacterBody2D
## Deterministic platform movement plus a separately sampled tunneling action.

signal tunneling_result(outcome: Dictionary)

const WALK_SPEED := 285.0
const ACCELERATION := 2100.0
const JUMP_SPEED := -530.0
const GRAVITY := 1400.0
const MAX_COHERENCE := 100.0
const REGEN_PER_SECOND := 8.0

var coherence := MAX_COHERENCE
var barrier: Variant = null
var discovered := false
var rng := RandomNumberGenerator.new()
var rng_initial_seed: int = 0
var attempts := 0
var glow_time := 0.0
var facing := 1.0
var _e_was_pressed := false


func _ready() -> void:
    rng.randomize()
    rng_initial_seed = rng.seed


func _physics_process(delta: float) -> void:
    coherence = minf(MAX_COHERENCE, coherence + REGEN_PER_SECOND * delta)
    var right_down := Input.is_physical_key_pressed(KEY_D) or Input.is_key_pressed(KEY_RIGHT)
    var left_down := Input.is_physical_key_pressed(KEY_A) or Input.is_key_pressed(KEY_LEFT)
    var axis := int(right_down) - int(left_down)
    velocity.x = move_toward(velocity.x, float(axis) * WALK_SPEED, ACCELERATION * delta)
    if axis != 0:
        facing = float(axis)
    if not is_on_floor():
        velocity.y += GRAVITY * delta
    if is_on_floor() and Input.is_action_just_pressed("ui_accept"):
        velocity.y = JUMP_SPEED
    move_and_slide()
    if barrier != null and barrier.has_method("is_near_player") \
            and barrier.is_near_player(self):
        discovered = true
    var e_pressed := Input.is_physical_key_pressed(KEY_E)
    if e_pressed and not _e_was_pressed:
        attempt_tunneling()
    _e_was_pressed = e_pressed
    if glow_time > 0.0:
        glow_time = maxf(0.0, glow_time - delta)
    queue_redraw()


func attempt_tunneling() -> Dictionary:
    if not discovered or barrier == null:
        var not_found := {"attempted": false, "success": false, "cost": 0.0,
                "reason": "Discover a barrier before using tunneling."}
        tunneling_result.emit(not_found)
        return not_found
    var outcome: Dictionary = barrier.try_tunnel(self, coherence, rng)
    if outcome.get("attempted", false):
        coherence -= float(outcome["cost"])
        attempts += 1
        glow_time = 0.42
        if outcome["success"]:
            global_position = outcome["destination"]
            velocity = Vector2.ZERO
    tunneling_result.emit(outcome)
    return outcome


func _draw() -> void:
    if glow_time > 0.0:
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
