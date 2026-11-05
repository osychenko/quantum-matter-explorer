extends StaticBody2D
## One special collision obstacle. Only a nearby player may attempt transmission.
## An accepted attempt must also pass an independent landing-space collision check.

const MODEL = preload("res://scripts/TunnelingModel.gd")
const WIDTH_PIXELS := 72.0
const REACH_FROM_FACE := 84.0

@export var incident_energy := 0.8
@export var barrier_height := 1.2
@export var model_width := 0.8
@export var kinetic_coefficient := 0.5

var visual_clock := 0.0
var pulse_time := 0.0
var pulse_success := false

func show_attempt(success: bool) -> void:
    pulse_success = success
    pulse_time = 0.62



func _process(delta: float) -> void:
    visual_clock += delta
    pulse_time = maxf(0.0, pulse_time - delta)
    queue_redraw()


func _draw() -> void:
    if pulse_time > 0.0:
        var flash := Color(0.21, 1.0, 0.61, pulse_time * 0.42) if pulse_success else Color(1.0, 0.39, 0.40, pulse_time * 0.42)
        draw_rect(Rect2(-55, -320, 110, 640), flash)
    draw_rect(Rect2(-50, -320, 100, 640), Color(0.17, 0.83, 1.0, 0.07))
    draw_rect(Rect2(-36, -320, 72, 640), Color(0.11, 0.42, 0.78, 0.52))
    for i in range(11):
        var y := -306.0 + float(i) * 61.0
        var offset := 9.0 * sin(visual_clock * 1.8 + float(i))
        draw_line(Vector2(-22 + offset, y), Vector2(22 + offset, y - 26),
                Color(0.47, 0.96, 1.0, 0.55), 2.0)
    draw_line(Vector2(-36, -320), Vector2(-36, 320), Color(0.63, 0.98, 1.0), 3.0)
    draw_line(Vector2(36, -320), Vector2(36, 320), Color(0.63, 0.98, 1.0), 3.0)


func transmission() -> Dictionary:
    return MODEL.evaluate(incident_energy, barrier_height, model_width, kinetic_coefficient)


func is_near_player(player: CharacterBody2D) -> bool:
    return absf(player.global_position.x - global_position.x) <= WIDTH_PIXELS * 0.5 + REACH_FROM_FACE


func exit_position(player: CharacterBody2D) -> Vector2:
    var side := 1.0 if player.global_position.x < global_position.x else -1.0
    # > half-width of player plus a small safety margin beyond the barrier.
    return Vector2(global_position.x + side * (WIDTH_PIXELS * 0.5 + 43.0),
            player.global_position.y - 4.0)


func exit_is_clear(player: CharacterBody2D, destination: Vector2) -> bool:
    var collider := player.get_node_or_null("CollisionShape2D") as CollisionShape2D
    if collider == null or collider.shape == null:
        return false
    var query := PhysicsShapeQueryParameters2D.new()
    query.shape = collider.shape
    query.transform = Transform2D(0.0, destination)
    query.collision_mask = player.collision_mask
    query.exclude = [player.get_rid()]
    query.collide_with_bodies = true
    query.collide_with_areas = false
    return get_world_2d().direct_space_state.intersect_shape(query, 8).is_empty()


func try_tunnel(player: CharacterBody2D, coherence: float,
        rng: RandomNumberGenerator) -> Dictionary:
    if not is_near_player(player):
        return {"attempted": false, "success": false, "cost": 0.0,
                "reason": "Move closer to the quantum barrier."}
    var destination := exit_position(player)
    if not exit_is_clear(player, destination):
        return {"attempted": false, "success": false, "cost": 0.0,
                "reason": "Exit space is occupied; tunneling cancelled safely."}
    var result: Dictionary = MODEL.attempt(incident_energy, barrier_height, model_width,
            kinetic_coefficient, coherence, MODEL.DEFAULT_COST, rng)
    if result["attempted"]:
        show_attempt(result["success"])
    if result["success"]:
        result["destination"] = destination
    return result
