@tool
extends Node2D
## Editor-movable research console. Local lab settings never modify this source.

@export var incident_energy := 0.8
@export var barrier_height := 1.2
@export var model_width := 0.8
@export var kinetic_coefficient := 0.5

var completed := false


func set_completed(value: bool) -> void:
    completed = value
    queue_redraw()


func _draw() -> void:
    var active := Color("#77ffd0") if completed else Color("#77c6ff")
    draw_rect(Rect2(-36, -56, 72, 112), Color("#15304a"))
    draw_rect(Rect2(-30, -47, 60, 78), Color("#234b64"))
    draw_rect(Rect2(-25, -40, 50, 66), Color(active.r, active.g, active.b, 0.14))
    draw_line(Vector2(-22, -20), Vector2(21, -20), active, 2.0)
    draw_line(Vector2(-22, -7), Vector2(5, -7), active, 2.0)
    draw_line(Vector2(-22, 6), Vector2(13, 6), active, 2.0)
    draw_circle(Vector2(0, 42), 7.0, active)
