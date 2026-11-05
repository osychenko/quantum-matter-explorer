extends Node2D
## Visual-only replay of a short recorded player trajectory.
## No collision, player agency, QMC sampling, or quantum cloning.

var samples: Array[Vector2] = []
var playback_index := 0


func configure(recording: Array[Vector2]) -> void:
    samples.clear()
    for point in recording:
        samples.append(point)
    playback_index = 0
    if not samples.is_empty():
        global_position = samples[0]


func _physics_process(_delta: float) -> void:
    if playback_index >= samples.size():
        queue_free()
        return
    global_position = samples[playback_index]
    playback_index += 1
    queue_redraw()


func _draw() -> void:
    # Distinct from the player: translucent violet historical trace.
    draw_circle(Vector2(0, -5), 28, Color(0.62, 0.43, 0.98, 0.13))
    draw_arc(Vector2(0, -5), 24, 0, TAU, 42, Color(0.76, 0.61, 1.0, 0.67), 2.0)
    draw_colored_polygon(PackedVector2Array([
            Vector2(0, -31), Vector2(15, -6),
            Vector2(11, 15), Vector2(-11, 15),
            Vector2(-15, -6)]), Color(0.70, 0.52, 1.0, 0.5))
    draw_circle(Vector2(0, -13), 9, Color(0.89, 0.8, 1.0, 0.74))
