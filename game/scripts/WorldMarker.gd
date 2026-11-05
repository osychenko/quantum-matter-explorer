@tool
extends Area2D
## Editor-visible visual for a pickup, checkpoint, pressure pad, hazard or goal.
## The drawing uses local coordinates, so dragging the Area2D moves its art too.

@export_enum("tunnel", "echo", "realm", "checkpoint", "realm_checkpoint", "hazard", "realm_hazard", "echo_switch", "milestone", "goal") var marker_kind: String = "tunnel":
    set(value):
        marker_kind = value
        queue_redraw()

var marker_active := false


func set_marker_active(value: bool) -> void:
    marker_active = value
    queue_redraw()


func _draw() -> void:
    match marker_kind:
        "tunnel", "echo", "realm":
            var tint := Color("#87d8ff")
            if marker_kind == "echo":
                tint = Color("#c9a8ff")
            elif marker_kind == "realm":
                tint = Color("#84ffca")
            draw_circle(Vector2.ZERO, 37.0, Color(tint.r, tint.g, tint.b, 0.15))
            draw_arc(Vector2.ZERO, 29.0, 0.0, TAU, 48, tint, 3.0)
            draw_colored_polygon(PackedVector2Array([Vector2(0, -22),
                    Vector2(16, 0), Vector2(0, 22), Vector2(-16, 0)]), tint)
            draw_circle(Vector2.ZERO, 7.0, Color("#e4fff8"))
        "checkpoint", "realm_checkpoint":
            var tint := Color("#70eecb") if marker_active else Color("#567e8b")
            draw_circle(Vector2.ZERO, 26.0, Color(0.31, 0.94, 0.81, 0.20))
            draw_arc(Vector2.ZERO, 22.0, 0.0, TAU, 48, tint, 3.0)
            if marker_kind == "realm_checkpoint":
                draw_line(Vector2(-8, 0), Vector2(8, 0), tint, 3.0)
        "echo_switch":
            var tint := Color("#e3baff") if marker_active else Color("#ac83db")
            draw_rect(Rect2(-30, 22, 60, 10), Color("#74569b"))
            draw_line(Vector2(-30, 22), Vector2(30, 22), tint, 3.0)
        "hazard", "realm_hazard":
            var collider := get_node_or_null("CollisionShape2D") as CollisionShape2D
            var width := 72.0
            if collider != null and collider.shape is RectangleShape2D:
                var shape := collider.shape as RectangleShape2D
                width = shape.size.x
            var count := maxi(1, int(width / 14.0))
            var left := -width / 2.0
            var tint := Color("#ee6b80") if marker_kind == "hazard" else Color("#70f0c5")
            for i in range(count):
                var x := left + float(i) * width / float(count)
                var step := width / float(count)
                draw_colored_polygon(PackedVector2Array([Vector2(x, 20),
                        Vector2(x + step / 2.0, -27), Vector2(x + step, 20)]), tint)
        "milestone":
            draw_arc(Vector2.ZERO, 31.0, 0.0, TAU, 40, Color("#a2c4ff"), 3.0)
            draw_circle(Vector2.ZERO, 9.0, Color("#7a9bc9"))
        "goal":
            draw_circle(Vector2.ZERO, 45, Color(0.31, 0.94, 0.81, 0.13))
            draw_arc(Vector2.ZERO, 31, 0.0, TAU, 50, Color("#70eecb"), 4.0)
            draw_circle(Vector2.ZERO, 13, Color("#c4ffdc"))
            draw_arc(Vector2.ZERO, 51, 0.1, 4.1, 60,
                    Color(0.53, 1.0, 0.9, 0.23), 3.0)
