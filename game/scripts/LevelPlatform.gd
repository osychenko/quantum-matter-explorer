@tool
extends StaticBody2D
## Editor-placeable solid block. Change Position or Dimensions in Godot's Inspector.
## Visible geometry and collision use the same centered rectangle.

@export var dimensions: Vector2 = Vector2(160.0, 22.0):
    set(value):
        dimensions = Vector2(maxf(value.x, 16.0), maxf(value.y, 8.0))
        if is_inside_tree():
            _sync_geometry()
        queue_redraw()

@export var fill_color: Color = Color("#234257"):
    set(value):
        fill_color = value
        queue_redraw()

@export var edge_color: Color = Color("#8bd3d1"):
    set(value):
        edge_color = value
        queue_redraw()

@export var gate_style := false:
    set(value):
        gate_style = value
        queue_redraw()

@export var initially_solid := true
var solid_active := true


func _ready() -> void:
    _sync_geometry()
    solid_active = initially_solid
    if not Engine.is_editor_hint():
        set_solid(initially_solid)
    queue_redraw()


func _sync_geometry() -> void:
    var collider := get_node_or_null("CollisionShape2D") as CollisionShape2D
    if collider == null:
        return
    var rectangle := collider.shape as RectangleShape2D
    if rectangle != null:
        rectangle.size = dimensions


func set_solid(enabled: bool) -> void:
    solid_active = enabled
    var collider := get_node_or_null("CollisionShape2D") as CollisionShape2D
    if collider != null:
        # Physics-space changes must be deferred (including when a gate closes).
        if Engine.is_editor_hint():
            collider.disabled = not enabled
        else:
            collider.set_deferred("disabled", not enabled)
    queue_redraw()


func _draw() -> void:
    var bounds := Rect2(-dimensions / 2.0, dimensions)
    if solid_active:
        draw_rect(bounds, fill_color)
        if gate_style:
            draw_line(bounds.position, bounds.position + Vector2(0.0, dimensions.y),
                    edge_color, 3.0)
            draw_line(bounds.position + Vector2(dimensions.x, 0.0),
                    bounds.end, edge_color, 3.0)
        else:
            draw_line(bounds.position, bounds.position + Vector2(dimensions.x, 0.0),
                    edge_color, 3.0)
            # Local-coordinate engraved details follow editor drag/resize.
            draw_rect(Rect2(bounds.position + Vector2(0, 5),
                    Vector2(dimensions.x, 3)),
                    Color(edge_color.r, edge_color.g, edge_color.b, 0.18))
            var divisions := mini(32, maxi(1, int(dimensions.x / 55.0)))
            for i in range(1, divisions):
                var x := bounds.position.x + float(i) / float(divisions) * dimensions.x
                draw_line(Vector2(x, bounds.position.y + 11),
                        Vector2(x, bounds.end.y - 5),
                        Color(edge_color.r, edge_color.g, edge_color.b, 0.12), 1.0)
    else:
        draw_rect(bounds, Color(fill_color.r, fill_color.g, fill_color.b, 0.12))
        draw_rect(bounds, Color(edge_color.r, edge_color.g, edge_color.b, 0.48), false, 2.0)
