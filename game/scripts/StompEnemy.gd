@tool
extends Area2D
## G9: optional training enemy. Land on its head while falling to defeat it.
## Side contact is a hazard. Physics and art follow the same editable dimensions.

signal stomped(body: Node2D)
signal player_hit(body: Node2D)

@export var dimensions := Vector2(44.0, 34.0):
    set(value):
        dimensions = Vector2(maxf(20.0, value.x), maxf(16.0, value.y))
        if is_inside_tree():
            _sync_shape()
        queue_redraw()

const MIN_STOMP_FALL_SPEED := 70.0
const HEAD_TOLERANCE_PX := 15.0
var defeated := false


func _ready() -> void:
    _sync_shape()
    if not Engine.is_editor_hint():
        body_entered.connect(_on_body_entered)
    queue_redraw()


func _sync_shape() -> void:
    var collider := get_node_or_null("CollisionShape2D") as CollisionShape2D
    if collider == null:
        return
    var rectangle := collider.shape as RectangleShape2D
    if rectangle == null:
        rectangle = RectangleShape2D.new()
        collider.shape = rectangle
    rectangle.size = dimensions


func _on_body_entered(body: Node2D) -> void:
    if defeated or not body is CharacterBody2D:
        return
    var character := body as CharacterBody2D
    var player_half_height := 22.0
    var player_collider := character.get_node_or_null("CollisionShape2D") as CollisionShape2D
    if player_collider != null and player_collider.shape is RectangleShape2D:
        player_half_height = (player_collider.shape as RectangleShape2D).size.y * 0.5
    var player_feet := character.global_position.y + player_half_height
    var enemy_head := global_position.y - dimensions.y * 0.5
    var from_above := player_feet <= enemy_head + HEAD_TOLERANCE_PX
    if character.velocity.y >= MIN_STOMP_FALL_SPEED and from_above \
            and character.has_method("stomp_bounce"):
        defeated = true
        set_deferred("monitoring", false)
        character.call("stomp_bounce")
        stomped.emit(character)
        queue_redraw()
        # Keep the defeated outline visible for one brief beat.
        get_tree().create_timer(0.18).timeout.connect(queue_free)
    else:
        player_hit.emit(character)


func _draw() -> void:
    var half := dimensions * 0.5
    if defeated:
        draw_rect(Rect2(-half.x, -half.y + 12, dimensions.x, dimensions.y - 12),
                Color(0.28, 0.47, 0.55, 0.35))
        return
    draw_rect(Rect2(-half, dimensions), Color("#4f6c8e"))
    draw_rect(Rect2(Vector2(-half.x, -half.y), Vector2(dimensions.x, 7.0)),
            Color("#d3f57a"))
    draw_circle(Vector2(-9.0, -1.0), 3.3, Color("#f7fbfd"))
    draw_circle(Vector2(9.0, -1.0), 3.3, Color("#f7fbfd"))
    draw_circle(Vector2(-8.5, -1.0), 1.2, Color("#14253b"))
    draw_circle(Vector2(9.5, -1.0), 1.2, Color("#14253b"))
