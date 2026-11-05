extends Node2D
## A single-room vertical slice, with original programmatic placeholder art.
## The educational calculation is deliberately local and does not call FastAPI.

const WORLD_WIDTH := 2000.0
const FLOOR_TOP := 640.0

@onready var player: Variant = $Player
@onready var barrier: Variant = $Barrier
@onready var barrier2: Variant = $Barrier2
@onready var checkpoint: Area2D = $Checkpoint
@onready var hazard: Area2D = $Hazard
@onready var goal: Area2D = $Goal
var respawn_position := Vector2(210, 595)
var checkpoint_activated := false

var coherence_bar: ProgressBar
var status_text: Label
var hint_text: Label
var science_panel: PanelContainer
var science_body: Label
var win := false
var _f_was_pressed := false
var _status_time := 0.0
var _discovery_announced := false


func _ready() -> void:
    _make_solid("Ground", Vector2(WORLD_WIDTH / 2.0, 690), Vector2(WORLD_WIDTH, 100))
    _make_solid("LeftLedge", Vector2(450, 550), Vector2(160, 22))
    _make_solid("RightLedge", Vector2(1400, 535), Vector2(180, 22))
    player.barrier = barrier
    player.additional_barriers.append(barrier2)
    checkpoint.body_entered.connect(_on_checkpoint)
    hazard.body_entered.connect(_on_hazard)
    player.tunneling_result.connect(_on_tunneling_result)
    goal.body_entered.connect(_on_goal_reached)
    _make_hud()
    _show_status("Cross two barriers, touch the checkpoint, avoid the red instability.", 6.0)


func _make_solid(label: String, place: Vector2, size: Vector2) -> void:
    var body := StaticBody2D.new()
    body.name = label
    body.position = place
    body.collision_layer = 1
    body.collision_mask = 0
    var shape := RectangleShape2D.new()
    shape.size = size
    var collider := CollisionShape2D.new()
    collider.shape = shape
    body.add_child(collider)
    add_child(body)


func _draw() -> void:
    draw_rect(Rect2(0, 0, WORLD_WIDTH, 720), Color("#071325"))
    # Sparse distant structures and lights suggest a mysterious science-fantasy ruin.
    for i in range(18):
        var xx := float(i) * 119.0 + 24.0
        var yy := 105.0 + 31.0 * sin(float(i) * 2.13)
        draw_circle(Vector2(xx, yy), 1.9, Color(0.6, 0.9, 1.0, 0.6))
    for i in range(11):
        var x := float(i) * 206.0 - 26.0
        var h := 110.0 + 52.0 * sin(float(i) * 1.7)
        draw_rect(Rect2(x, FLOOR_TOP - h, 82, h), Color("#0d2034"))
        draw_rect(Rect2(x + 12, FLOOR_TOP - h - 18, 58, 18), Color("#132c44"))
        draw_line(Vector2(x + 41, FLOOR_TOP - h + 10),
                Vector2(x + 41, FLOOR_TOP - 12), Color("#1f4261"), 2.0)
    draw_rect(Rect2(0, FLOOR_TOP, WORLD_WIDTH, 80), Color("#122538"))
    draw_line(Vector2(0, FLOOR_TOP), Vector2(WORLD_WIDTH, FLOOR_TOP), Color("#78c6ba"), 3.0)
    for x in range(0, 2000, 54):
        draw_line(Vector2(x, FLOOR_TOP + 21), Vector2(x + 28, FLOOR_TOP + 30),
                Color("#2a5362"), 2.0)
    for ledge in [Rect2(370, 539, 160, 22), Rect2(1310, 524, 180, 22)]:
        draw_rect(ledge, Color("#234257"))
        draw_line(ledge.position, ledge.position + Vector2(ledge.size.x, 0),
                Color("#8bd3d1"), 3.0)
    draw_circle(Vector2(1160, 585), 26, Color(0.31, 0.94, 0.81, 0.22))
    draw_arc(Vector2(1160, 585), 22, 0, TAU, 48,
            Color("#70eecb") if checkpoint_activated else Color("#567e8b"), 3.0)
    for i in range(5):
        var px := 1272.0 + float(i) * 14.0
        draw_colored_polygon(PackedVector2Array([Vector2(px, 640),
                Vector2(px + 7.0, 593), Vector2(px + 14.0, 640)]), Color("#ee6b80"))
    draw_circle(Vector2(1770, 585), 45, Color(0.31, 0.94, 0.81, 0.13))
    draw_arc(Vector2(1770, 585), 31, 0, TAU, 50, Color("#70eecb"), 4.0)
    draw_circle(Vector2(1770, 585), 13, Color("#c4ffdc"))
    draw_arc(Vector2(1770, 585), 51, 0.1, 4.1, 60, Color(0.53, 1.0, 0.9, 0.23), 3.0)


func _make_label(parent: Node, text_value: String, where: Vector2,
        size: Vector2, font_size: int, color: Color) -> Label:
    var label := Label.new()
    label.text = text_value
    label.position = where
    label.size = size
    label.add_theme_font_size_override("font_size", font_size)
    label.add_theme_color_override("font_color", color)
    parent.add_child(label)
    return label


func _make_hud() -> void:
    var layer := CanvasLayer.new()
    layer.name = "HUD"
    add_child(layer)
    var root := Control.new()
    root.name = "Overlay"
    root.set_anchors_preset(Control.PRESET_FULL_RECT)
    root.mouse_filter = Control.MOUSE_FILTER_IGNORE
    layer.add_child(root)
    _make_label(root, "QUANTUM MATTER EXPLORER   /   BARRIER RUINS", Vector2(24, 19),
            Vector2(640, 35), 22, Color("#d9f8ea"))
    _make_label(root, "COHERENCE", Vector2(24, 65), Vector2(200, 26),
            15, Color("#89c7d1"))
    coherence_bar = ProgressBar.new()
    coherence_bar.position = Vector2(24, 93)
    coherence_bar.custom_minimum_size = Vector2(255, 18)
    coherence_bar.size = Vector2(255, 18)
    coherence_bar.max_value = player.MAX_COHERENCE
    coherence_bar.value = player.coherence
    coherence_bar.show_percentage = false
    var bg := StyleBoxFlat.new()
    bg.bg_color = Color("#18344a")
    bg.set_corner_radius_all(5)
    var fg := StyleBoxFlat.new()
    fg.bg_color = Color("#64dbbb")
    fg.set_corner_radius_all(5)
    coherence_bar.add_theme_stylebox_override("background", bg)
    coherence_bar.add_theme_stylebox_override("fill", fg)
    root.add_child(coherence_bar)
    status_text = _make_label(root, "", Vector2(24, 130), Vector2(720, 28),
            17, Color("#e6e2bc"))
    hint_text = _make_label(root, "A/D or ←/→  MOVE    ·    SPACE  JUMP    ·    E  TUNNEL    ·    F  SCIENCE",
            Vector2(24, 675), Vector2(1210, 30), 16, Color("#bbdce0"))
    var panel := PanelContainer.new()
    panel.name = "SciencePanel"
    panel.position = Vector2(765, 22)
    panel.custom_minimum_size = Vector2(485, 230)
    panel.mouse_filter = Control.MOUSE_FILTER_IGNORE
    var box := StyleBoxFlat.new()
    box.bg_color = Color(0.06, 0.13, 0.22, 0.95)
    box.border_color = Color("#53bfc3")
    box.set_border_width_all(2)
    box.set_corner_radius_all(8)
    box.set_content_margin_all(16)
    panel.add_theme_stylebox_override("panel", box)
    science_body = Label.new()
    science_body.custom_minimum_size = Vector2(450, 220)
    science_body.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
    science_body.add_theme_font_size_override("font_size", 16)
    science_body.add_theme_color_override("font_color", Color("#cee7e8"))
    panel.add_child(science_body)
    root.add_child(panel)
    science_panel = panel
    science_panel.visible = false
    _refresh_science()


func _refresh_science() -> void:
    _refresh_science_for(barrier)


func _refresh_science_for(active: Variant) -> void:
    var model: Dictionary = active.transmission()
    var percent := "invalid" if not model["valid"] else "%.1f%%" % (100.0 * model["probability"])
    science_body.text = ("TUNNELING  ·  ANALYTICAL 1D MODEL\n"
            + "H = -D d²/dx² + V(x)\n"
            + "E = %.2f   V0 = %.2f   a = %.2f   D = %.2f\n" % [active.incident_energy,
                    active.barrier_height, active.model_width, active.kinetic_coefficient]
            + "Transmission T = %s\n" % percent
            + "This is stationary scattering through a rectangular\n"
            + "barrier. No QMC simulation is running.\n"
            + "Coherence is fictional gameplay currency, NOT entropy\n"
            + "or a physical probability budget.  [F] close")


func _process(delta: float) -> void:
    coherence_bar.value = player.coherence
    var active: Variant = player.get_nearby_barrier()
    var near: bool = active != null
    if player.discovered and not _discovery_announced:
        _discovery_announced = true
        science_panel.visible = true
        _show_status("NEW ABILITY: QUANTUM TUNNELING — inspect the physics on the right.", 5.0)
    var probability: Dictionary = active.transmission() if near else barrier.transmission()
    if win:
        hint_text.text = "CHAMBER CLEARED  ·  PHASE G2 COMPLETE  ·  F  SCIENCE"
    elif near:
        hint_text.text = ("BARRIER DISCOVERED  ·  E  ATTEMPT  ·  T = %.1f%%"
                % (100.0 * probability.get("probability", 0.0))) + "  ·  COST 18 COHERENCE  ·  F  INSPECT"
    else:
        hint_text.text = "A/D or ←/→ MOVE  ·  SPACE JUMP  ·  E TUNNEL  ·  F SCIENCE"
    var f_pressed := Input.is_physical_key_pressed(KEY_F)
    if f_pressed and not _f_was_pressed:
        science_panel.visible = not science_panel.visible
    _f_was_pressed = f_pressed
    if near:
        _refresh_science_for(active)
    if _status_time > 0.0:
        _status_time -= delta
        if _status_time <= 0.0 and not win:
            status_text.text = ""


func _show_status(message: String, duration: float = 3.5) -> void:
    status_text.text = message
    _status_time = duration


func _on_tunneling_result(result: Dictionary) -> void:
    if not result.get("attempted", false):
        _show_status(str(result.get("reason", "Cannot tunnel.")))
    elif result["success"]:
        _show_status("TRANSMISSION!  Roll %.3f < T %.3f" % [result["roll"], result["probability"]])
    else:
        _show_status("REFLECTION!  Roll %.3f ≥ T %.3f — try again." % [result["roll"], result["probability"]])


func _on_goal_reached(body: Node2D) -> void:
    if body != player or win:
        return
    win = true
    _show_status("CHAMBER CLEARED — Two barriers crossed!", 9999.0)


func _on_checkpoint(body: Node2D) -> void:
    if body != player or checkpoint_activated:
        return
    checkpoint_activated = true
    respawn_position = Vector2(1160, 595)
    _show_status("CHECKPOINT ACTIVATED — safe position recorded.", 3.5)
    queue_redraw()


func _on_hazard(body: Node2D) -> void:
    if body != player or win:
        return
    player.global_position = respawn_position
    player.velocity = Vector2.ZERO
    player.coherence = maxf(0.0, player.coherence - 15.0)
    _show_status("QUANTUM INSTABILITY — returned to checkpoint (-15 Coherence).", 4.0)
