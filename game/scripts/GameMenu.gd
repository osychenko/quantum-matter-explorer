extends Control
## G8 home screen / room selector. Progress is read only; game saves are not
## modified by entering the menu. No third-party textures or fonts required.

const PROGRESSION = preload("res://scripts/ProgressionState.gd")
const FLOW = preload("res://scripts/GameFlow.gd")

@export var persistence_enabled := true
var progression = PROGRESSION.new()
var ruins_button: Button
var calibration_button: Button
var progress_label: Label
var _clock := 0.0


func _ready() -> void:
    mouse_filter = Control.MOUSE_FILTER_PASS
    if persistence_enabled:
        progression.load_from_path()
    _build_ui()
    refresh_progress()


func _process(delta: float) -> void:
    _clock += delta
    queue_redraw()


func _draw() -> void:
    var bounds := get_viewport_rect().size
    draw_rect(Rect2(Vector2.ZERO, bounds), Color("#091424"))
    for layer in range(5):
        var rise := float(layer) * 85.0
        var points := PackedVector2Array()
        for step in range(37):
            var x := float(step) / 36.0 * bounds.x
            var y := bounds.y * 0.88 - rise + 17.0 * sin(x / 105.0 + float(layer) + _clock * 0.09)
            points.append(Vector2(x, y))
        draw_polyline(points, Color(0.16, 0.52, 0.69, 0.12 + 0.04 * float(layer)), 2.0)
    for i in range(54):
        var x := fposmod(float(i * 233 + 29), bounds.x)
        var y := fposmod(float(i * 97 + 83), bounds.y * 0.73)
        var glow := 0.24 + 0.18 * sin(_clock * 1.2 + float(i))
        draw_circle(Vector2(x, y), 1.6 + glow, Color(0.66, 0.89, 1.0, glow))
    # A stylized potential barrier and incoming quantum wave.
    var origin := Vector2(bounds.x * 0.67, bounds.y * 0.35)
    draw_rect(Rect2(origin.x + 120, origin.y - 95, 90, 190), Color(0.19, 0.61, 0.90, 0.22))
    draw_rect(Rect2(origin.x + 120, origin.y - 95, 90, 190), Color("#6bcce8"), false, 2.0)
    var wave := PackedVector2Array()
    for step in range(160):
        var px := float(step) * 2.7
        var attenuation := exp(-maxf(0.0, px - 120.0) / 110.0)
        wave.append(origin + Vector2(px - 100.0,
                37.0 * sin(float(step) * 0.18 - _clock * 2.3) * attenuation))
    draw_polyline(wave, Color("#95f2dd"), 3.0, true)


func _make_label(parent: Node, value: String, at: Vector2, size_px: int,
        tint: Color, width: float = 720.0) -> Label:
    var label := Label.new()
    label.text = value
    label.position = at
    label.custom_minimum_size = Vector2(width, 40)
    label.add_theme_font_size_override("font_size", size_px)
    label.add_theme_color_override("font_color", tint)
    parent.add_child(label)
    return label


func _style_button(button: Button) -> void:
    button.custom_minimum_size = Vector2(385, 58)
    button.add_theme_font_size_override("font_size", 21)
    button.add_theme_color_override("font_color", Color("#ddfaf7"))
    button.add_theme_color_override("font_disabled_color", Color("#839aa9"))
    for style_name in ["normal", "hover", "pressed", "disabled"]:
        var style := StyleBoxFlat.new()
        style.bg_color = Color("#172f45")
        if style_name == "hover":
            style.bg_color = Color("#24546c")
        elif style_name == "pressed":
            style.bg_color = Color("#214c50")
        elif style_name == "disabled":
            style.bg_color = Color("#17212e")
        style.border_color = Color("#78d8d5") if style_name != "disabled" else Color("#364b5c")
        style.set_border_width_all(2)
        style.set_corner_radius_all(9)
        button.add_theme_stylebox_override(style_name, style)


func _build_ui() -> void:
    _make_label(self, "QUANTUM MATTER EXPLORER", Vector2(58, 60), 38, Color("#e4fff7"))
    _make_label(self, "BARRIER RUINS  /  QUANTUM EXPERIMENTS", Vector2(60, 112),
            17, Color("#91bac9"))
    _make_label(self, "Discover. Experiment. Understand.", Vector2(60, 171),
            23, Color("#a5f2dc"))

    var actions := VBoxContainer.new()
    actions.position = Vector2(60, 254)
    actions.add_theme_constant_override("separation", 16)
    add_child(actions)
    ruins_button = Button.new()
    ruins_button.text = "PLAY / CONTINUE BARRIER RUINS"
    _style_button(ruins_button)
    actions.add_child(ruins_button)
    ruins_button.pressed.connect(func() -> void:
        FLOW.navigate(get_tree(), FLOW.RUINS)
    )
    calibration_button = Button.new()
    calibration_button.text = "CALIBRATION CHAMBER"
    _style_button(calibration_button)
    actions.add_child(calibration_button)
    calibration_button.pressed.connect(func() -> void:
        if FLOW.calibration_unlocked(progression):
            FLOW.navigate(get_tree(), FLOW.CALIBRATION)
    )
    var quit_button := Button.new()
    quit_button.text = "QUIT"
    _style_button(quit_button)
    actions.add_child(quit_button)
    quit_button.pressed.connect(func() -> void:
        get_tree().quit()
    )

    progress_label = _make_label(self, "", Vector2(61, 510), 16,
            Color("#c6dce6"), 790.0)
    _make_label(self, "A/D OR ARROWS  MOVE    ·    SPACE  JUMP    ·    H  RETURN TO MENU",
            Vector2(60, 673), 15, Color("#85b7c0"), 1140.0)


func refresh_progress() -> void:
    if calibration_button == null:
        return
    var accessible: bool = FLOW.calibration_unlocked(progression)
    calibration_button.disabled = not accessible
    calibration_button.text = ("CALIBRATION CHAMBER  /  UNLOCKED" if accessible
            else "CALIBRATION CHAMBER  /  LOCKED")
    var count: int = int(progression.unlocked.size())
    var chambers: String = "COMPLETE" if progression.calibration_chamber_cleared else "IN PROGRESS"
    progress_label.text = ("Abilities discovered: %d / 3    ·    Barrier Ruins: %s\n"
            + "Calibration: %s    ·    Saves are local to this computer") % [
            count, "CLEARED" if progression.chamber_cleared else "EXPLORING", chambers]
