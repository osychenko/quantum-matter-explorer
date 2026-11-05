extends HBoxContainer
## Compact, read-only ability bar shared by the two rooms.

var player: Variant = null
var _labels: Array[Label] = []
var _last_display := ""


func _ready() -> void:
    add_theme_constant_override("separation", 8)
    for i in range(3):
        var panel := PanelContainer.new()
        var style := StyleBoxFlat.new()
        style.bg_color = Color(0.07, 0.13, 0.24, 0.85)
        style.border_color = [Color("#70d7fa"), Color("#c1a0ff"),
                Color("#76eac7")][i]
        style.set_border_width_all(1)
        style.set_corner_radius_all(5)
        style.set_content_margin_all(7)
        panel.add_theme_stylebox_override("panel", style)
        var label := Label.new()
        label.add_theme_font_size_override("font_size", 13)
        label.add_theme_color_override("font_color", Color("#e8f6f3"))
        label.custom_minimum_size = Vector2(125, 25)
        panel.add_child(label)
        add_child(panel)
        _labels.append(label)
    refresh_display()


func _process(_delta: float) -> void:
    refresh_display()


func refresh_display() -> void:
    if player == null or _labels.size() != 3:
        return
    var values := [
        _status("E", "TUNNEL", player.tunneling_unlocked, player.tunnel_cooldown_remaining),
        _status("Q", "ECHO", player.echoes_unlocked, player.echo_cooldown_remaining),
        _status("R", "REALM", player.imaginary_time_unlocked,
                player.realm_cooldown_remaining),
    ]
    var display: String = values[0] + " | " + values[1] + " | " + values[2]
    if display == _last_display:
        return
    _last_display = display
    for i in range(3):
        _labels[i].text = values[i]


func _status(key: String, ability: String, unlocked: bool, cooldown: float) -> String:
    if not unlocked:
        return "%s %s  /  LOCKED" % [key, ability]
    if cooldown > 0.05:
        return "%s %s  /  %.1fs" % [key, ability, cooldown]
    return "%s %s  /  READY" % [key, ability]
