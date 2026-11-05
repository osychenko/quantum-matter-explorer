extends Node2D
## A single-room vertical slice, with original programmatic placeholder art.
## The educational calculation is deliberately local and does not call FastAPI.

const WORLD_WIDTH := 2000.0
const FLOOR_TOP := 640.0
const START_POSITION := Vector2(210, 595)
const CHECKPOINT_POSITION := Vector2(1160, 595)
const PROGRESSION = preload("res://scripts/ProgressionState.gd")
const ECHO_GHOST = preload("res://scripts/EchoGhost.gd")
const GATE_OPEN_SECONDS := 4.0
const ECHO_SWITCH_RANGE := 43.0

@export var persistence_enabled := true
var progression = PROGRESSION.new()

@onready var player: Variant = $Player
@onready var barrier: Variant = $Barrier
@onready var barrier2: Variant = $Barrier2
@onready var checkpoint: Area2D = $Checkpoint
@onready var hazard: Area2D = $Hazard
@onready var goal: Area2D = $Goal
@onready var ability_relic: Area2D = $AbilityRelic
@onready var echo_relic: Area2D = $EchoRelic
@onready var echo_switch: Area2D = $EchoSwitch
@onready var echo_gate: StaticBody2D = $EchoGate
var echo_ghost: Node2D = null
var gate_open_remaining := 0.0
var echo_gate_open := false
var respawn_position := START_POSITION
var checkpoint_activated := false

var coherence_bar: ProgressBar
var status_text: Label
var hint_text: Label
var science_panel: PanelContainer
var science_body: Label
var inventory_panel: PanelContainer
var inventory_body: Label
var win := false
var _f_was_pressed := false
var _i_was_pressed := false
var _n_was_pressed := false
var _status_time := 0.0


func _ready() -> void:
    _make_solid("Ground", Vector2(WORLD_WIDTH / 2.0, 690), Vector2(WORLD_WIDTH, 100))
    # Invisible walls prevent the player from leaving the playable world.
    _make_solid("LeftBoundary", Vector2(-20, 360), Vector2(40, 1200))
    _make_solid("RightBoundary", Vector2(WORLD_WIDTH + 20, 360), Vector2(40, 1200))
    _make_solid("LeftLedge", Vector2(450, 550), Vector2(160, 22))
    _make_solid("RightLedge", Vector2(1400, 535), Vector2(180, 22))
    player.barrier = barrier
    player.additional_barriers.append(barrier2)
    ability_relic.body_entered.connect(_on_relic_collected)
    echo_relic.body_entered.connect(_on_echo_relic_collected)
    player.echo_requested.connect(_on_echo_requested)
    player.echo_result.connect(_on_echo_result)
    checkpoint.body_entered.connect(_on_checkpoint)
    hazard.body_entered.connect(_on_hazard)
    player.tunneling_result.connect(_on_tunneling_result)
    goal.body_entered.connect(_on_goal_reached)
    _make_hud()
    if persistence_enabled:
        progression.load_from_path()
    _restore_progression()
    if player.echoes_unlocked:
        _show_status("Echoes restored. Press Q to replay motion at the violet switch.", 6.0)
    elif player.tunneling_unlocked:
        _show_status("Progress restored. Cross barrier two to find the Echo shard.", 5.0)
    else:
        _show_status("Find the glowing quantum core to unlock tunneling. I: inventory.", 6.0)


func _physics_process(delta: float) -> void:
    gate_open_remaining = maxf(0.0, gate_open_remaining - delta)
    if is_instance_valid(echo_ghost) and not echo_ghost.is_queued_for_deletion():
        if echo_ghost.global_position.distance_to(echo_switch.global_position) <= ECHO_SWITCH_RANGE:
            gate_open_remaining = GATE_OPEN_SECONDS
    # Never re-enable a collider with the player in its doorway.
    if echo_gate_open and absf(player.global_position.x - echo_gate.global_position.x) < 42.0:
        gate_open_remaining = maxf(gate_open_remaining, 0.2)
    _set_echo_gate_open(gate_open_remaining > 0.0)


func _set_echo_gate_open(opened: bool) -> void:
    if echo_gate_open == opened:
        return
    echo_gate_open = opened
    var collider := echo_gate.get_node("CollisionShape2D") as CollisionShape2D
    collider.set_deferred("disabled", opened)
    queue_redraw()


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
    if not player.tunneling_unlocked:
        # A visible collectible, paired with the AbilityRelic Area2D trigger.
        draw_circle(Vector2(640, 595), 38, Color(0.48, 0.77, 1.0, 0.16))
        draw_arc(Vector2(640, 595), 30, 0, TAU, 48, Color("#7fe2fc"), 3.0)
        draw_colored_polygon(PackedVector2Array([
                Vector2(640, 573), Vector2(657, 595),
                Vector2(640, 617), Vector2(623, 595)]), Color("#87d8ff"))
        draw_circle(Vector2(640, 595), 7, Color("#e4fff8"))
    if not player.echoes_unlocked:
        draw_circle(Vector2(1570, 595), 35, Color(0.67, 0.48, 1.0, 0.19))
        draw_arc(Vector2(1570, 595), 27, 0, TAU, 42, Color("#bb91ff"), 3.0)
        draw_colored_polygon(PackedVector2Array([
                Vector2(1570, 575), Vector2(1583, 595),
                Vector2(1570, 615), Vector2(1557, 595)]), Color("#c9a8ff"))
    draw_rect(Rect2(1580, 627, 60, 10), Color("#74569b"))
    draw_line(Vector2(1580, 627), Vector2(1640, 627),
            Color("#e3baff") if echo_gate_open else Color("#ac83db"), 3.0)
    if echo_gate_open:
        draw_rect(Rect2(1680, 0, 40, 640), Color(0.45, 0.95, 0.85, 0.10))
        draw_line(Vector2(1700, 0), Vector2(1700, 640), Color("#70eecb"), 2.0)
    else:
        draw_rect(Rect2(1680, 0, 40, 640), Color(0.55, 0.31, 0.86, 0.54))
        draw_line(Vector2(1680, 0), Vector2(1680, 640), Color("#e3bdff"), 2.0)
        draw_line(Vector2(1720, 0), Vector2(1720, 640), Color("#e3bdff"), 2.0)
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
    hint_text = _make_label(root, "A/D MOVE   ·   SPACE JUMP   ·   E TUNNEL   ·   Q ECHO   ·   F SCIENCE   ·   I INVENTORY",
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

    var inventory := PanelContainer.new()
    inventory.name = "AbilityInventory"
    inventory.position = Vector2(24, 182)
    inventory.custom_minimum_size = Vector2(610, 350)
    inventory.mouse_filter = Control.MOUSE_FILTER_IGNORE
    var inventory_style := StyleBoxFlat.new()
    inventory_style.bg_color = Color(0.07, 0.14, 0.23, 0.97)
    inventory_style.border_color = Color("#8d9ee8")
    inventory_style.set_border_width_all(2)
    inventory_style.set_corner_radius_all(8)
    inventory_style.set_content_margin_all(17)
    inventory.add_theme_stylebox_override("panel", inventory_style)
    inventory_body = Label.new()
    inventory_body.custom_minimum_size = Vector2(570, 315)
    inventory_body.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
    inventory_body.add_theme_font_size_override("font_size", 17)
    inventory_body.add_theme_color_override("font_color", Color("#dcebf7"))
    inventory.add_child(inventory_body)
    root.add_child(inventory)
    inventory_panel = inventory
    inventory_panel.visible = false
    _refresh_inventory()


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


func _refresh_echo_science() -> void:
    science_body.text = ("QMC ECHOES  ·  PLAYBACK PUZZLE\n"
            + "A violet ghost replays your last ~2 seconds of motion.\n"
            + "Only an echo can activate the violet switch.\n"
            + "Quantum Monte Carlo uses statistical sampling to\n"
            + "estimate observables for quantum systems.\n"
            + "This replay is NOT QMC, a physical clone, or\n"
            + "a quantum-mechanical prediction.  [F] close")


func _process(delta: float) -> void:
    # Recover from falling outside the playable area.
    if player.global_position.y > 900.0:
        player.global_position = respawn_position
        player.velocity = Vector2.ZERO
        _show_status("Fell outside the world — returned to checkpoint.")
    coherence_bar.value = player.coherence
    var active: Variant = player.get_nearby_barrier()
    var near: bool = active != null
    var probability: Dictionary = active.transmission() if near else barrier.transmission()
    if win:
        hint_text.text = "CHAMBER CLEARED  ·  PHASE G4 COMPLETE  ·  I INVENTORY"
    elif not player.tunneling_unlocked:
        hint_text.text = "TUNNELING LOCKED  ·  FIND THE GLOWING CORE  ·  I INVENTORY"
    elif near:
        hint_text.text = ("E TUNNEL  ·  T = %.1f%%" % (100.0 * probability.get("probability", 0.0))) \
                + "  ·  COST 18 COHERENCE  ·  F SCIENCE  ·  I INVENTORY"
    elif player.echoes_unlocked and player.global_position.x > 1520.0:
        hint_text.text = "WALK OVER VIOLET PAD  ·  Q REPLAY (22)  ·  CROSS THE OPEN GATE  ·  I INVENTORY"
    elif player.tunneling_unlocked:
        hint_text.text = "A/D MOVE  ·  SPACE JUMP  ·  E TUNNEL  ·  Q ECHO  ·  I INVENTORY"
    var f_pressed := Input.is_physical_key_pressed(KEY_F)
    if f_pressed and not _f_was_pressed:
        science_panel.visible = not science_panel.visible
        if science_panel.visible:
            inventory_panel.visible = false
    _f_was_pressed = f_pressed
    var i_pressed := Input.is_physical_key_pressed(KEY_I)
    if i_pressed and not _i_was_pressed:
        inventory_panel.visible = not inventory_panel.visible
        if inventory_panel.visible:
            science_panel.visible = false
            _refresh_inventory()
    _i_was_pressed = i_pressed
    var n_pressed := Input.is_physical_key_pressed(KEY_N)
    if n_pressed and not _n_was_pressed and inventory_panel.visible \
            and Input.is_key_pressed(KEY_SHIFT):
        _reset_progression()
    _n_was_pressed = n_pressed
    if near:
        _refresh_science_for(active)
    elif player.echoes_unlocked and player.global_position.x > 1510.0:
        _refresh_echo_science()
    if _status_time > 0.0:
        _status_time -= delta
        if _status_time <= 0.0 and not win:
            status_text.text = ""


func _show_status(message: String, duration: float = 3.5) -> void:
    status_text.text = message
    _status_time = duration


func _refresh_inventory() -> void:
    var tunneling_status := "UNLOCKED" if player.tunneling_unlocked else "LOCKED"
    var echo_status := "UNLOCKED" if player.echoes_unlocked else "LOCKED"
    var checkpoint_status := "ACTIVE" if checkpoint_activated else "NOT REACHED"
    var cleared_status := "YES" if progression.echo_gate_cleared else "NO"
    inventory_body.text = ("QUANTUM ABILITIES  /  DISCOVERY JOURNAL\n\n"
            + "1. QUANTUM TUNNELING  [%s]\n" % tunneling_status
            + "   Cross a barrier using analytical 1D transmission.\n"
            + "   Unlock by collecting the blue quantum core.  [E]\n\n"
            + "2. QMC ECHOES  [%s]\n" % echo_status
            + "   Collect violet shard after barrier two; press [Q].\n"
            + "   Replay last ~2 s; ghost opens violet gate (22 Coherence).\n"
            + "   Motion recording only: NOT real QMC or a clone.\n\n"
            + "3. IMAGINARY-TIME REALM  [COMING LATER]\n"
            + "   A future mechanic; imaginary time is mathematical, not travel.\n\n"
            + "Checkpoint: %s    Echo gate cleared: %s\n" % [checkpoint_status, cleared_status]
            + "[I] close    [SHIFT+N] reset local progress")


func _restore_progression() -> void:
    player.tunneling_unlocked = progression.is_unlocked(PROGRESSION.TUNNELING)
    player.echoes_unlocked = progression.is_unlocked(PROGRESSION.ECHOES)
    checkpoint_activated = progression.checkpoint_reached
    respawn_position = CHECKPOINT_POSITION if checkpoint_activated else START_POSITION
    player.global_position = respawn_position
    player.velocity = Vector2.ZERO
    ability_relic.monitoring = not player.tunneling_unlocked
    echo_relic.monitoring = not player.echoes_unlocked
    _refresh_inventory()
    queue_redraw()


func _save_progression() -> void:
    if persistence_enabled and not progression.save_to_path():
        push_warning("Progress could not be saved. Check write access to user://.")


func _on_relic_collected(body: Node2D) -> void:
    if body != player or player.tunneling_unlocked:
        return
    if not progression.unlock(PROGRESSION.TUNNELING):
        return
    player.tunneling_unlocked = true
    ability_relic.set_deferred("monitoring", false)
    _save_progression()
    _refresh_inventory()
    science_panel.visible = true
    inventory_panel.visible = false
    _show_status("QUANTUM TUNNELING UNLOCKED! E near barriers; F explains the physics.", 6.0)
    queue_redraw()


func _on_echo_relic_collected(body: Node2D) -> void:
    if body != player or player.echoes_unlocked:
        return
    if not progression.unlock(PROGRESSION.ECHOES):
        return
    player.echoes_unlocked = true
    echo_relic.set_deferred("monitoring", false)
    _save_progression()
    _refresh_inventory()
    _refresh_echo_science()
    _show_status("QMC ECHOES UNLOCKED! Walk across the violet pad; press Q near the gate.", 7.0)
    queue_redraw()


func _on_echo_requested(recording: Array) -> void:
    # Only recorded player positions affect the puzzle; the ghost has no physics body.
    if is_instance_valid(echo_ghost):
        echo_ghost.queue_free()
    var typed_trace: Array[Vector2] = []
    for sample in recording:
        if sample is Vector2:
            typed_trace.append(sample)
    var ghost := ECHO_GHOST.new()
    ghost.name = "EchoGhost"
    ghost.configure(typed_trace)
    add_child(ghost)
    echo_ghost = ghost


func _on_echo_result(result: Dictionary) -> void:
    if result.get("spawned", false):
        _show_status("ECHO DEPLOYED — recent movement replaying (not a quantum clone).", 3.0)
    else:
        _show_status(str(result.get("reason", "Echo unavailable.")), 3.0)


func _reset_progression() -> void:
    progression.reset()
    player.tunneling_unlocked = false
    player.echoes_unlocked = false
    player.echo_history.clear()
    player.echo_cooldown_remaining = 0.0
    if is_instance_valid(echo_ghost):
        echo_ghost.queue_free()
    echo_ghost = null
    gate_open_remaining = 0.0
    _set_echo_gate_open(false)
    player.coherence = player.MAX_COHERENCE
    player.tunnel_cooldown_remaining = 0.0
    player.velocity = Vector2.ZERO
    player.global_position = START_POSITION
    respawn_position = START_POSITION
    checkpoint_activated = false
    win = false
    ability_relic.set_deferred("monitoring", true)
    echo_relic.set_deferred("monitoring", true)
    science_panel.visible = false
    _save_progression()
    _refresh_inventory()
    _show_status("Progress reset. Collect blue and violet relics to unlock abilities.", 5.0)
    queue_redraw()


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
    progression.chamber_cleared = true
    progression.echo_gate_cleared = true
    _save_progression()
    _refresh_inventory()
    _show_status("CHAMBER CLEARED — Tunneling and Echoes mastered!", 9999.0)


func _on_checkpoint(body: Node2D) -> void:
    if body != player or checkpoint_activated:
        return
    checkpoint_activated = true
    respawn_position = CHECKPOINT_POSITION
    progression.checkpoint_reached = true
    _save_progression()
    _refresh_inventory()
    _show_status("CHECKPOINT ACTIVATED — safe position recorded.", 3.5)
    queue_redraw()


func _on_hazard(body: Node2D) -> void:
    if body != player or win:
        return
    player.global_position = respawn_position
    player.velocity = Vector2.ZERO
    player.coherence = maxf(0.0, player.coherence - 15.0)
    _show_status("QUANTUM INSTABILITY — returned to checkpoint (-15 Coherence).", 4.0)
