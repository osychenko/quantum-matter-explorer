extends Node2D
## A single-room vertical slice, with original programmatic placeholder art.
## The educational calculation is deliberately local and does not call FastAPI.

const WORLD_WIDTH := 2880.0
const FLOOR_TOP := 640.0
const START_POSITION := Vector2(210, 595)
const CHECKPOINT_POSITION := Vector2(1160, 595)
const REALM_CHECKPOINT_POSITION := Vector2(2090, 595)
const PROGRESSION = preload("res://scripts/ProgressionState.gd")
const FLOW = preload("res://scripts/GameFlow.gd")
const FX = preload("res://scripts/AbilityVFX.gd")
const STRIP = preload("res://scripts/AbilityStrip.gd")
const ECHO_GHOST = preload("res://scripts/EchoGhost.gd")
const MEASUREMENT_LAB = preload("res://scripts/MeasurementLab.gd")
const GATE_OPEN_SECONDS := 4.0
const ECHO_SWITCH_RANGE := 43.0
const REALM_GATE_SAFETY_RANGE := 48.0

@export var persistence_enabled := true
var progression = PROGRESSION.new()

@onready var player: Variant = $Player
@onready var barrier: Variant = $Barrier
@onready var barrier2: Variant = $Barrier2
@onready var checkpoint: Variant = $Checkpoint
@onready var hazard: Area2D = $Hazard
@onready var goal: Area2D = $Goal
@onready var ability_relic: Area2D = $AbilityRelic
@onready var echo_relic: Area2D = $EchoRelic
@onready var echo_switch: Variant = $EchoSwitch
@onready var echo_gate: Variant = $EchoGate
@onready var imaginary_relic: Area2D = $ImaginaryRelic
@onready var realm_gate: Variant = $RealmGate
@onready var spectral_bridge: Variant = $SpectralBridge
@onready var realm_hazard: Area2D = $RealmHazard
@onready var realm_checkpoint: Variant = $RealmCheckpoint
@onready var echo_milestone: Area2D = $EchoMilestone
var echo_ghost: Node2D = null
var gate_open_remaining := 0.0
var echo_gate_open := false
var realm_environment_active := false
var realm_checkpoint_activated := false
var respawn_position := START_POSITION
var checkpoint_activated := false

var coherence_bar: ProgressBar
var status_text: Label
var hint_text: Label
var science_panel: PanelContainer
var science_body: Label
var inventory_panel: PanelContainer
var inventory_body: Label
var measurement_lab: Variant = null
var ability_fx: Variant = null
var ability_strip: Variant = null
var _h_was_pressed := false
var win := false
var _f_was_pressed := false
var _i_was_pressed := false
var _n_was_pressed := false
var _m_was_pressed := false
var _p_was_pressed := false
var _status_time := 0.0


func _ready() -> void:
    # All ground, ledges, boundaries and gates are editor-managed scene nodes.
    # Their visuals and colliders are driven by LevelPlatform.gd.
    player.barrier = barrier
    player.additional_barriers.append(barrier2)
    ability_relic.body_entered.connect(_on_relic_collected)
    echo_relic.body_entered.connect(_on_echo_relic_collected)
    imaginary_relic.body_entered.connect(_on_imaginary_relic_collected)
    echo_milestone.body_entered.connect(_on_echo_milestone)
    realm_checkpoint.body_entered.connect(_on_realm_checkpoint)
    realm_hazard.body_entered.connect(_on_hazard)
    player.realm_result.connect(_on_realm_result)
    player.echo_requested.connect(_on_echo_requested)
    player.echo_result.connect(_on_echo_result)
    checkpoint.body_entered.connect(_on_checkpoint)
    hazard.body_entered.connect(_on_hazard)
    player.tunneling_result.connect(_on_tunneling_result)
    goal.body_entered.connect(_on_goal_reached)
    _make_hud()
    ability_fx = FX.new()
    ability_fx.name = "AbilityVFX"
    add_child(ability_fx)
    player.tunneling_result.connect(_on_fx_tunneling)
    player.echo_result.connect(_on_fx_echo)
    player.realm_result.connect(_on_fx_realm)
    if persistence_enabled:
        progression.load_from_path()
    _restore_progression()
    if progression.chamber_cleared:
        _show_status("Barrier Ruins cleared: press P to enter the G7 Calibration Chamber.", 8.0)
    elif player.imaginary_time_unlocked:
        _show_status("Realm unlocked: press R near the green gate; cross the spectral bridge.", 6.0)
    elif player.echoes_unlocked:
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
    # Extend the realm briefly if its collider would materialize on top of the player.
    if realm_environment_active and absf(player.global_position.x - realm_gate.global_position.x) \
            < REALM_GATE_SAFETY_RANGE and player.realm_remaining <= 0.0:
        player.realm_remaining = 0.2
    _set_realm_environment(player.is_realm_active())


func _set_echo_gate_open(opened: bool) -> void:
    if echo_gate_open == opened:
        return
    echo_gate_open = opened
    echo_gate.set_solid(not opened)
    echo_switch.set_marker_active(opened)


func _set_realm_environment(active: bool) -> void:
    if realm_environment_active == active:
        return
    realm_environment_active = active
    realm_gate.set_solid(not active)
    spectral_bridge.set_solid(active)
    queue_redraw()


func _draw() -> void:
    # Decorative atmosphere only: ALL physical geometry and game markers are
    # individual draggable nodes in DemoLevel.tscn, not hardcoded coordinates.
    draw_rect(Rect2(0, 0, WORLD_WIDTH, 720), Color("#071325"))
    for i in range(int(WORLD_WIDTH / 119.0) + 1):
        var xx := float(i) * 119.0 + 24.0
        var yy := 105.0 + 31.0 * sin(float(i) * 2.13)
        draw_circle(Vector2(xx, yy), 1.9, Color(0.6, 0.9, 1.0, 0.6))
    for i in range(int(WORLD_WIDTH / 206.0) + 1):
        var x := float(i) * 206.0 - 26.0
        var h := 110.0 + 52.0 * sin(float(i) * 1.7)
        draw_rect(Rect2(x, FLOOR_TOP - h, 82, h), Color("#0d2034"))
        draw_rect(Rect2(x + 12, FLOOR_TOP - h - 18, 58, 18), Color("#132c44"))
    if realm_environment_active:
        draw_rect(Rect2(1860, 0, WORLD_WIDTH - 1860, 720),
                Color(0.15, 0.48, 0.40, 0.12))


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
    inventory.custom_minimum_size = Vector2(620, 485)
    inventory.mouse_filter = Control.MOUSE_FILTER_IGNORE
    var inventory_style := StyleBoxFlat.new()
    inventory_style.bg_color = Color(0.07, 0.14, 0.23, 0.97)
    inventory_style.border_color = Color("#8d9ee8")
    inventory_style.set_border_width_all(2)
    inventory_style.set_corner_radius_all(8)
    inventory_style.set_content_margin_all(17)
    inventory.add_theme_stylebox_override("panel", inventory_style)
    inventory_body = Label.new()
    inventory_body.custom_minimum_size = Vector2(580, 455)
    inventory_body.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
    inventory_body.add_theme_font_size_override("font_size", 16)
    inventory_body.add_theme_color_override("font_color", Color("#dcebf7"))
    inventory.add_child(inventory_body)
    root.add_child(inventory)
    inventory_panel = inventory
    inventory_panel.visible = false
    _refresh_inventory()

    # G6: a modal learning UI. It is independent of save data and physics barriers.
    measurement_lab = MEASUREMENT_LAB.new()
    root.add_child(measurement_lab)
    measurement_lab.visible = false
    measurement_lab.close_requested.connect(_close_measurement_lab)

    ability_strip = STRIP.new()
    ability_strip.name = "AbilityStrip"
    ability_strip.position = Vector2(665, 615)
    ability_strip.player = player
    root.add_child(ability_strip)


func _on_fx_tunneling(result: Dictionary) -> void:
    if result.get("attempted", false):
        ability_fx.play_event("tunnel", player.global_position, result.get("success", false))


func _on_fx_echo(result: Dictionary) -> void:
    if result.get("spawned", false):
        ability_fx.play_event("echo", player.global_position)


func _on_fx_realm(result: Dictionary) -> void:
    if result.get("activated", false):
        ability_fx.play_event("realm", player.global_position)


func _open_measurement_lab() -> bool:
    if measurement_lab.visible:
        return true
    var nearby: Variant = player.get_nearby_barrier()
    if nearby == null:
        _show_status("Approach a blue tunneling barrier, then press M for the lab.", 4.0)
        return false
    var display_name := "BARRIER 1" if nearby == barrier else "BARRIER 2"
    measurement_lab.configure_barrier(nearby, display_name)
    science_panel.visible = false
    inventory_panel.visible = false
    player.velocity = Vector2.ZERO
    player.set_physics_process(false)
    measurement_lab.visible = true
    return true


func _close_measurement_lab() -> void:
    if not measurement_lab.visible:
        return
    measurement_lab.visible = false
    player.set_physics_process(true)


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


func _refresh_realm_science() -> void:
    science_body.text = ("IMAGINARY-TIME PROJECTION  ·  GAMEPLAY METAPHOR\n"
            + "In ground-state QMC, exp[-τ(H - Eref)] suppresses\n"
            + "excited-state amplitudes relative to low-energy ones\n"
            + "(given nonzero ground-state overlap).\n"
            + "Press R: a spectral bridge appears and a gate opens.\n"
            + "This timed realm is FICTIONAL: no real-time travel,\n"
            + "imaginary-time dynamics, or QMC is computed. [F] close")


func _process(delta: float) -> void:
    var m_pressed := Input.is_physical_key_pressed(KEY_M)
    if m_pressed and not _m_was_pressed:
        if measurement_lab.visible:
            _close_measurement_lab()
        else:
            _open_measurement_lab()
    _m_was_pressed = m_pressed
    ability_strip.visible = not measurement_lab.visible
    if measurement_lab.visible:
        if Input.is_action_just_pressed("ui_cancel"):
            _close_measurement_lab()
        # Keep the world stable while adjusting the laboratory's UI controls.
        return
    var h_pressed := Input.is_physical_key_pressed(KEY_H)
    if h_pressed and not _h_was_pressed:
        FLOW.navigate(get_tree(), FLOW.MENU)
        return
    _h_was_pressed = h_pressed
    # Recover from falling outside the playable area.
    if player.global_position.y > 900.0:
        player.global_position = respawn_position
        player.velocity = Vector2.ZERO
        _show_status("Fell outside the world — returned to checkpoint.")
    coherence_bar.value = player.coherence
    var active: Variant = player.get_nearby_barrier()
    var near: bool = active != null
    var probability: Dictionary = active.transmission() if near else barrier.transmission()
    if progression.chamber_cleared:
        hint_text.text = "CHAMBER CLEARED  ·  P CALIBRATION CHAMBER  ·  H MAIN MENU"
    elif not player.tunneling_unlocked:
        hint_text.text = "TUNNELING LOCKED · FIND BLUE CORE · M LAB NEAR BARRIER · I INVENTORY"
    elif near:
        hint_text.text = ("E TUNNEL  ·  T = %.1f%%" % (100.0 * probability.get("probability", 0.0))) \
                + "  ·  M MEASUREMENT LAB  ·  F SCIENCE  ·  I INVENTORY"
    elif player.imaginary_time_unlocked and player.global_position.x > 2100.0:
        hint_text.text = "R REALM (30)  ·  CROSS GREEN GATE  ·  JUMP OVER GREEN SPIKES  ·  I INVENTORY"
    elif player.echoes_unlocked and player.global_position.x > 1520.0:
        hint_text.text = "WALK OVER VIOLET PAD  ·  Q REPLAY (22)  ·  CROSS THE OPEN GATE  ·  I INVENTORY"
    elif player.tunneling_unlocked:
        hint_text.text = "A/D MOVE · SPACE JUMP · E TUNNEL · Q ECHO · R REALM · M LAB · I INVENTORY"
    var p_pressed := Input.is_physical_key_pressed(KEY_P)
    if p_pressed and not _p_was_pressed and progression.chamber_cleared:
        FLOW.navigate(get_tree(), FLOW.CALIBRATION)
        return
    _p_was_pressed = p_pressed
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
    elif player.imaginary_time_unlocked and player.global_position.x > 1900.0:
        _refresh_realm_science()
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
    var realm_status := "UNLOCKED" if player.imaginary_time_unlocked else "LOCKED"
    var checkpoint_status := "ACTIVE" if checkpoint_activated else "NOT REACHED"
    var echo_cleared := "YES" if progression.echo_gate_cleared else "NO"
    var realm_cleared := "YES" if progression.realm_gate_cleared else "NO"
    var calibration_status := "COMPLETE" if progression.calibration_chamber_cleared else ("GATE OPEN" if progression.calibration_gate_cleared else "PENDING")
    inventory_body.text = ("QUANTUM ABILITIES  /  DISCOVERY JOURNAL\n\n"
            + "1. QUANTUM TUNNELING  [%s]\n" % tunneling_status
            + "   Blue core; [E]; exact stationary 1D transmission.\n\n"
            + "2. QMC ECHOES  [%s]\n" % echo_status
            + "   Violet shard; [Q]; replay last ~2 seconds (22 Coherence).\n"
            + "   Recorded gameplay history, NOT quantum cloning/QMC.\n\n"
            + "3. IMAGINARY-TIME REALM  [%s]\n" % realm_status
            + "   Green prism; [R]; 7 s realm (30 Coherence).\n"
            + "   Opens green gate and activates spectral bridge.\n"
            + "   Inspired by exp[-τ(H-Eref)], NOT time travel/QMC.\n\n"
            + "Checkpoint: %s    Echo puzzle: %s    Realm: %s\n" % [
                    checkpoint_status, echo_cleared, realm_cleared]
            + "G7 Calibration: %s   [P] enter after first chamber is cleared\n" % calibration_status
            + "[I] close    [SHIFT+N] reset local progress")


func _restore_progression() -> void:
    player.tunneling_unlocked = progression.is_unlocked(PROGRESSION.TUNNELING)
    player.echoes_unlocked = progression.is_unlocked(PROGRESSION.ECHOES)
    player.imaginary_time_unlocked = progression.is_unlocked(PROGRESSION.IMAGINARY_TIME)
    checkpoint_activated = progression.checkpoint_reached
    realm_checkpoint_activated = progression.realm_checkpoint_reached
    respawn_position = START_POSITION
    if realm_checkpoint_activated:
        respawn_position = realm_checkpoint.global_position + Vector2(0, 10)
    elif checkpoint_activated:
        respawn_position = checkpoint.global_position + Vector2(0, 10)
    player.global_position = respawn_position
    player.velocity = Vector2.ZERO
    ability_relic.monitoring = not player.tunneling_unlocked
    echo_relic.monitoring = not player.echoes_unlocked
    imaginary_relic.monitoring = not player.imaginary_time_unlocked
    ability_relic.visible = not player.tunneling_unlocked
    echo_relic.visible = not player.echoes_unlocked
    imaginary_relic.visible = not player.imaginary_time_unlocked
    checkpoint.set_marker_active(checkpoint_activated)
    realm_checkpoint.set_marker_active(realm_checkpoint_activated)
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
    ability_fx.play_event("tunnel", ability_relic.global_position)
    ability_relic.set_deferred("monitoring", false)
    ability_relic.visible = false
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
    ability_fx.play_event("echo", echo_relic.global_position)
    echo_relic.set_deferred("monitoring", false)
    echo_relic.visible = false
    _save_progression()
    _refresh_inventory()
    _refresh_echo_science()
    _show_status("QMC ECHOES UNLOCKED! Walk across the violet pad; press Q near the gate.", 7.0)
    queue_redraw()


func _on_imaginary_relic_collected(body: Node2D) -> void:
    if body != player or player.imaginary_time_unlocked:
        return
    if not progression.unlock(PROGRESSION.IMAGINARY_TIME):
        return
    player.imaginary_time_unlocked = true
    ability_fx.play_event("realm", imaginary_relic.global_position)
    imaginary_relic.set_deferred("monitoring", false)
    imaginary_relic.visible = false
    _save_progression()
    _refresh_inventory()
    _refresh_realm_science()
    _show_status("IMAGINARY-TIME REALM UNLOCKED! Press R by the green gate.", 7.0)


func _on_echo_milestone(body: Node2D) -> void:
    if body != player or progression.echo_gate_cleared:
        return
    progression.echo_gate_cleared = true
    _save_progression()
    _refresh_inventory()
    _show_status("ECHO PUZZLE COMPLETE — a green prism lies ahead.", 5.0)


func _on_realm_checkpoint(body: Node2D) -> void:
    if body != player or realm_checkpoint_activated:
        return
    realm_checkpoint_activated = true
    checkpoint_activated = true
    progression.checkpoint_reached = true
    progression.realm_checkpoint_reached = true
    respawn_position = realm_checkpoint.global_position + Vector2(0, 10)
    realm_checkpoint.set_marker_active(true)
    ability_fx.play_event("checkpoint", realm_checkpoint.global_position)
    _save_progression()
    _refresh_inventory()
    _show_status("REALM CHECKPOINT — respawn now starts beyond the Echo gate.", 5.0)


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


func _on_realm_result(result: Dictionary) -> void:
    if result.get("activated", false):
        _set_realm_environment(true)
        _show_status("IMAGINARY-TIME REALM — gate and spectral bridge active (7 s).", 4.0)
    else:
        _show_status(str(result.get("reason", "Realm unavailable.")), 3.0)


func _reset_progression() -> void:
    _close_measurement_lab()
    progression.reset()
    player.tunneling_unlocked = false
    player.echoes_unlocked = false
    player.imaginary_time_unlocked = false
    player.realm_remaining = 0.0
    player.realm_cooldown_remaining = 0.0
    player.echo_history.clear()
    player.echo_cooldown_remaining = 0.0
    if is_instance_valid(echo_ghost):
        echo_ghost.queue_free()
    echo_ghost = null
    gate_open_remaining = 0.0
    _set_echo_gate_open(false)
    _set_realm_environment(false)
    player.coherence = player.MAX_COHERENCE
    player.tunnel_cooldown_remaining = 0.0
    player.velocity = Vector2.ZERO
    player.global_position = START_POSITION
    respawn_position = START_POSITION
    checkpoint_activated = false
    realm_checkpoint_activated = false
    checkpoint.set_marker_active(false)
    realm_checkpoint.set_marker_active(false)
    win = false
    _p_was_pressed = false
    ability_relic.set_deferred("monitoring", true)
    echo_relic.set_deferred("monitoring", true)
    imaginary_relic.set_deferred("monitoring", true)
    ability_relic.visible = true
    echo_relic.visible = true
    imaginary_relic.visible = true
    science_panel.visible = false
    _save_progression()
    _refresh_inventory()
    _show_status("Progress reset. Collect blue, violet and green relics.", 5.0)
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
    progression.realm_gate_cleared = true
    _save_progression()
    _refresh_inventory()
    ability_fx.play_event("finish", goal.global_position)
    _show_status("CHAMBER CLEARED — Press P for the Calibration Chamber.", 9999.0)


func _on_checkpoint(body: Node2D) -> void:
    if body != player or checkpoint_activated:
        return
    checkpoint_activated = true
    respawn_position = checkpoint.global_position + Vector2(0, 10)
    progression.checkpoint_reached = true
    checkpoint.set_marker_active(true)
    _save_progression()
    _refresh_inventory()
    ability_fx.play_event("checkpoint", checkpoint.global_position)
    _show_status("CHECKPOINT ACTIVATED — safe position recorded.", 3.5)
    queue_redraw()


func _on_hazard(body: Node2D) -> void:
    if body != player or win:
        return
    player.global_position = respawn_position
    player.velocity = Vector2.ZERO
    player.coherence = maxf(0.0, player.coherence - 15.0)
    ability_fx.play_event("hazard", player.global_position)
    _show_status("QUANTUM INSTABILITY — returned to checkpoint (-15 Coherence).", 4.0)
