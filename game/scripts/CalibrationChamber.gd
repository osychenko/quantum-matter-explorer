extends Node2D
## G7: a second room where statistical measurement becomes a reproducible puzzle.
## This is Bernoulli sampling of analytical tunneling T, not quantum Monte Carlo.

const PROGRESSION = preload("res://scripts/ProgressionState.gd")
const LAB = preload("res://scripts/MeasurementLab.gd")
const WORLD_WIDTH := 1640.0
const START_POSITION := Vector2(160, 595)
const CHECKPOINT_POSITION := Vector2(975, 595)
const TARGET_MIN := 0.40
const TARGET_MAX := 0.50
const MAX_STANDARD_ERROR := 0.02
const TERMINAL_RANGE := 190.0

@export var persistence_enabled := true
var progression = PROGRESSION.new()

@onready var player: Variant = $Player
@onready var console: Variant = $CalibrationConsole
@onready var gate: Variant = $CalibrationGate
@onready var checkpoint: Variant = $Checkpoint
@onready var hazard: Area2D = $Hazard
@onready var goal: Area2D = $Goal

var lab: Variant = null
var status_label: Label
var hints_label: Label
var objective_label: Label
var small_batch_seen := false
var small_standard_error := INF
var small_energy := 0.0
var small_width := 0.0
var gate_open := false
var checkpoint_active := false
var room_completed := false
var respawn_position := START_POSITION
var _m_pressed := false
var _b_pressed := false


func _ready() -> void:
    # The calibration room is available only through the G6 completion link.
    # In headless tests persistence is disabled before add_child().
    if persistence_enabled:
        progression.load_from_path()
    player.tunneling_unlocked = progression.is_unlocked(PROGRESSION.TUNNELING)
    player.echoes_unlocked = progression.is_unlocked(PROGRESSION.ECHOES)
    player.imaginary_time_unlocked = progression.is_unlocked(PROGRESSION.IMAGINARY_TIME)
    player.global_position = START_POSITION
    player.velocity = Vector2.ZERO
    checkpoint.body_entered.connect(_on_checkpoint)
    hazard.body_entered.connect(_on_hazard)
    goal.body_entered.connect(_on_goal)
    _make_hud()
    _set_gate_open(progression.calibration_gate_cleared)
    _refresh_lab_challenge()
    room_completed = progression.calibration_chamber_cleared
    _refresh_objective()
    _show_status("Measurement lab: press M near the blue console. B returns to Barrier Ruins.")


func _draw() -> void:
    draw_rect(Rect2(0, 0, WORLD_WIDTH, 720), Color("#101724"))
    for i in range(19):
        var x := float(i) * 89.0 + 22.0
        draw_circle(Vector2(x, 95.0 + 24.0 * sin(float(i) * 1.2)),
                1.7, Color(0.67, 0.86, 1.0, 0.5))
    draw_rect(Rect2(0, 535, WORLD_WIDTH, 105), Color("#142639"))


func _make_hud() -> void:
    var layer := CanvasLayer.new()
    layer.name = "HUD"
    add_child(layer)
    var overlay := Control.new()
    overlay.name = "Overlay"
    overlay.mouse_filter = Control.MOUSE_FILTER_IGNORE
    overlay.set_anchors_preset(Control.PRESET_FULL_RECT)
    layer.add_child(overlay)
    var title := Label.new()
    title.text = "G7 / CALIBRATION CHAMBER"
    title.position = Vector2(22, 18)
    title.add_theme_font_size_override("font_size", 23)
    title.add_theme_color_override("font_color", Color("#c3e6fa"))
    overlay.add_child(title)
    objective_label = Label.new()
    objective_label.position = Vector2(22, 55)
    objective_label.custom_minimum_size = Vector2(1000, 66)
    objective_label.add_theme_font_size_override("font_size", 17)
    objective_label.add_theme_color_override("font_color", Color("#d8f0ef"))
    overlay.add_child(objective_label)
    status_label = Label.new()
    status_label.position = Vector2(22, 143)
    status_label.custom_minimum_size = Vector2(1120, 50)
    status_label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
    status_label.add_theme_font_size_override("font_size", 17)
    status_label.add_theme_color_override("font_color", Color("#f2dca6"))
    overlay.add_child(status_label)
    hints_label = Label.new()
    hints_label.position = Vector2(22, 674)
    hints_label.text = "A/D MOVE · SPACE JUMP · M LAB AT CONSOLE · B BACK TO BARRIER RUINS"
    hints_label.add_theme_color_override("font_color", Color("#bee8ed"))
    overlay.add_child(hints_label)
    lab = LAB.new()
    overlay.add_child(lab)
    lab.visible = false
    lab.close_requested.connect(_close_lab)
    lab.settings_changed.connect(_on_lab_settings_changed)
    lab.batch_completed.connect(_on_batch_completed)
    _refresh_lab_challenge()


func _process(_delta: float) -> void:
    if player.global_position.y > 900.0:
        _respawn("Returned to checkpoint after falling.")
    var m_pressed := Input.is_physical_key_pressed(KEY_M)
    if m_pressed and not _m_pressed:
        if lab.visible:
            _close_lab()
        else:
            _open_lab()
    _m_pressed = m_pressed
    if lab.visible:
        if Input.is_action_just_pressed("ui_cancel"):
            _close_lab()
        return
    var b_pressed := Input.is_physical_key_pressed(KEY_B)
    if b_pressed and not _b_pressed:
        get_tree().change_scene_to_file("res://scenes/DemoLevel.tscn")
        return
    _b_pressed = b_pressed


func _open_lab() -> bool:
    if lab.visible:
        return true
    if absf(player.global_position.x - console.global_position.x) > TERMINAL_RANGE:
        _show_status("Move close to the blue calibration console, then press M.")
        return false
    lab.configure_barrier(console, "CALIBRATION CONSOLE")
    player.velocity = Vector2.ZERO
    player.set_physics_process(false)
    lab.visible = true
    _refresh_objective()
    return true


func _close_lab() -> void:
    if not lab.visible:
        return
    lab.visible = false
    player.set_physics_process(true)


func _on_lab_settings_changed() -> void:
    if gate_open:
        return
    small_batch_seen = false
    small_standard_error = INF
    _refresh_objective()
    _refresh_lab_challenge()


func _on_batch_completed(result: Dictionary) -> void:
    if gate_open or not result.get("valid", false):
        return
    var probability := float(result["predicted_probability"])
    var count := int(result["trials"])
    var error := float(result["standard_error"])
    var energy := float(result["energy"])
    var width := float(result["width"])
    if probability < TARGET_MIN or probability > TARGET_MAX:
        _show_status("Adjust E or width until predicted T is between 40% and 50%.")
        _refresh_objective()
        return
    if count <= 10:
        small_batch_seen = true
        small_standard_error = error
        small_energy = energy
        small_width = width
        _show_status("Small sample recorded. Now run 1,000 trials without changing parameters.")
    elif count >= 1000 and small_batch_seen:
        # No random pass/fail: the gate checks a well-posed experiment and
        # theoretical uncertainty reduction, NOT a lucky observed frequency.
        if is_equal_approx(energy, small_energy) and is_equal_approx(width, small_width) \
                and error <= MAX_STANDARD_ERROR and error < small_standard_error:
            _set_gate_open(true)
            progression.calibration_gate_cleared = true
            _save_progression()
            _show_status("CALIBRATED! More trials reduced the expected sampling error; gate unlocked.")
        else:
            _show_status("Match the small batch settings and reduce theoretical SE to 2% or less.")
    elif not small_batch_seen:
        _show_status("First collect a small sample with Run 10, then try Run 1000.")
    _refresh_objective()
    _refresh_lab_challenge()


func _set_gate_open(opened: bool) -> void:
    gate_open = opened
    gate.set_solid(not opened)
    console.set_completed(opened)


func _refresh_objective() -> void:
    if objective_label == null:
        return
    if room_completed:
        objective_label.text = "LAB COMPLETE · Evidence collected · Press B to revisit Barrier Ruins."
    elif gate_open:
        objective_label.text = "CALIBRATION GATE OPEN · Cross the jump section to reach the green exit."
    elif small_batch_seen:
        objective_label.text = "STEP 2/2: Keep E and width unchanged; Run 1000 to reduce theoretical SE."
    else:
        objective_label.text = ("STEP 1/2: Set predicted T to 40–50% (try E=0.8, a=1.0), "
                + "then Run 10.\nSTEP 2: Run 1000; unlocking does not depend on lucky outcomes.")


func _refresh_lab_challenge() -> void:
    if lab == null:
        return
    if gate_open:
        lab.set_challenge_text("GATE UNLOCKED: close the lab and proceed to the green exit.\n"
                + "Sampling uses an analytical T, NOT quantum Monte Carlo.")
    elif small_batch_seen:
        lab.set_challenge_text("STEP 2: Keep energy and width fixed; now click Run 1000.\n"
                + "Theoretical sampling error must fall to 2% or less.")
    else:
        lab.set_challenge_text("PUZZLE: Set theoretical T between 40% and 50%, then click Run 10.\n"
                + "Tip: E = 0.8, width = 1.0. This is NOT quantum Monte Carlo.")


func _show_status(message: String) -> void:
    if status_label != null:
        status_label.text = message


func _on_checkpoint(body: Node2D) -> void:
    if body != player or checkpoint_active or not gate_open:
        return
    checkpoint_active = true
    respawn_position = checkpoint.global_position + Vector2(0, 10)
    checkpoint.set_marker_active(true)
    _show_status("CHECKPOINT ACTIVATED — the hazard won't send you back to the lab.")


func _respawn(message: String) -> void:
    player.velocity = Vector2.ZERO
    player.global_position = respawn_position
    player.coherence = player.MAX_COHERENCE
    _show_status(message)


func _on_hazard(body: Node2D) -> void:
    if body != player or room_completed:
        return
    _respawn("Instability field! Returned to checkpoint; Coherence restored.")


func _on_goal(body: Node2D) -> void:
    if body != player or room_completed or not gate_open:
        return
    room_completed = true
    progression.calibration_chamber_cleared = true
    _save_progression()
    _refresh_objective()
    _show_status("G7 COMPLETE — measurement statistics now drive puzzle progression!")


func _save_progression() -> void:
    if persistence_enabled and not progression.save_to_path():
        push_warning("Could not save calibration progression.")
