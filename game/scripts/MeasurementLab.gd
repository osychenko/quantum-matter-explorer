extends PanelContainer
## Self-contained, mouse-operable measurement lab. Does not mutate real barriers.
## Built at runtime so no separate scene/art dependency is required.

signal close_requested

const EXPERIMENT = preload("res://scripts/MeasurementExperiment.gd")
const MODEL = preload("res://scripts/TunnelingModel.gd")

var barrier_name := ""
var barrier_height := 1.2
var kinetic_coefficient := 0.5
var reference_energy := 0.8
var reference_width := 0.8
var run_number := 0
var total_trials := 0
var total_transmitted := 0
var last_result: Dictionary = {}

var energy_spin: SpinBox
var width_spin: SpinBox
var seed_spin: SpinBox
var barrier_label: Label
var predicted_label: Label
var sample_label: Label
var total_label: Label
var transmitted_label: Label
var reflected_label: Label
var transmitted_bar: ProgressBar
var reflected_bar: ProgressBar


func _ready() -> void:
    name = "MeasurementLab"
    position = Vector2(265, 46)
    custom_minimum_size = Vector2(750, 608)
    mouse_filter = Control.MOUSE_FILTER_STOP
    var style := StyleBoxFlat.new()
    style.bg_color = Color(0.037, 0.076, 0.13, 0.98)
    style.border_color = Color("#69cde5")
    style.set_border_width_all(2)
    style.set_corner_radius_all(10)
    style.set_content_margin_all(18)
    add_theme_stylebox_override("panel", style)

    var vertical := VBoxContainer.new()
    vertical.add_theme_constant_override("separation", 9)
    add_child(vertical)

    var header := HBoxContainer.new()
    vertical.add_child(header)
    _new_label(header, "QUANTUM MEASUREMENT LAB", 23, Color("#d8f7ff"))
    var space := Control.new()
    space.size_flags_horizontal = Control.SIZE_EXPAND_FILL
    header.add_child(space)
    var close_button := Button.new()
    close_button.text = "Close [M / Esc]"
    close_button.pressed.connect(_request_close)
    header.add_child(close_button)

    _new_label(vertical,
            "Stationary 1D scattering: each trial is transmitted or reflected.\n"
            + "Sampling is a teaching experiment, NOT quantum Monte Carlo.",
            15, Color("#c2dce5"))
    barrier_label = _new_label(vertical, "", 16, Color("#f0db98"))

    var controls := HBoxContainer.new()
    controls.add_theme_constant_override("separation", 15)
    vertical.add_child(controls)
    energy_spin = _make_spin(controls, "Energy E", 0.05, 2.0, 0.05, 0.8)
    width_spin = _make_spin(controls, "Width a", 0.1, 3.0, 0.05, 0.8)
    seed_spin = _make_spin(controls, "Seed", 0.0, 999999999.0, 1.0, 2026.0)
    energy_spin.value_changed.connect(_on_parameters_changed)
    width_spin.value_changed.connect(_on_parameters_changed)
    seed_spin.value_changed.connect(_on_parameters_changed)

    predicted_label = _new_label(vertical, "", 16, Color("#92e9c5"))
    var actions := HBoxContainer.new()
    actions.add_theme_constant_override("separation", 10)
    vertical.add_child(actions)
    for count in [1, 10, 100, 1000]:
        var button := Button.new()
        button.text = "Run %d" % count
        button.pressed.connect(_on_run_pressed.bind(count))
        actions.add_child(button)
    var reset_button := Button.new()
    reset_button.text = "Clear results"
    reset_button.pressed.connect(clear_results)
    actions.add_child(reset_button)
    var defaults_button := Button.new()
    defaults_button.text = "Barrier defaults"
    defaults_button.pressed.connect(_restore_defaults)
    actions.add_child(defaults_button)

    _new_label(vertical, "Observed outcomes (latest batch)", 16, Color("#d8f7ff"))
    var row_t := HBoxContainer.new()
    vertical.add_child(row_t)
    transmitted_label = _new_label(row_t, "Transmitted", 15, Color("#87e5bd"))
    transmitted_label.custom_minimum_size.x = 225.0
    transmitted_bar = _make_bar(row_t, Color("#66d9ae"))
    var row_r := HBoxContainer.new()
    vertical.add_child(row_r)
    reflected_label = _new_label(row_r, "Reflected", 15, Color("#f3adba"))
    reflected_label.custom_minimum_size.x = 225.0
    reflected_bar = _make_bar(row_r, Color("#f18ca3"))
    sample_label = _new_label(vertical, "No trials yet.", 16, Color("#e1e9fa"))
    total_label = _new_label(vertical, "", 16, Color("#bfd6ed"))
    _new_label(vertical,
            "Theoretical sampling error: SE = sqrt(T (1 - T) / N).\n"
            + "In a finite batch, the observed fraction generally differs from T.\n"
            + "Experiment settings affect only this lab, not gameplay physics or Coherence.",
            14, Color("#aabdcf"))
    _refresh_prediction()
    _refresh_results()


func _new_label(parent: Node, label_text: String, font_size: int, color: Color) -> Label:
    var label := Label.new()
    label.text = label_text
    label.autowrap_mode = TextServer.AUTOWRAP_WORD_SMART
    label.add_theme_font_size_override("font_size", font_size)
    label.add_theme_color_override("font_color", color)
    parent.add_child(label)
    return label


func _make_spin(parent: Node, label_text: String, minimum: float,
        maximum: float, step_size: float, start: float) -> SpinBox:
    var group := VBoxContainer.new()
    parent.add_child(group)
    _new_label(group, label_text, 15, Color("#d8f7ff"))
    var spin := SpinBox.new()
    spin.min_value = minimum
    spin.max_value = maximum
    spin.step = step_size
    spin.value = start
    spin.custom_minimum_size.x = 145.0
    group.add_child(spin)
    return spin


func _make_bar(parent: Node, color: Color) -> ProgressBar:
    var bar := ProgressBar.new()
    bar.min_value = 0.0
    bar.max_value = 100.0
    bar.show_percentage = false
    bar.custom_minimum_size = Vector2(420, 21)
    var fill := StyleBoxFlat.new()
    fill.bg_color = color
    fill.set_corner_radius_all(4)
    bar.add_theme_stylebox_override("fill", fill)
    var background := StyleBoxFlat.new()
    background.bg_color = Color("#233a52")
    background.set_corner_radius_all(4)
    bar.add_theme_stylebox_override("background", background)
    parent.add_child(bar)
    return bar


func configure_barrier(source: Node, display_name: String) -> void:
    barrier_name = display_name
    barrier_height = float(source.get("barrier_height"))
    kinetic_coefficient = float(source.get("kinetic_coefficient"))
    reference_energy = float(source.get("incident_energy"))
    reference_width = float(source.get("model_width"))
    energy_spin.max_value = maxf(0.1, barrier_height - 0.05)
    _restore_defaults()


func _restore_defaults() -> void:
    energy_spin.value = clampf(reference_energy, energy_spin.min_value, energy_spin.max_value)
    width_spin.value = clampf(reference_width, width_spin.min_value, width_spin.max_value)
    clear_results()


func _on_parameters_changed(_value: float) -> void:
    clear_results()


func _refresh_prediction() -> void:
    if predicted_label == null:
        return
    barrier_label.text = "%s  |  V0 = %.2f  |  D = %.2f" % [barrier_name,
            barrier_height, kinetic_coefficient]
    var result: Dictionary = MODEL.evaluate(float(energy_spin.value),
            barrier_height, float(width_spin.value), kinetic_coefficient)
    if result.get("valid", false):
        predicted_label.text = "Analytical prediction: T = %.3f%%    R = %.3f%%" % [
                100.0 * float(result["probability"]),
                100.0 * (1.0 - float(result["probability"]))]
    else:
        predicted_label.text = "Invalid parameters: " + str(result.get("reason", ""))


func clear_results() -> void:
    run_number = 0
    total_trials = 0
    total_transmitted = 0
    last_result.clear()
    _refresh_prediction()
    _refresh_results()


func _on_run_pressed(count: int) -> void:
    run_batch(count)


func run_batch(count: int) -> Dictionary:
    var result: Dictionary = EXPERIMENT.run_barrier(float(energy_spin.value),
            barrier_height, float(width_spin.value), kinetic_coefficient,
            count, int(seed_spin.value) + run_number)
    last_result = result.duplicate()
    if result.get("valid", false):
        run_number += 1
        total_trials += int(result["trials"])
        total_transmitted += int(result["transmitted"])
    _refresh_results()
    return result


func _refresh_results() -> void:
    if sample_label == null:
        return
    if not last_result.get("valid", false):
        sample_label.text = "No trials yet." if last_result.is_empty() else \
                "Cannot run experiment: " + str(last_result.get("reason", ""))
        transmitted_label.text = "Transmitted: —"
        reflected_label.text = "Reflected: —"
        transmitted_bar.value = 0.0
        reflected_bar.value = 0.0
    else:
        var trials := int(last_result["trials"])
        var trans := int(last_result["transmitted"])
        var reflected := int(last_result["reflected"])
        var frequency := float(last_result["observed_frequency"])
        transmitted_bar.value = frequency * 100.0
        reflected_bar.value = 100.0 - frequency * 100.0
        transmitted_label.text = "Transmitted: %d / %d" % [trans, trials]
        reflected_label.text = "Reflected: %d / %d" % [reflected, trials]
        sample_label.text = ("Latest: %.2f%% transmitted vs T = %.2f%%; "
                + "SE = %.2f percentage points; seed %d") % [frequency * 100.0,
                float(last_result["predicted_probability"]) * 100.0,
                float(last_result["standard_error"]) * 100.0, int(last_result["seed"])]
    var total_fraction := float(total_transmitted) / float(total_trials) \
            if total_trials > 0 else 0.0
    total_label.text = "Cumulative: %d transmitted of %d (%.2f%%)  |  %d batches" % [
            total_transmitted, total_trials, total_fraction * 100.0, run_number]


func _request_close() -> void:
    close_requested.emit()
