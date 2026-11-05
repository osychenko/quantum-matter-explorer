extends RefCounted
## Versioned local progression, independent of rendering and the QMC research engine.
## Only implemented abilities can be unlocked; future entries are display-only.

const SAVE_PATH := "user://qme_platformer_progress_v1.json"
const VERSION := 1
const TUNNELING := "tunneling"
const ECHOES := "echoes"
const IMAGINARY_TIME := "imaginary_time"

var unlocked: Array[String] = []
var checkpoint_reached := false
var chamber_cleared := false
var echo_gate_cleared := false
var realm_checkpoint_reached := false
var realm_gate_cleared := false
var calibration_gate_cleared := false
var calibration_chamber_cleared := false


func is_unlocked(ability_id: String) -> bool:
    return unlocked.has(ability_id)


func unlock(ability_id: String) -> bool:
    if (ability_id != TUNNELING and ability_id != ECHOES \
            and ability_id != IMAGINARY_TIME) or unlocked.has(ability_id):
        return false
    unlocked.append(ability_id)
    return true


func reset() -> void:
    unlocked.clear()
    checkpoint_reached = false
    chamber_cleared = false
    echo_gate_cleared = false
    realm_checkpoint_reached = false
    realm_gate_cleared = false
    calibration_gate_cleared = false
    calibration_chamber_cleared = false


func to_dictionary() -> Dictionary:
    return {
        "version": VERSION,
        "unlocked": unlocked.duplicate(),
        "checkpoint_reached": checkpoint_reached,
        "chamber_cleared": chamber_cleared,
        "echo_gate_cleared": echo_gate_cleared,
        "realm_checkpoint_reached": realm_checkpoint_reached,
        "realm_gate_cleared": realm_gate_cleared,
        "calibration_gate_cleared": calibration_gate_cleared,
        "calibration_chamber_cleared": calibration_chamber_cleared,
    }


func save_to_path(path: String = SAVE_PATH) -> bool:
    var file := FileAccess.open(path, FileAccess.WRITE)
    if file == null:
        push_warning("Could not save Quantum Matter Explorer progression: %s" % path)
        return false
    file.store_string(JSON.stringify(to_dictionary(), "  "))
    file.close()
    return true


func load_from_path(path: String = SAVE_PATH) -> bool:
    if not FileAccess.file_exists(path):
        return false
    var file := FileAccess.open(path, FileAccess.READ)
    if file == null:
        return false

    var parser := JSON.new()
    var parse_error := parser.parse(file.get_as_text())

    if parse_error != OK:
        file.close()
        return false

    var decoded: Variant = parser.data

    file.close()
    if typeof(decoded) != TYPE_DICTIONARY:
        return false
    var data: Dictionary = decoded
    if data.get("version", -1) != VERSION:
        return false
    var saved_unlocks: Variant = data.get("unlocked", [])
    var saved_checkpoint: Variant = data.get("checkpoint_reached", false)
    var saved_cleared: Variant = data.get("chamber_cleared", false)
    # Optional since G3 saves predate the echo-gate puzzle.
    var saved_echo_gate: Variant = data.get("echo_gate_cleared", false)
    var saved_realm_checkpoint: Variant = data.get("realm_checkpoint_reached", false)
    var saved_realm_gate: Variant = data.get("realm_gate_cleared", false)
    var saved_calibration_gate: Variant = data.get("calibration_gate_cleared", false)
    var saved_calibration_chamber: Variant = data.get("calibration_chamber_cleared", false)
    if typeof(saved_unlocks) != TYPE_ARRAY or typeof(saved_checkpoint) != TYPE_BOOL \
            or typeof(saved_cleared) != TYPE_BOOL or typeof(saved_echo_gate) != TYPE_BOOL \
            or typeof(saved_realm_checkpoint) != TYPE_BOOL \
            or typeof(saved_realm_gate) != TYPE_BOOL \
            or typeof(saved_calibration_gate) != TYPE_BOOL \
            or typeof(saved_calibration_chamber) != TYPE_BOOL:
        return false
    # Accept only recognized ability IDs, and only after validating the save.
    reset()
    for ability_id in saved_unlocks:
        if typeof(ability_id) == TYPE_STRING:
            unlock(ability_id)
    checkpoint_reached = saved_checkpoint
    chamber_cleared = saved_cleared
    echo_gate_cleared = saved_echo_gate
    realm_checkpoint_reached = saved_realm_checkpoint
    realm_gate_cleared = saved_realm_gate
    calibration_gate_cleared = saved_calibration_gate
    calibration_chamber_cleared = saved_calibration_chamber
    return true
