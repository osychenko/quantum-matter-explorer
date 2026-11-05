extends RefCounted
## Versioned local progression, independent of rendering and the QMC research engine.
## Only implemented abilities can be unlocked; future entries are display-only.

const SAVE_PATH := "user://qme_platformer_progress_v1.json"
const VERSION := 1
const TUNNELING := "tunneling"

var unlocked: Array[String] = []
var checkpoint_reached := false
var chamber_cleared := false


func is_unlocked(ability_id: String) -> bool:
    return unlocked.has(ability_id)


func unlock(ability_id: String) -> bool:
    if ability_id != TUNNELING or unlocked.has(ability_id):
        return false
    unlocked.append(ability_id)
    return true


func reset() -> void:
    unlocked.clear()
    checkpoint_reached = false
    chamber_cleared = false


func to_dictionary() -> Dictionary:
    return {
        "version": VERSION,
        "unlocked": unlocked.duplicate(),
        "checkpoint_reached": checkpoint_reached,
        "chamber_cleared": chamber_cleared,
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
    if typeof(saved_unlocks) != TYPE_ARRAY or typeof(saved_checkpoint) != TYPE_BOOL \
            or typeof(saved_cleared) != TYPE_BOOL:
        return false
    # Accept only recognized ability IDs, and only after validating the save.
    reset()
    for ability_id in saved_unlocks:
        if typeof(ability_id) == TYPE_STRING:
            unlock(ability_id)
    checkpoint_reached = saved_checkpoint
    chamber_cleared = saved_cleared
    return true
