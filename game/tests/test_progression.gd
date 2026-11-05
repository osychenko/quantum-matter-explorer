extends SceneTree
## Regression tests for versioned, local save/load without touching the real save.

const STATE = preload("res://scripts/ProgressionState.gd")
const TEST_PATH := "user://qme_progression_g3_test.json"
var failures: Array[String] = []


func _initialize() -> void:
    call_deferred("run_tests")


func check(value: bool, description: String) -> void:
    if not value:
        failures.append(description)
        push_error(description)


func run_tests() -> void:
    var state = STATE.new()
    check(not state.is_unlocked(STATE.TUNNELING), "Tunneling starts locked")
    check(not state.unlock("qmc_echo"), "An unimplemented ability cannot be unlocked")
    check(not state.unlock("imaginary_time"), "Imaginary-time ability is not implemented")
    check(state.unlock(STATE.TUNNELING), "Pickup unlocks tunneling")
    check(not state.unlock(STATE.TUNNELING), "Duplicate pickup has no effect")
    check(state.unlocked.size() == 1, "Ability IDs remain unique")
    state.checkpoint_reached = true
    state.chamber_cleared = true
    check(state.save_to_path(TEST_PATH), "Progress writes to an isolated test file")

    var loaded = STATE.new()
    check(loaded.load_from_path(TEST_PATH), "Progress restores from disk")
    check(loaded.is_unlocked(STATE.TUNNELING), "Unlock survives restart")
    check(loaded.checkpoint_reached and loaded.chamber_cleared,
            "Checkpoint and chamber completion survive restart")
    loaded.reset()
    check(not loaded.is_unlocked(STATE.TUNNELING) and not loaded.checkpoint_reached \
            and not loaded.chamber_cleared, "Reset clears all progress")
    check(loaded.save_to_path(TEST_PATH), "Reset state is persisted")

    var empty_again = STATE.new()
    check(empty_again.load_from_path(TEST_PATH), "Reset save is readable")
    check(not empty_again.is_unlocked(STATE.TUNNELING), "Reset persists locked state")

    # A bad save must be rejected without modifying the current state.
    var file := FileAccess.open(TEST_PATH, FileAccess.WRITE)
    check(file != null, "Can write invalid test fixture")
    if file != null:
        file.store_string("{not-json")
        file.close()
        state.checkpoint_reached = true
        check(not state.load_from_path(TEST_PATH), "Invalid JSON is rejected")
        check(state.is_unlocked(STATE.TUNNELING) and state.checkpoint_reached,
                "Invalid JSON does not erase current progress")

    file = FileAccess.open(TEST_PATH, FileAccess.WRITE)
    if file != null:
        file.store_string('{"version": 999, "unlocked": ["tunneling"]}')
        file.close()
        check(not state.load_from_path(TEST_PATH), "Unsupported save version is rejected")
    DirAccess.remove_absolute(ProjectSettings.globalize_path(TEST_PATH))

    if failures.is_empty():
        print("PASS test_progression.gd — unlock, save, restore, reset, corruption")
    else:
        printerr("FAIL test_progression.gd: %d failure(s)" % failures.size())
    quit(0 if failures.is_empty() else 1)
