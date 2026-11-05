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
    check(not state.unlock("qmc_echo"), "Unknown legacy ID cannot be unlocked")
    check(state.unlock(STATE.ECHOES), "QMC Echoes unlock is now supported")
    check(not state.unlock(STATE.ECHOES), "Echoes cannot be unlocked twice")
    check(state.unlock(STATE.IMAGINARY_TIME), "Imaginary-time realm unlock is supported")
    check(not state.unlock(STATE.IMAGINARY_TIME), "Realm unlock is idempotent")
    check(state.unlock(STATE.TUNNELING), "Pickup unlocks tunneling")
    check(not state.unlock(STATE.TUNNELING), "Duplicate pickup has no effect")
    check(state.unlocked.size() == 3, "Ability IDs remain unique")
    state.checkpoint_reached = true
    state.chamber_cleared = true
    state.echo_gate_cleared = true
    state.realm_checkpoint_reached = true
    state.realm_gate_cleared = true
    check(state.save_to_path(TEST_PATH), "Progress writes to an isolated test file")

    var loaded = STATE.new()
    check(loaded.load_from_path(TEST_PATH), "Progress restores from disk")
    check(loaded.is_unlocked(STATE.TUNNELING), "Unlock survives restart")
    check(loaded.is_unlocked(STATE.ECHOES), "Echoes unlock survives restart")
    check(loaded.is_unlocked(STATE.IMAGINARY_TIME), "Realm unlock survives restart")
    check(loaded.checkpoint_reached and loaded.chamber_cleared,
            "Checkpoint and chamber completion survive restart")
    check(loaded.echo_gate_cleared, "G4 gate completion survives restart")
    check(loaded.realm_checkpoint_reached and loaded.realm_gate_cleared,
            "G5 checkpoint and realm completion survive restart")
    loaded.reset()
    check(not loaded.is_unlocked(STATE.TUNNELING) and not loaded.checkpoint_reached \
            and not loaded.chamber_cleared and not loaded.echo_gate_cleared \
            and not loaded.realm_checkpoint_reached and not loaded.realm_gate_cleared,
            "Reset clears all progress")
    check(loaded.save_to_path(TEST_PATH), "Reset state is persisted")

    var empty_again = STATE.new()
    check(empty_again.load_from_path(TEST_PATH), "Reset save is readable")
    check(not empty_again.is_unlocked(STATE.TUNNELING)
            and not empty_again.is_unlocked(STATE.ECHOES) \
            and not empty_again.is_unlocked(STATE.IMAGINARY_TIME),
            "Reset persists all three locked states")
    # A G3 save has no echo_gate_cleared flag; it must load as not yet solved.
    var file := FileAccess.open(TEST_PATH, FileAccess.WRITE)
    if file != null:
        file.store_string('{"version": 1, "unlocked": ["tunneling"], "checkpoint_reached": true, "chamber_cleared": true}')
        file.close()
        check(empty_again.load_from_path(TEST_PATH), "Legacy G3 save remains readable")
        check(not empty_again.echo_gate_cleared, "Legacy G3 completion does not solve G4 gate")
        check(not empty_again.realm_checkpoint_reached and not empty_again.realm_gate_cleared,
                "Legacy G3 save defaults G5 flags safely")

    file = FileAccess.open(TEST_PATH, FileAccess.WRITE)
    if file != null:
        file.store_string('{"version": 1, "unlocked": ["tunneling", "echoes"], "checkpoint_reached": true, "chamber_cleared": true, "echo_gate_cleared": true}')
        file.close()
        check(empty_again.load_from_path(TEST_PATH), "Legacy G4 save remains readable")
        check(not empty_again.is_unlocked(STATE.IMAGINARY_TIME),
                "Legacy G4 save does not grant realm ability")
        check(not empty_again.realm_checkpoint_reached and not empty_again.realm_gate_cleared,
                "Legacy G4 save defaults G5 flags safely")

    # A bad save must be rejected without modifying the current state.
    file = FileAccess.open(TEST_PATH, FileAccess.WRITE)
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
