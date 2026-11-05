extends SceneTree
## G6 unit tests: physical consistency, reproducible Bernoulli batches, validation.

const EXPERIMENT = preload("res://scripts/MeasurementExperiment.gd")
const MODEL = preload("res://scripts/TunnelingModel.gd")
var failures: Array[String] = []


func _initialize() -> void:
    call_deferred("run_tests")


func check(condition: bool, description: String) -> void:
    if not condition:
        failures.append(description)
        push_error(description)


func run_tests() -> void:
    var zero: Dictionary = EXPERIMENT.run_trials(0.0, 100, 42)
    var one: Dictionary = EXPERIMENT.run_trials(1.0, 100, 42)
    check(zero.get("valid", false) and zero["transmitted"] == 0,
            "Zero probability reflects every trial")
    check(one.get("valid", false) and one["transmitted"] == 100,
            "Unit probability transmits every trial")
    var a: Dictionary = EXPERIMENT.run_trials(0.37, 1000, 2026)
    var b: Dictionary = EXPERIMENT.run_trials(0.37, 1000, 2026)
    check(a == b, "Same seed and model produce an identical batch")
    check(a["transmitted"] + a["reflected"] == 1000,
            "Every trial is classified as transmitted or reflected")
    check(absf(float(a["observed_frequency"]) - 0.37) < 0.08,
            "Large batch agrees with prediction within broad tolerance")
    check(absf(float(a["standard_error"]) - sqrt(0.37 * 0.63 / 1000.0)) < 0.000001,
            "Sampling standard error uses Bernoulli variance")
    check(not EXPERIMENT.run_trials(-0.1, 100)["valid"], "Reject p below zero")
    check(not EXPERIMENT.run_trials(1.1, 100)["valid"], "Reject p above one")
    check(not EXPERIMENT.run_trials(INF, 100)["valid"], "Reject nonfinite p")
    check(not EXPERIMENT.run_trials(0.5, 0)["valid"], "Reject zero trials")
    check(not EXPERIMENT.run_trials(0.5, 100001)["valid"], "Limit trial count")
    var analytical: Dictionary = MODEL.evaluate(0.8, 1.2, 0.8, 0.5)
    var scattering: Dictionary = EXPERIMENT.run_barrier(0.8, 1.2, 0.8, 0.5, 1000, 7)
    check(scattering.get("valid", false), "Valid rectangular-barrier experiment")
    check(is_equal_approx(float(scattering["predicted_probability"]),
            float(analytical["probability"])), "Lab uses same equation as gameplay")
    var thicker: Dictionary = MODEL.evaluate(0.8, 1.2, 1.4, 0.5)
    check(thicker["probability"] < analytical["probability"],
            "Larger width decreases under-barrier transmission")
    check(not EXPERIMENT.run_barrier(1.5, 1.2, 0.8, 0.5, 10)["valid"],
            "Reject E greater than barrier in this sub-barrier model")
    if failures.is_empty():
        print("PASS test_measurement.gd — reproducibility, probabilities and analytical model")
    else:
        printerr("FAIL test_measurement.gd: %d failures" % failures.size())
    quit(0 if failures.is_empty() else 1)
