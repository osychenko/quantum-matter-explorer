extends SceneTree
## Pure model regression tests; the process exits nonzero on failure.

const MODEL = preload("res://scripts/TunnelingModel.gd")
var failures: Array[String] = []


func _initialize() -> void:
    call_deferred("run_tests")


func check(value: bool, description: String) -> void:
    if not value:
        failures.append(description)
        push_error(description)


func near(a: float, b: float, tolerance: float = 0.000001) -> bool:
    return absf(a - b) <= tolerance


func run_tests() -> void:
    var baseline: Dictionary = MODEL.evaluate(0.8, 1.2, 0.8, 0.5)
    check(baseline["valid"], "Default rectangular barrier should be supported")
    var p: float = baseline.get("probability", -1.0)
    check(p > 0.0 and p < 1.0, "Sub-barrier transmission lies strictly between zero and one")
    var thin: Dictionary = MODEL.evaluate(0.8, 1.2, 0.2, 0.5)
    var thick: Dictionary = MODEL.evaluate(0.8, 1.2, 1.7, 0.5)
    check(thin["probability"] > p and p > thick["probability"], "Thickness reduces transmission")
    var high: Dictionary = MODEL.evaluate(0.8, 2.0, 0.8, 0.5)
    check(high["probability"] < p, "Increasing V0 reduces sub-barrier transmission")
    var tiny: Dictionary = MODEL.evaluate(0.8, 1.2, 0.00001, 0.5)
    check(tiny["probability"] > 0.999, "Zero-width limit tends to perfect transmission")
    var huge: Dictionary = MODEL.evaluate(0.8, 1.2, 100.0, 0.5)
    check(huge["valid"] and near(huge["probability"], 0.0), "Very thick barrier transmits approximately zero")
    for invalid in [
        [0.0, 1.2, 0.8, 0.5], [1.2, 1.2, 0.8, 0.5],
        [1.4, 1.2, 0.8, 0.5], [0.8, 1.2, -1.0, 0.5],
        [0.8, 1.2, 0.8, 0.0], [NAN, 1.2, 0.8, 0.5],
        [0.8, INF, 0.8, 0.5]
    ]:
        check(not MODEL.evaluate(invalid[0], invalid[1], invalid[2], invalid[3])["valid"],
                "Reject unsupported / invalid parameters: %s" % str(invalid))
    var rng1 := RandomNumberGenerator.new()
    var rng2 := RandomNumberGenerator.new()
    rng1.seed = 12345678
    rng2.seed = 12345678
    for i in range(24):
        var first: Dictionary = MODEL.attempt(0.8, 1.2, 0.8, 0.5, 100.0, 18.0, rng1)
        var second: Dictionary = MODEL.attempt(0.8, 1.2, 0.8, 0.5, 100.0, 18.0, rng2)
        check(first["attempted"] and first["cost"] == 18.0, "Valid attempt charges configured amount")
        check(first["roll"] == second["roll"] and first["success"] == second["success"],
                "Equal seeds reproduce outcomes (draw %d)" % i)
    var blocked: Dictionary = MODEL.attempt(0.8, 1.2, 0.8, 0.5, 5.0, 18.0, rng1)
    check(not blocked["attempted"] and blocked["cost"] == 0.0,
            "Insufficient Coherence prevents charging and sampling")
    var unsupported: Dictionary = MODEL.attempt(2.0, 1.2, 0.8, 0.5, 100, 18, rng1)
    check(not unsupported["attempted"] and unsupported["cost"] == 0.0,
            "Unsupported energy regime never spends Coherence")
    if failures.is_empty():
        print("PASS test_tunneling.gd — analytical and reproducibility checks")
    else:
        printerr("FAIL test_tunneling.gd: %d failure(s)" % failures.size())
    quit(0 if failures.is_empty() else 1)
