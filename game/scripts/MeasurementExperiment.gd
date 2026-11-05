extends RefCounted
## Independent repeated measurements of the exact rectangular-barrier 1D model.
## Uses Bernoulli sampling of the predicted transmission probability, NOT QMC.

const MODEL = preload("res://scripts/TunnelingModel.gd")
const MAX_TRIALS := 100000


static func run_trials(probability: float, count: int, seed_value: int = 2026) -> Dictionary:
    if is_nan(probability) or is_inf(probability) or probability < 0.0 or probability > 1.0:
        return {"valid": false, "reason": "Probability must be finite and between 0 and 1."}
    if count < 1 or count > MAX_TRIALS:
        return {"valid": false, "reason": "Number of trials must be between 1 and 100000."}
    var rng := RandomNumberGenerator.new()
    rng.seed = seed_value
    var transmitted := 0
    for i in range(count):
        if rng.randf() < probability:
            transmitted += 1
    var observed := float(transmitted) / float(count)
    # Standard deviation of the sample proportion, under the supplied model p.
    var standard_error := sqrt(probability * (1.0 - probability) / float(count))
    return {"valid": true, "trials": count, "transmitted": transmitted,
            "reflected": count - transmitted, "predicted_probability": probability,
            "observed_frequency": observed, "standard_error": standard_error,
            "seed": seed_value}


static func run_barrier(energy: float, height: float, width: float, d: float,
        count: int, seed_value: int = 2026) -> Dictionary:
    var model: Dictionary = MODEL.evaluate(energy, height, width, d)
    if not model.get("valid", false):
        return {"valid": false, "reason": str(model.get("reason", "Invalid barrier."))}
    var result: Dictionary = run_trials(float(model["probability"]), count, seed_value)
    if result.get("valid", false):
        result["energy"] = energy
        result["height"] = height
        result["width"] = width
        result["d"] = d
    return result
