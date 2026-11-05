extends RefCounted
## Exact stationary 1D rectangular-barrier transmission for 0 < E < V0.
## H = -D d²/dx² + V(x), D = hbar²/(2m); a is barrier width.
## This is an analytical teaching model, not VMC/DMC, not a moving 2D body.

const DEFAULT_D := 0.5
const DEFAULT_COST := 18.0


static func evaluate(energy: float, height: float, width: float, d: float = DEFAULT_D) -> Dictionary:
    if is_nan(energy) or is_inf(energy) or is_nan(height) or is_inf(height) \
            or is_nan(width) or is_inf(width) or is_nan(d) or is_inf(d):
        return {"valid": false, "reason": "All model parameters must be finite."}
    if energy <= 0.0 or height <= energy or width <= 0.0 or d <= 0.0:
        return {"valid": false, "reason": "Requires 0 < E < V0, width > 0, and D > 0."}
    # Avoid overflow in sinh for extreme input; those barriers transmit ~0.
    var ka: float = sqrt((height - energy) / d) * width
    var transmission := 0.0
    if ka < 40.0:
        var s: float = sinh(ka)
        var factor: float = height * height * s * s / (4.0 * energy * (height - energy))
        transmission = 1.0 / (1.0 + factor)
    return {"valid": true, "probability": clampf(transmission, 0.0, 1.0),
            "regime": "sub_barrier_stationary_1d", "approximation": "none_for_rectangular_1d_model"}


static func attempt(energy: float, height: float, width: float, d: float,
        coherence: float, cost: float, rng: RandomNumberGenerator) -> Dictionary:
    var model: Dictionary = evaluate(energy, height, width, d)
    if not model["valid"]:
        return {"attempted": false, "success": false, "cost": 0.0,
                "reason": model["reason"], "probability": 0.0}
    if is_nan(coherence) or is_inf(coherence) or is_nan(cost) or is_inf(cost) \
            or cost <= 0.0 or coherence < cost:
        return {"attempted": false, "success": false, "cost": 0.0,
                "reason": "Insufficient Coherence or invalid ability cost.",
                "probability": model["probability"]}
    var roll: float = rng.randf()
    return {"attempted": true, "success": roll < model["probability"],
            "probability": model["probability"], "roll": roll, "cost": cost,
            "reason": ""}
