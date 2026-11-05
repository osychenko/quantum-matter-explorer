extends Node2D
## Small pooled world-space rings and streaks. Visual only: zero collision,
## resource cost, physics modifications or probabilistic gameplay effects.

const LIFETIME := 0.62
const MAX_BURSTS := 24
var bursts: Array[Dictionary] = []


func play_event(kind: String, world_position: Vector2, success: bool = true) -> void:
    var color := Color("#77eaff")
    match kind:
        "tunnel":
            color = Color("#69f5b8") if success else Color("#ff8992")
        "echo":
            color = Color("#cf9dff")
        "realm":
            color = Color("#78ffbf")
        "checkpoint":
            color = Color("#ffdea1")
        "finish":
            color = Color("#e5ffb2")
        "hazard":
            color = Color("#ff8e9c")
        "calibration":
            color = Color("#8dedff")
    bursts.append({"at": to_local(world_position), "age": 0.0,
            "color": color, "success": success})
    if bursts.size() > MAX_BURSTS:
        bursts.pop_front()
    queue_redraw()


func _process(delta: float) -> void:
    for i in range(bursts.size() - 1, -1, -1):
        var burst: Dictionary = bursts[i]
        burst["age"] = float(burst["age"]) + delta
        if float(burst["age"]) >= LIFETIME:
            bursts.remove_at(i)
    if not bursts.is_empty():
        queue_redraw()


func _draw() -> void:
    for burst in bursts:
        var at: Vector2 = burst["at"]
        var t: float = float(burst["age"]) / LIFETIME
        var color: Color = burst["color"]
        color.a = (1.0 - t) * 0.75
        var radius := lerpf(10.0, 74.0, t)
        draw_arc(at, radius, 0.0, TAU, 48, color, 3.0 * (1.0 - 0.6 * t), true)
        for i in range(8):
            var angle := TAU * float(i) / 8.0 + t * 0.4
            var direction := Vector2(cos(angle), sin(angle))
            draw_line(at + direction * (radius + 3.0),
                    at + direction * (radius + 13.0 + t * 15.0),
                    color, 2.0, true)
