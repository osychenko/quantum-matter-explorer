extends RefCounted
## G8 screen navigation. Fade lives on the SceneTree root so changing scenes
## cannot prematurely free the transition overlay.

const MENU := "res://scenes/GameMenu.tscn"
const RUINS := "res://scenes/DemoLevel.tscn"
const CALIBRATION := "res://scenes/CalibrationChamber.tscn"
const FADE_NODE := "G8FadeLayer"


static func calibration_unlocked(state: Variant) -> bool:
    return state != null and bool(state.chamber_cleared)


static func navigate(tree: SceneTree, destination: String) -> bool:
    if destination not in [MENU, RUINS, CALIBRATION] or not ResourceLoader.exists(destination):
        return false
    if tree.root.has_node(FADE_NODE):
        return false

    var layer := CanvasLayer.new()
    layer.name = FADE_NODE
    layer.layer = 100
    tree.root.add_child(layer)

    var curtain := ColorRect.new()
    curtain.color = Color(0.025, 0.045, 0.095, 0.0)
    curtain.mouse_filter = Control.MOUSE_FILTER_STOP
    layer.add_child(curtain)
    curtain.set_anchors_and_offsets_preset(Control.PRESET_FULL_RECT)

    var transition := layer.create_tween()
    transition.tween_property(curtain, "color:a", 1.0, 0.24).set_trans(Tween.TRANS_SINE)
    transition.tween_callback(func() -> void:
        var error := tree.change_scene_to_file(destination)
        if error != OK:
            push_error("G8 scene transition failed: %s" % destination)
    )
    transition.tween_interval(0.04)
    transition.tween_property(curtain, "color:a", 0.0, 0.30).set_trans(Tween.TRANS_SINE)
    transition.tween_callback(layer.queue_free)
    return true
