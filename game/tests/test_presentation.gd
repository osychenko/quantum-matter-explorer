extends SceneTree
## G8 smoke/regression: menu gating, room HUD, visual-only VFX and navigation.
## All scenes disable persistence before they are added to the active tree.

const FLOW = preload("res://scripts/GameFlow.gd")
const FX = preload("res://scripts/AbilityVFX.gd")
var failures: Array[String] = []


func _initialize() -> void:
    call_deferred("run_tests")


func check(value: bool, description: String) -> void:
    if not value:
        failures.append(description)
        push_error(description)


func run_tests() -> void:
    check(ProjectSettings.get_setting("application/run/main_scene") == FLOW.MENU,
            "G8 menu is the default main scene")
    check(ResourceLoader.exists(FLOW.MENU) and ResourceLoader.exists(FLOW.RUINS) \
            and ResourceLoader.exists(FLOW.CALIBRATION),
            "All three routes point to real scenes")
    check(not FLOW.navigate(self, "res://scenes/nonexistent.tscn"),
            "Scene navigator ignores invalid scene paths")
    check(not root.has_node(FLOW.FADE_NODE),
            "Invalid navigation creates no transition overlay")

    var menu_scene: PackedScene = load(FLOW.MENU)
    check(menu_scene != null, "Menu scene loads")
    if menu_scene == null:
        quit(1)
        return
    var menu = menu_scene.instantiate()
    menu.persistence_enabled = false
    root.add_child(menu)
    await process_frame
    check(menu.ruins_button != null and not menu.ruins_button.disabled,
            "Starting or continuing Barrier Ruins is always possible")
    check(menu.calibration_button.disabled,
            "Calibration is locked before the first chamber is complete")
    check(not FLOW.calibration_unlocked(menu.progression),
            "Progression gate starts closed")
    menu.progression.chamber_cleared = true
    menu.refresh_progress()
    check(not menu.calibration_button.disabled and FLOW.calibration_unlocked(menu.progression),
            "Completing Barrier Ruins opens the second room in the menu")
    check(menu.calibration_button.text.contains("UNLOCKED"),
            "Menu announces the unlocked state")
    menu.queue_free()
    await process_frame

    var ruins_scene: PackedScene = load(FLOW.RUINS)
    check(ruins_scene != null, "Barrier Ruins still loads")
    if ruins_scene == null:
        quit(1)
        return
    var ruins = ruins_scene.instantiate()
    ruins.persistence_enabled = false
    root.add_child(ruins)
    await physics_frame
    check(ruins.ability_fx != null and ruins.get_node_or_null("AbilityVFX") != null,
            "World-space effects exist without changing physics nodes")
    check(ruins.ability_strip != null and ruins.ability_strip._labels.size() == 3,
            "The ability status strip displays E, Q and R")
    var player = ruins.player
    var before_coherence: float = player.coherence
    var before_location: Vector2 = player.global_position
    ruins.ability_fx.play_event("echo", player.global_position)
    check(ruins.ability_fx.bursts.size() == 1,
            "Visual echo event creates one transient burst")
    check(player.coherence == before_coherence and player.global_position == before_location,
            "Visual events consume no resource and do not move the player")
    player.tunneling_unlocked = true
    player.tunnel_cooldown_remaining = 1.2
    ruins.ability_strip.refresh_display()
    check(ruins.ability_strip._labels[0].text.contains("1.2s"),
            "Ability HUD displays actual cooldown")
    player.tunnel_cooldown_remaining = 0.0
    ruins.ability_strip.refresh_display()
    check(ruins.ability_strip._labels[0].text.contains("READY"),
            "Ability HUD returns to ready state")
    ruins.queue_free()
    await process_frame

    var chamber_scene: PackedScene = load(FLOW.CALIBRATION)
    check(chamber_scene != null, "Calibration Chamber still loads")
    if chamber_scene != null:
        var chamber = chamber_scene.instantiate()
        chamber.persistence_enabled = false
        root.add_child(chamber)
        await physics_frame
        check(chamber.ability_fx != null and chamber.ability_strip != null,
                "Second room shares the feedback and cooldown systems")
        check(chamber.lab != null and not chamber.lab.visible,
                "G7 measurement lab starts closed")
        chamber.queue_free()
        await process_frame

    var vfx = FX.new()
    root.add_child(vfx)
    vfx.play_event("tunnel", Vector2(10, 20), false)
    check(vfx.bursts.size() == 1, "Effect lifetime begins with one burst")
    vfx._process(FX.LIFETIME + 0.02)
    check(vfx.bursts.is_empty(), "Expired particle effects are removed")
    vfx.queue_free()
    await process_frame

    if failures.is_empty():
        print("PASS test_presentation.gd — menu, gating, HUD, VFX, both scenes")
    else:
        printerr("FAIL test_presentation.gd: %d failure(s)" % failures.size())
    quit(0 if failures.is_empty() else 1)
