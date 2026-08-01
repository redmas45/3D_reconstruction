import math

import bpy
from mathutils import Vector

from animation import animate_entity
from evidence_compositor import configure_evidence_compositor
from environment_builder import build_environment, build_path_trail
from frame_rate import blender_frame_rate
from hud import build_hud
from human_builder import build_human
from motion_asset import (
    MotionAssetError,
    build_rigged_human,
    motion_asset_available,
)
from render_device import CYCLES_RENDER_ENGINE, configure_cycles_render
from render_passes import (
    ACTOR_PASS_INDEX,
    ENVIRONMENT_PASS_INDEX,
    HUD_PASS_INDEX,
    UNCERTAINTY_PASS_INDEX,
    assign_pass_index,
)
from vehicle_builder import build_vehicle
from visual_alignment import align_entity_scale


RENDERABLE_HUMANS = {"person"}
DEFAULT_RENDER_SCALE_PERCENT = 75
WORKBENCH_RENDER_ENGINE = "BLENDER_WORKBENCH"


def build_scene(plan: dict) -> bpy.types.Scene:
    clear_scene()
    scene = bpy.context.scene
    configure_render(scene, plan)
    configure_evidence_compositor(scene, plan)
    camera = build_camera(plan["camera"])
    build_lighting()
    objects_before_environment = set(scene.objects)
    build_environment(plan)
    assign_pass_index(set(scene.objects) - objects_before_environment, ENVIRONMENT_PASS_INDEX)
    show_debug_paths = plan["environment"].get("show_debug_paths", False)
    for entity in plan["entities"]:
        objects_before_entity = set(scene.objects)
        parts = _build_actor(entity, plan)
        align_entity_scale(parts, entity, scene, camera)
        animate_entity(parts, entity, plan["frame_count"], float(plan["fps"]))
        if show_debug_paths:
            build_path_trail(entity)
        entity_objects = set(scene.objects) - objects_before_entity
        uncertainty_objects = {
            scene_object for scene_object in entity_objects
            if scene_object.name.startswith(("Confidence_", "Path_"))
        }
        assign_pass_index(entity_objects - uncertainty_objects, ACTOR_PASS_INDEX)
        assign_pass_index(uncertainty_objects, UNCERTAINTY_PASS_INDEX)
    objects_before_hud = set(scene.objects)
    build_hud(plan, camera)
    assign_pass_index(set(scene.objects) - objects_before_hud, HUD_PASS_INDEX)
    scene.frame_start = 1
    scene.frame_end = plan["frame_count"]
    return scene


def _build_actor(entity: dict, plan: dict) -> dict:
    if entity["kind"] not in RENDERABLE_HUMANS:
        return build_vehicle(entity)
    if entity["fidelity_tier"] == "weak" or not motion_asset_available():
        return build_human(entity)
    try:
        return build_rigged_human(
            entity,
            int(plan["frame_count"]),
            float(plan["fps"]),
        )
    except MotionAssetError as error:
        print(f"MOTION_ASSET_FALLBACK entity={entity['id']} reason={error}", flush=True)
        return build_human(entity)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for data_collection in (bpy.data.curves, bpy.data.materials, bpy.data.cameras, bpy.data.lights):
        for data_block in list(data_collection):
            if data_block.users == 0:
                data_collection.remove(data_block)


def configure_render(scene: bpy.types.Scene, plan: dict) -> None:
    render_contract = plan.get("render", {})
    scene.render.engine = render_contract.get("engine", "BLENDER_EEVEE_NEXT")
    if scene.render.engine == WORKBENCH_RENDER_ENGINE:
        _configure_workbench(scene)
    if scene.render.engine == CYCLES_RENDER_ENGINE:
        configure_cycles_render(scene, render_contract)
    scene.render.resolution_x = int(render_contract["source_width"])
    scene.render.resolution_y = int(render_contract["source_height"])
    scene.render.resolution_percentage = int(
        render_contract.get("preview_scale_percent", DEFAULT_RENDER_SCALE_PERCENT)
    )
    nominal_fps, fps_base = blender_frame_rate(float(plan["fps"]))
    scene.render.fps = nominal_fps
    scene.render.fps_base = fps_base
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = bool(
        plan.get("environment", {}).get("hybrid_backplate_enabled", False),
    )
    scene.world.color = (0.006, 0.010, 0.018)
    scene.render.use_file_extension = True
    scene.view_settings.look = "AgX - Medium High Contrast"
    _configure_forensic_outlines(scene)


def _configure_forensic_outlines(scene: bpy.types.Scene) -> None:
    scene.render.use_freestyle = True
    scene.render.line_thickness = 1.15
    if not hasattr(scene, "view_layers"):
        return
    line_sets = scene.view_layers["ViewLayer"].freestyle_settings.linesets
    if not line_sets:
        return
    line_style = line_sets[0].linestyle
    line_style.color = (0.015, 0.025, 0.035)
    line_style.alpha = 0.80


def _configure_workbench(scene: bpy.types.Scene) -> None:
    scene.display.shading.light = "STUDIO"
    scene.display.shading.color_type = "MATERIAL"
    scene.display.shading.show_shadows = True
    scene.display.shading.show_cavity = True
    scene.display.shading.cavity_type = "WORLD"
    scene.display.shading.show_specular_highlight = True


def build_camera(camera_contract: dict) -> bpy.types.Object:
    camera_data = bpy.data.cameras.new("EvidenceCamera")
    camera = bpy.data.objects.new("EvidenceCamera", camera_data)
    bpy.context.collection.objects.link(camera)
    camera.location = camera_contract["position"]
    camera_data.lens = camera_contract["focal_length_mm"]
    camera_data.clip_start = 0.05
    target = Vector(camera_contract["look_at"])
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()
    bpy.context.scene.camera = camera
    return camera


def build_lighting() -> None:
    world = bpy.context.scene.world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.012, 0.018, 0.028, 1.0)
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.20
    _area_light("Key", (4.0, 2.0, 8.0), 1300.0, 7.0, (0.70, 0.91, 1.0))
    _area_light("Fill", (-6.0, 8.0, 5.0), 700.0, 9.0, (0.18, 0.52, 0.62))
    bpy.ops.object.light_add(type="SUN", location=(0.0, 0.0, 12.0))
    sun = bpy.context.object
    sun.data.energy = 1.5
    sun.rotation_euler = (math.radians(24.0), math.radians(-18.0), math.radians(32.0))


def _area_light(
    name: str,
    location: tuple[float, float, float],
    energy: float,
    size: float,
    color: tuple[float, float, float],
) -> None:
    light_data = bpy.data.lights.new(name=name, type="AREA")
    light_data.energy = energy
    light_data.shape = "DISK"
    light_data.size = size
    light_data.color = color
    light = bpy.data.objects.new(name, light_data)
    bpy.context.collection.objects.link(light)
    light.location = location
    light.rotation_euler = (0.0, 0.0, 0.0)
