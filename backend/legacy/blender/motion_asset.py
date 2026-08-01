import math
from pathlib import Path

import bpy

from human_builder import create_contact_shadow, human_colors


ASSET_COLLECTION_NAME = "FOR3D_Humanoid"
ASSET_ARMATURE_NAME = "FOR3D_Rig"
ACTION_PREFIX = "FOR3D_"
SUPPORTED_MOTION_CLIPS = ("idle", "walk", "brisk_walk", "run")
DEFAULT_MOTION_ASSET_PATH = (
    Path(__file__).resolve().parents[1]
    / "assets"
    / "animation"
    / "humanoid_motion_library.blend"
)
MATERIAL_ROLES = {
    "FOR3D_Upper": "upper",
    "FOR3D_Lower": "lower",
    "FOR3D_Skin": "skin",
    "FOR3D_Shoe": "shoe",
}
CLIP_CYCLE_SECONDS = {
    "idle": 2.4,
    "walk": 1.05,
    "brisk_walk": 0.82,
    "run": 0.62,
}
MINIMUM_ALPHA = 0.96


class MotionAssetError(RuntimeError):
    pass


def motion_asset_available(asset_path: Path = DEFAULT_MOTION_ASSET_PATH) -> bool:
    return asset_path.is_file()


def build_rigged_human(
    entity: dict,
    frame_count: int,
    fps: float,
    asset_path: Path = DEFAULT_MOTION_ASSET_PATH,
) -> dict:
    if not asset_path.is_file():
        raise MotionAssetError(f"Motion asset is missing: {asset_path}")
    collection, actions = _append_asset(asset_path)
    root = _root_object(str(entity["id"]))
    objects = list(collection.all_objects)
    for scene_object in objects:
        if scene_object.parent is None:
            scene_object.parent = root
    rig = _find_rig(objects)
    materials = _apply_evidence_materials(objects, entity)
    _apply_motion_action(rig, actions, entity, frame_count, fps)
    shadow = create_contact_shadow(
        root,
        float(entity["body_proportions"]["height_scale"]),
        str(entity["id"]),
    )
    return {
        "root": root,
        "rig": rig,
        "motion_rig": rig,
        "arms": [],
        "elbows": [],
        "legs": [],
        "knees": [],
        "feet": [],
        "wheels": [],
        "steering_wheels": [],
        "wheel_radius": None,
        "materials": [*materials, shadow],
        "visual_height_meters": 1.91,
        "height_scale": float(entity["body_proportions"]["height_scale"]),
        "animation_system": "rigged_nla_motion_asset",
        "motion_asset_path": str(asset_path),
    }


def _append_asset(
    asset_path: Path,
) -> tuple[bpy.types.Collection, dict[str, bpy.types.Action]]:
    action_names = [f"{ACTION_PREFIX}{clip}" for clip in SUPPORTED_MOTION_CLIPS]
    with bpy.data.libraries.load(str(asset_path), link=False) as (source, target):
        if ASSET_COLLECTION_NAME not in source.collections:
            raise MotionAssetError("Motion asset collection is missing")
        missing_actions = [name for name in action_names if name not in source.actions]
        if missing_actions:
            raise MotionAssetError(f"Motion asset actions are missing: {missing_actions}")
        target.collections = [ASSET_COLLECTION_NAME]
        target.actions = action_names
    collection = target.collections[0]
    bpy.context.scene.collection.children.link(collection)
    actions = {
        clip: action
        for clip, action in zip(SUPPORTED_MOTION_CLIPS, target.actions)
    }
    return collection, actions


def _root_object(entity_id: str) -> bpy.types.Object:
    root = bpy.data.objects.new(f"Human_{entity_id}", None)
    bpy.context.collection.objects.link(root)
    return root


def _find_rig(objects: list[bpy.types.Object]) -> bpy.types.Object:
    rigs = [
        scene_object for scene_object in objects
        if scene_object.type == "ARMATURE"
        and scene_object.name.startswith(ASSET_ARMATURE_NAME)
    ]
    if len(rigs) != 1:
        raise MotionAssetError("Motion asset must contain exactly one humanoid armature")
    return rigs[0]


def _apply_evidence_materials(
    objects: list[bpy.types.Object],
    entity: dict,
) -> list[bpy.types.Material]:
    colors = human_colors(entity)
    alpha = 1.0 if entity["fidelity_tier"] == "supported" else MINIMUM_ALPHA
    materials: dict[str, bpy.types.Material] = {}
    for scene_object in objects:
        for slot in getattr(scene_object, "material_slots", []):
            role = _material_role(slot.material)
            if slot.material is None:
                continue
            material_key = role or f"source:{slot.material.name}"
            material_color = (
                colors[role]
                if role else tuple(slot.material.diffuse_color[:3])
            )
            if material_key not in materials:
                materials[material_key] = _material_copy(
                    slot.material,
                    entity,
                    material_key,
                    material_color,
                    alpha,
                )
            slot.material = materials[material_key]
    return list(materials.values())


def _material_role(material: bpy.types.Material | None) -> str | None:
    if material is None:
        return None
    return next(
        (role for prefix, role in MATERIAL_ROLES.items() if material.name.startswith(prefix)),
        None,
    )


def _material_copy(
    material: bpy.types.Material,
    entity: dict,
    role: str,
    color: tuple[float, float, float] | list[float],
    alpha: float,
) -> bpy.types.Material:
    copied = material.copy()
    copied.name = f"Motion_{role}_{entity['id']}"
    copied.diffuse_color = (*color, alpha)
    shader = copied.node_tree.nodes.get("Principled BSDF") if copied.use_nodes else None
    if shader is not None:
        shader.inputs["Base Color"].default_value = (*color, 1.0)
        shader.inputs["Alpha"].default_value = alpha
    if hasattr(copied, "surface_render_method"):
        copied.surface_render_method = "DITHERED"
    return copied


def _apply_motion_action(
    rig: bpy.types.Object,
    actions: dict[str, bpy.types.Action],
    entity: dict,
    frame_count: int,
    fps: float,
) -> None:
    profile = entity.get("motion_profile", {})
    clip = str(profile.get("clip", entity["animation"]["state"]))
    if clip not in actions:
        clip = "idle" if entity["animation"]["state"] == "idle" else "walk"
    cadence = max(0.1, float(profile.get("cadence_scale", 1.0)))
    phase = float(profile.get("phase_offset", entity["animation"].get("phase_offset", 0.0))) % 1.0
    blend_seconds = max(0.0, float(profile.get("blend_seconds", 0.18)))
    _create_nla_strip(
        rig,
        actions[clip],
        clip,
        frame_count,
        fps,
        cadence,
        phase,
        blend_seconds,
    )


def _create_nla_strip(
    rig: bpy.types.Object,
    action: bpy.types.Action,
    clip: str,
    frame_count: int,
    fps: float,
    cadence: float,
    phase: float,
    blend_seconds: float,
) -> None:
    rig.animation_data_create()
    rig.animation_data.action = None
    track = rig.animation_data.nla_tracks.new()
    track.name = f"Evidence motion · {clip}"
    action_length = max(1.0, float(action.frame_range[1] - action.frame_range[0]))
    cycle_frames = max(1.0, fps * CLIP_CYCLE_SECONDS[clip] / cadence)
    start_frame = 1.0 - phase * cycle_frames
    strip = track.strips.new(clip, int(math.floor(start_frame)), action)
    strip.action_frame_start = float(action.frame_range[0])
    strip.action_frame_end = float(action.frame_range[1])
    strip.scale = cycle_frames / action_length
    strip.repeat = max(1.0, math.ceil((frame_count - start_frame) / cycle_frames))
    strip.blend_type = "REPLACE"
    strip.extrapolation = "HOLD_FORWARD"
    strip.blend_in = min(cycle_frames * 0.25, blend_seconds * fps)
    strip.blend_out = strip.blend_in
