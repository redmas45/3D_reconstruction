import argparse
import math
import sys
from pathlib import Path

import bpy


ASSET_COLLECTION_NAME = "FOR3D_Humanoid"
ARMATURE_NAME = "FOR3D_Rig"
ACTION_PREFIX = "FOR3D_"
CLIP_FRAME_END = 25
CONTROL_BONES = (
    "pelvis",
    "spine",
    "upper_arm.L",
    "forearm.L",
    "upper_arm.R",
    "forearm.R",
    "thigh.L",
    "shin.L",
    "foot.L",
    "thigh.R",
    "shin.R",
    "foot.R",
)


def parse_arguments() -> argparse.Namespace:
    arguments = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description="Build the reusable forensic humanoid asset")
    parser.add_argument("--output", required=True)
    return parser.parse_args(arguments)


def clear_file() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in list(bpy.data.collections):
        if collection.users == 0:
            bpy.data.collections.remove(collection)


def build_asset(output_path: Path) -> None:
    clear_file()
    collection = bpy.data.collections.new(ASSET_COLLECTION_NAME)
    bpy.context.scene.collection.children.link(collection)
    armature = _build_armature(collection)
    materials = _materials()
    _build_character_geometry(collection, armature, materials)
    _build_actions(armature)
    armature.animation_data_clear()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output_path))


def _build_armature(collection: bpy.types.Collection) -> bpy.types.Object:
    armature_data = bpy.data.armatures.new(f"{ARMATURE_NAME}_Data")
    armature = bpy.data.objects.new(ARMATURE_NAME, armature_data)
    collection.objects.link(armature)
    bpy.context.view_layer.objects.active = armature
    armature.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    _add_bones(armature_data)
    bpy.ops.object.mode_set(mode="POSE")
    for pose_bone in armature.pose.bones:
        pose_bone.rotation_mode = "XYZ"
    bpy.ops.object.mode_set(mode="OBJECT")
    armature.show_in_front = True
    return armature


def _add_bones(armature_data: bpy.types.Armature) -> None:
    bones = {}
    bones["root"] = _bone(armature_data, "root", (0, 0, 0), (0, 0, 0.25))
    bones["pelvis"] = _bone(armature_data, "pelvis", (0, 0, 0.82), (0, 0, 1.04), bones["root"])
    bones["spine"] = _bone(armature_data, "spine", (0, 0, 1.04), (0, 0, 1.50), bones["pelvis"], True)
    bones["head"] = _bone(armature_data, "head", (0, 0, 1.50), (0, 0, 1.88), bones["spine"], True)
    _add_limb_bones(armature_data, bones, "L", -1.0)
    _add_limb_bones(armature_data, bones, "R", 1.0)


def _add_limb_bones(
    armature_data: bpy.types.Armature,
    bones: dict,
    suffix: str,
    side: float,
) -> None:
    upper_arm = _bone(
        armature_data,
        f"upper_arm.{suffix}",
        (side * 0.32, 0, 1.43),
        (side * 0.34, 0, 1.04),
        bones["spine"],
    )
    _bone(
        armature_data,
        f"forearm.{suffix}",
        upper_arm.tail,
        (side * 0.35, 0, 0.68),
        upper_arm,
        True,
    )
    thigh = _bone(
        armature_data,
        f"thigh.{suffix}",
        (side * 0.14, 0, 0.90),
        (side * 0.14, 0, 0.48),
        bones["pelvis"],
    )
    shin = _bone(
        armature_data,
        f"shin.{suffix}",
        thigh.tail,
        (side * 0.14, 0, 0.10),
        thigh,
        True,
    )
    _bone(
        armature_data,
        f"foot.{suffix}",
        shin.tail,
        (side * 0.14, -0.24, 0.08),
        shin,
        True,
    )


def _bone(
    armature_data: bpy.types.Armature,
    name: str,
    head: tuple[float, float, float],
    tail: tuple[float, float, float],
    parent: bpy.types.EditBone | None = None,
    connected: bool = False,
) -> bpy.types.EditBone:
    bone = armature_data.edit_bones.new(name)
    bone.head = head
    bone.tail = tail
    bone.parent = parent
    bone.use_connect = connected
    return bone


def _materials() -> dict[str, bpy.types.Material]:
    return {
        "upper": _material("FOR3D_Upper", (0.08, 0.62, 0.78, 1.0)),
        "lower": _material("FOR3D_Lower", (0.045, 0.07, 0.11, 1.0)),
        "skin": _material("FOR3D_Skin", (0.46, 0.31, 0.23, 1.0)),
        "shoe": _material("FOR3D_Shoe", (0.015, 0.022, 0.032, 1.0)),
    }


def _material(name: str, color: tuple[float, float, float, float]) -> bpy.types.Material:
    material = bpy.data.materials.new(name)
    material.diffuse_color = color
    material.use_nodes = True
    shader = material.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = color
    shader.inputs["Roughness"].default_value = 0.72
    return material


def _build_character_geometry(
    collection: bpy.types.Collection,
    armature: bpy.types.Object,
    materials: dict[str, bpy.types.Material],
) -> None:
    _body_geometry(collection, armature, materials)
    for suffix, side in (("L", -1.0), ("R", 1.0)):
        _limb_geometry(collection, armature, materials, suffix, side)


def _body_geometry(
    collection: bpy.types.Collection,
    armature: bpy.types.Object,
    materials: dict[str, bpy.types.Material],
) -> None:
    _cube_part(collection, armature, "Pelvis", "pelvis", (0, 0, 0.91), (0.25, 0.16, 0.16), materials["lower"])
    _cube_part(collection, armature, "Torso", "spine", (0, 0, 1.27), (0.31, 0.18, 0.32), materials["upper"])
    _sphere_part(collection, armature, "Head", "head", (0, 0, 1.72), 0.19, materials["skin"])


def _limb_geometry(
    collection: bpy.types.Collection,
    armature: bpy.types.Object,
    materials: dict[str, bpy.types.Material],
    suffix: str,
    side: float,
) -> None:
    _cylinder_part(collection, armature, f"UpperArm.{suffix}", f"upper_arm.{suffix}", (side * 0.33, 0, 1.24), 0.085, 0.42, materials["upper"])
    _cylinder_part(collection, armature, f"Forearm.{suffix}", f"forearm.{suffix}", (side * 0.35, 0, 0.86), 0.070, 0.38, materials["skin"])
    _sphere_part(collection, armature, f"Hand.{suffix}", f"forearm.{suffix}", (side * 0.35, 0, 0.65), 0.078, materials["skin"])
    _cylinder_part(collection, armature, f"Thigh.{suffix}", f"thigh.{suffix}", (side * 0.14, 0, 0.68), 0.115, 0.46, materials["lower"])
    _cylinder_part(collection, armature, f"Shin.{suffix}", f"shin.{suffix}", (side * 0.14, 0, 0.29), 0.092, 0.42, materials["lower"])
    _cube_part(collection, armature, f"Foot.{suffix}", f"foot.{suffix}", (side * 0.14, -0.12, 0.08), (0.105, 0.20, 0.07), materials["shoe"])


def _cube_part(
    collection: bpy.types.Collection,
    armature: bpy.types.Object,
    name: str,
    bone_name: str,
    location: tuple[float, float, float],
    dimensions: tuple[float, float, float],
    material: bpy.types.Material,
) -> None:
    bpy.ops.mesh.primitive_cube_add(location=location)
    part = bpy.context.object
    part.scale = dimensions
    _finish_part(part, collection, armature, name, bone_name, material, 0.06)


def _cylinder_part(
    collection: bpy.types.Collection,
    armature: bpy.types.Object,
    name: str,
    bone_name: str,
    location: tuple[float, float, float],
    radius: float,
    depth: float,
    material: bpy.types.Material,
) -> None:
    bpy.ops.mesh.primitive_cylinder_add(vertices=12, radius=radius, depth=depth, location=location)
    _finish_part(bpy.context.object, collection, armature, name, bone_name, material, 0.035)


def _sphere_part(
    collection: bpy.types.Collection,
    armature: bpy.types.Object,
    name: str,
    bone_name: str,
    location: tuple[float, float, float],
    radius: float,
    material: bpy.types.Material,
) -> None:
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=2, radius=radius, location=location)
    _finish_part(bpy.context.object, collection, armature, name, bone_name, material, 0.025)


def _finish_part(
    part: bpy.types.Object,
    collection: bpy.types.Collection,
    armature: bpy.types.Object,
    name: str,
    bone_name: str,
    material: bpy.types.Material,
    bevel_width: float,
) -> None:
    part.name = name
    _move_to_collection(part, collection)
    bpy.context.view_layer.objects.active = part
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    bevel = part.modifiers.new("Soft profile", "BEVEL")
    bevel.width = bevel_width
    bevel.segments = 2
    part.data.materials.append(material)
    group = part.vertex_groups.new(name=bone_name)
    group.add(range(len(part.data.vertices)), 1.0, "REPLACE")
    modifier = part.modifiers.new("FOR3D Armature", "ARMATURE")
    modifier.object = armature


def _move_to_collection(part: bpy.types.Object, collection: bpy.types.Collection) -> None:
    for source_collection in list(part.users_collection):
        source_collection.objects.unlink(part)
    collection.objects.link(part)


def _build_actions(armature: bpy.types.Object) -> None:
    _idle_action(armature)
    _locomotion_action(armature, "walk", 0.44, 0.62, 0.30, 0.04)
    _locomotion_action(armature, "brisk_walk", 0.60, 0.82, 0.46, 0.09)
    _locomotion_action(armature, "run", 0.82, 1.05, 0.68, 0.17)
    _turn_action(armature, "turn_left", 1.0)
    _turn_action(armature, "turn_right", -1.0)
    _stop_action(armature)


def _idle_action(armature: bpy.types.Object) -> None:
    action = _new_action(armature, "idle")
    for frame, breath in ((1, 0.0), (13, 1.0), (25, 0.0)):
        _reset_pose(armature)
        armature.pose.bones["pelvis"].location.z = breath * 0.015
        armature.pose.bones["spine"].rotation_euler[0] = breath * math.radians(1.5)
        _keyframe_controls(armature, frame)
    _finish_action(action)


def _locomotion_action(
    armature: bpy.types.Object,
    clip_name: str,
    leg_swing: float,
    knee_bend: float,
    arm_swing: float,
    forward_lean: float,
) -> None:
    action = _new_action(armature, clip_name)
    phases = ((1, 0.0), (7, 1.0), (13, 0.0), (19, -1.0), (25, 0.0))
    for frame, phase in phases:
        _reset_pose(armature)
        _locomotion_pose(armature, phase, leg_swing, knee_bend, arm_swing, forward_lean)
        _keyframe_controls(armature, frame)
    _finish_action(action)


def _locomotion_pose(
    armature: bpy.types.Object,
    phase: float,
    leg_swing: float,
    knee_bend: float,
    arm_swing: float,
    forward_lean: float,
) -> None:
    pose = armature.pose.bones
    pose["spine"].rotation_euler[0] = forward_lean
    pose["pelvis"].rotation_euler[1] = phase * 0.045
    pose["pelvis"].location.z = abs(phase) * 0.025
    for suffix, direction in (("L", 1.0), ("R", -1.0)):
        leg_phase = phase * direction
        pose[f"thigh.{suffix}"].rotation_euler[0] = leg_phase * leg_swing
        pose[f"shin.{suffix}"].rotation_euler[0] = max(0.0, -leg_phase) * knee_bend
        pose[f"foot.{suffix}"].rotation_euler[0] = -max(0.0, -leg_phase) * knee_bend * 0.65
        pose[f"upper_arm.{suffix}"].rotation_euler[0] = -leg_phase * arm_swing
        pose[f"forearm.{suffix}"].rotation_euler[0] = 0.18 + max(0.0, leg_phase) * 0.24


def _turn_action(armature: bpy.types.Object, clip_name: str, direction: float) -> None:
    action = _new_action(armature, clip_name)
    for frame, weight in ((1, 0.0), (13, 1.0), (25, 0.0)):
        _reset_pose(armature)
        armature.pose.bones["pelvis"].rotation_euler[2] = direction * weight * 0.22
        armature.pose.bones["spine"].rotation_euler[2] = direction * weight * 0.30
        _keyframe_controls(armature, frame)
    _finish_action(action)


def _stop_action(armature: bpy.types.Object) -> None:
    action = _new_action(armature, "stop")
    for frame, weight in ((1, 1.0), (13, 0.35), (25, 0.0)):
        _reset_pose(armature)
        _locomotion_pose(armature, weight, 0.38, 0.50, 0.24, 0.03)
        _keyframe_controls(armature, frame)
    _finish_action(action)


def _new_action(armature: bpy.types.Object, clip_name: str) -> bpy.types.Action:
    action = bpy.data.actions.new(f"{ACTION_PREFIX}{clip_name}")
    action.use_fake_user = True
    armature.animation_data_create()
    armature.animation_data.action = action
    return action


def _reset_pose(armature: bpy.types.Object) -> None:
    for bone_name in CONTROL_BONES:
        pose_bone = armature.pose.bones[bone_name]
        pose_bone.location = (0.0, 0.0, 0.0)
        pose_bone.rotation_euler = (0.0, 0.0, 0.0)
        pose_bone.scale = (1.0, 1.0, 1.0)


def _keyframe_controls(armature: bpy.types.Object, frame: int) -> None:
    for bone_name in CONTROL_BONES:
        pose_bone = armature.pose.bones[bone_name]
        pose_bone.keyframe_insert("location", frame=frame, group=bone_name)
        pose_bone.keyframe_insert("rotation_euler", frame=frame, group=bone_name)


def _finish_action(action: bpy.types.Action) -> None:
    for curve in action.fcurves:
        for keyframe in curve.keyframe_points:
            keyframe.interpolation = "BEZIER"


def main() -> None:
    arguments = parse_arguments()
    build_asset(Path(arguments.output).resolve())


if __name__ == "__main__":
    main()
