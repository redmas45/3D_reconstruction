"""Packages one Mixamo character and matching clips into the runtime asset contract."""

import argparse
import sys
from pathlib import Path

import bpy


ASSET_COLLECTION_NAME = "FOR3D_Humanoid"
ASSET_ARMATURE_NAME = "FOR3D_Rig"
ACTION_PREFIX = "FOR3D_"
REQUIRED_CLIPS = ("idle", "walk", "brisk_walk", "run")


def parse_arguments() -> argparse.Namespace:
    arguments = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--character", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--clip",
        action="append",
        default=[],
        help="Clip mapping in name=path form; required names: idle, walk, brisk_walk, run",
    )
    return parser.parse_args(arguments)


def package_mixamo_asset(
    character_path: Path,
    clip_paths: dict[str, Path],
    output_path: Path,
) -> None:
    _validate_inputs(character_path, clip_paths)
    _clear_file()
    character_objects = _import_fbx(character_path)
    character_rig = _single_armature(character_objects, "character")
    character_rig.name = ASSET_ARMATURE_NAME
    _discard_active_action(character_rig)
    collection = _character_collection(character_objects)
    for clip_name, clip_path in clip_paths.items():
        _import_clip_action(character_rig, clip_name, clip_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.view_layer.layer_collection.children[collection.name].exclude = False
    bpy.ops.wm.save_as_mainfile(filepath=str(output_path))


def _validate_inputs(character_path: Path, clip_paths: dict[str, Path]) -> None:
    missing_names = sorted(set(REQUIRED_CLIPS) - set(clip_paths))
    if missing_names:
        raise ValueError(f"Missing required Mixamo clips: {missing_names}")
    missing_paths = [
        path for path in (character_path, *clip_paths.values())
        if not path.is_file()
    ]
    if missing_paths:
        raise FileNotFoundError(f"Mixamo FBX files are missing: {missing_paths}")


def _clear_file() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for collection in list(bpy.data.collections):
        if collection.users == 0:
            bpy.data.collections.remove(collection)


def _import_fbx(path: Path) -> list[bpy.types.Object]:
    objects_before = set(bpy.data.objects)
    result = bpy.ops.import_scene.fbx(filepath=str(path.resolve()))
    if "FINISHED" not in result:
        raise RuntimeError(f"Blender could not import Mixamo FBX: {path.name}")
    return list(set(bpy.data.objects) - objects_before)


def _single_armature(
    objects: list[bpy.types.Object],
    source_name: str,
) -> bpy.types.Object:
    armatures = [scene_object for scene_object in objects if scene_object.type == "ARMATURE"]
    if len(armatures) != 1:
        raise ValueError(f"Mixamo {source_name} must contain exactly one armature")
    return armatures[0]


def _discard_active_action(armature: bpy.types.Object) -> None:
    if armature.animation_data is not None:
        armature.animation_data.action = None


def _character_collection(
    objects: list[bpy.types.Object],
) -> bpy.types.Collection:
    collection = bpy.data.collections.new(ASSET_COLLECTION_NAME)
    bpy.context.scene.collection.children.link(collection)
    for scene_object in objects:
        for current_collection in list(scene_object.users_collection):
            current_collection.objects.unlink(scene_object)
        collection.objects.link(scene_object)
    return collection


def _import_clip_action(
    character_rig: bpy.types.Object,
    clip_name: str,
    clip_path: Path,
) -> None:
    imported_objects = _import_fbx(clip_path)
    clip_rig = _single_armature(imported_objects, f"clip '{clip_name}'")
    action = getattr(getattr(clip_rig, "animation_data", None), "action", None)
    if action is None:
        raise ValueError(f"Mixamo clip '{clip_name}' contains no animation action")
    _validate_skeleton_compatibility(character_rig, clip_rig, clip_name)
    packaged_action = action.copy()
    packaged_action.name = f"{ACTION_PREFIX}{clip_name}"
    packaged_action.use_fake_user = True
    _remove_imported_objects(imported_objects)


def _validate_skeleton_compatibility(
    character_rig: bpy.types.Object,
    clip_rig: bpy.types.Object,
    clip_name: str,
) -> None:
    character_bones = set(character_rig.pose.bones.keys())
    clip_bones = set(clip_rig.pose.bones.keys())
    missing_bones = sorted(clip_bones - character_bones)
    if missing_bones:
        raise ValueError(
            f"Mixamo clip '{clip_name}' does not match the character skeleton: "
            f"{missing_bones[:8]}"
        )


def _remove_imported_objects(objects: list[bpy.types.Object]) -> None:
    for scene_object in objects:
        bpy.data.objects.remove(scene_object, do_unlink=True)


def _clip_mapping(values: list[str]) -> dict[str, Path]:
    mappings = {}
    for value in values:
        if "=" not in value:
            raise ValueError("Each --clip value must use name=path")
        name, path_text = value.split("=", 1)
        if name not in REQUIRED_CLIPS or name in mappings:
            raise ValueError(f"Unsupported or duplicate Mixamo clip name: {name}")
        mappings[name] = Path(path_text).resolve()
    return mappings


def main() -> None:
    arguments = parse_arguments()
    package_mixamo_asset(
        arguments.character.resolve(),
        _clip_mapping(arguments.clip),
        arguments.output.resolve(),
    )


if __name__ == "__main__":
    main()
