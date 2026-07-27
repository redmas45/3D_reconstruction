"""Builds the actor asset library, run once by `scripts/build_actor_library.py`.

Generates one object per catalog class into a fresh file and saves it. Everything it
builds comes from the same functions the runtime uses, so a library asset and a
procedurally generated one are the same geometry — the library only decides *when* the
cost is paid, never *what* is built.

    blender --background --factory-startup --python blender/build_library.py -- \
            --output assets/actors/library.blend
"""

import json
import sys
from pathlib import Path

import bpy

SCRIPT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_ROOT.parent
for _path in (str(SCRIPT_ROOT), str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "src" / "domain")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import humanoid  # noqa: E402
import warm_shell  # noqa: E402
from humanoid_rig import skeleton_for_height  # noqa: E402

MARKER = "@LIBRARY@"


def emit(payload: dict) -> None:
    sys.stdout.write(f"{MARKER} {json.dumps(payload)}\n")
    sys.stdout.flush()


def parse_arguments(argv: list[str]) -> dict:
    arguments = argv[argv.index("--") + 1:] if "--" in argv else []
    parsed = {"output": "assets/actors/library.blend"}
    for index in range(0, len(arguments) - 1, 2):
        parsed[arguments[index].lstrip("-").replace("-", "_")] = arguments[index + 1]
    return parsed


def build_asset(class_name: str, asset: dict, collection: bpy.types.Collection):
    """One library object for one class, built by the same code the renderer uses."""
    material = warm_shell.build_shared_actor_material()
    actor = {
        "id": asset["object_name"],
        "class_name": class_name,
        "proxy": asset["proxy"],
        "dimensions": asset["dimensions"],
        "body_height_ratio": asset["body_height_ratio"],
        "cabin_length_ratio": asset["cabin_length_ratio"],
        "cabin_width_ratio": asset["cabin_width_ratio"],
        "color": [0.5, 0.5, 0.5],
    }
    if asset["proxy"] == "humanoid":
        skeleton = skeleton_for_height(float(asset["dimensions"][2]))
        rig, body = humanoid.instance_humanoid(
            f"{asset['object_name']}_src", humanoid.build_humanoid_mesh(skeleton),
            humanoid.build_humanoid_armature(skeleton), material, collection,
        )
        humanoid.add_subdivision(body)
        # The manifest name must land on the *rig*, because that is what carries the
        # pose and what the renderer copies. Naming the mesh body instead produces an
        # actor with no armature, which fails at the first pose rather than silently.
        # The body is renamed first so the name it currently holds is free.
        body.name = f"{asset['object_name']}_body"
        rig.name = asset["object_name"]
        return rig
    instance = bpy.data.objects.new(
        asset["object_name"], warm_shell._rigid_mesh_for_actor(actor),
    )
    if not instance.data.materials:
        instance.data.materials.append(material)
    collection.objects.link(instance)
    return instance


def main() -> None:
    arguments = parse_arguments(sys.argv)
    from actor_library import build_manifest

    bpy.ops.wm.read_factory_settings(use_empty=True)
    collection = bpy.context.scene.collection
    manifest = build_manifest()
    built = []
    for asset in manifest["assets"]:
        build_asset(asset["class_name"], asset, collection)
        built.append(asset["object_name"])
    output = Path(arguments["output"]).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=str(output), compress=True)
    emit({
        "built": built,
        "catalog_digest": manifest["catalog_digest"],
        "output": str(output),
        "bytes": output.stat().st_size,
    })


if __name__ == "__main__":
    main()
