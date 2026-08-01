"""The reusable per-job scene shell (Implementation_plan.md §6.3).

Built once by `open_job` and kept for the whole job. Gaps only instance actors against
it and bind animation; geometry, materials, camera, and lighting are never rebuilt.
That is what turns the §6.1 per-gap overhead into a one-off cost.

Two properties matter more than they look:

  * **One shared material** (§6.4). Per-actor colours arrive through the object colour
    attribute and are read by a single Object Info node, so every actor in the video
    resolves to one compiled shader instead of one compile per clothing colour.
  * **Shared mesh datablocks.** `bpy.data.objects.new(name, existing_mesh)` creates an
    instance, not a copy, so an actor costs almost nothing to add or remove.

Imports `bpy`; only ever executed inside Blender. Deliberately free of project
business logic — it receives validated dicts and builds geometry (§3).
"""

import math
import sys
from pathlib import Path

import bpy

SCRIPT_ROOT = Path(__file__).resolve().parent
DOMAIN_ROOT = SCRIPT_ROOT.parent / "src" / "domain"
for _path in (str(SCRIPT_ROOT), str(DOMAIN_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import humanoid  # noqa: E402
# The skeleton is geometry shared by the rig builder and the gait, and is stdlib-only.
# Everything that decides *how* an actor moves stays on the host; this only says where
# the bones are (see the boundary test).
from humanoid_rig import skeleton_for_height  # noqa: E402


SHELL_COLLECTION_NAME = "FOR3D_Shell"
ACTOR_COLLECTION_NAME = "FOR3D_Actors"
SHARED_ACTOR_MATERIAL_NAME = "FOR3D_ActorShared"

PROXY_MESH_PREFIX = "FOR3D_Proxy"
CYLINDER_SIDES = 16
MINIMUM_CABIN_HEIGHT_METERS = 0.01
# The cabin sits forward of centre, which is what distinguishes a truck from a van.
CABIN_FORWARD_BIAS = 0.18

DEFAULT_FOCAL_LENGTH_MM = 35.0
DEFAULT_SENSOR_WIDTH_MM = 36.0
DEFAULT_CAMERA_LOCATION = (0.0, -12.0, 4.5)
DEFAULT_CAMERA_ROTATION_DEGREES = (68.0, 0.0, 0.0)
DEFAULT_RESOLUTION = (1280, 720)

SUN_ENERGY = 3.0
SUN_LOCATION = (4.0, -6.0, 9.0)
SUN_ROTATION_RADIANS = (0.6, 0.2, 0.3)

# Ambient fill. With a transparent film the world contributes light but no pixels, so
# this only lifts the side of an actor the sun does not reach — without it, half of
# every figure is pure black and reads as a cutout against real footage.
WORLD_AMBIENT_STRENGTH = 0.35
WORLD_AMBIENT_COLOR = (0.42, 0.45, 0.5, 1.0)

SHADOW_CATCHER_NAME = "FOR3D_ShadowCatcher"
SHADOW_CATCHER_SIZE_METERS = 200.0
# Shadow catchers are a Cycles feature. Under EEVEE the same plane renders as an opaque
# ground filling the frame, which would bury the plate under grey instead of compositing
# actors onto it — so the catcher exists only when Cycles will honour it.
SHADOW_CATCHER_ENGINES = frozenset({"CYCLES"})


class ShellStateError(RuntimeError):
    """The warm scene is not in the state the next command requires."""


# --------------------------------------------------------------------------
# Collections
# --------------------------------------------------------------------------

def _ensure_collection(name: str) -> bpy.types.Collection:
    existing = bpy.data.collections.get(name)
    if existing is not None:
        return existing
    collection = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(collection)
    return collection


def actor_collection() -> bpy.types.Collection:
    return _ensure_collection(ACTOR_COLLECTION_NAME)


# --------------------------------------------------------------------------
# Shared material — §6.4
# --------------------------------------------------------------------------

def build_shared_actor_material() -> bpy.types.Material:
    """One shader for every actor; colour arrives per object, not per material."""
    existing = bpy.data.materials.get(SHARED_ACTOR_MATERIAL_NAME)
    if existing is not None:
        return existing
    material = bpy.data.materials.new(SHARED_ACTOR_MATERIAL_NAME)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    principled = nodes.get("Principled BSDF")
    if principled is None:
        return material
    object_info = nodes.new("ShaderNodeObjectInfo")
    object_info.location = (principled.location.x - 320, principled.location.y)
    material.node_tree.links.new(object_info.outputs["Color"], principled.inputs["Base Color"])
    principled.inputs["Roughness"].default_value = 0.72
    return material


# --------------------------------------------------------------------------
# Mesh library
# --------------------------------------------------------------------------

def _detach_mesh(temporary_object: bpy.types.Object, mesh_name: str) -> bpy.types.Mesh:
    mesh = temporary_object.data
    mesh.name = mesh_name
    bpy.data.objects.remove(temporary_object, do_unlink=True)
    return mesh


def _join_parts_onto_the_ground(parts: list[bpy.types.Object]) -> bpy.types.Object:
    """Merge primitives into one mesh whose origin sits at the ground point.

    Plan waypoints are ground positions with z = 0, so every proxy must have its origin
    at its feet. Joining leaves the merged origin wherever the active object's was, so
    it is moved to the world origin afterwards rather than assumed.
    """
    for part in parts:
        part.select_set(True)
    bpy.context.view_layer.objects.active = parts[0]
    bpy.ops.object.join()
    merged = bpy.context.active_object
    bpy.context.scene.cursor.location = (0.0, 0.0, 0.0)
    bpy.ops.object.origin_set(type="ORIGIN_CURSOR")
    return merged


def build_box_mesh(mesh_name: str, dimensions: list[float]) -> bpy.types.Mesh:
    """A plain box sitting on its own origin. Suitcases, signs, handheld objects."""
    length, width, height = dimensions
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, height / 2.0))
    box = bpy.context.active_object
    box.scale = (length, width, height)
    merged = _join_parts_onto_the_ground([box])
    bpy.ops.object.transform_apply(location=True, scale=True)
    return _detach_mesh(merged, mesh_name)


def build_cylinder_mesh(mesh_name: str, dimensions: list[float]) -> bpy.types.Mesh:
    """An upright cylinder. Bottles, cups, hydrants, planters."""
    length, width, height = dimensions
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=CYLINDER_SIDES, radius=0.5, depth=height,
        location=(0.0, 0.0, height / 2.0),
    )
    cylinder = bpy.context.active_object
    cylinder.scale = (length, width, 1.0)
    merged = _join_parts_onto_the_ground([cylinder])
    bpy.ops.object.transform_apply(location=True, scale=True)
    for polygon in merged.data.polygons:
        polygon.use_smooth = True
    return _detach_mesh(merged, mesh_name)


def build_vehicle_mesh(
    mesh_name: str, dimensions: list[float], actor: dict,
) -> bpy.types.Mesh:
    """A body with a raised mass above it, which reads as a vehicle where a box does not.

    The same shape serves animals: a low body with a smaller volume above the front is
    closer to a dog or a horse in silhouette than a rectangle is.
    """
    length, width, height = dimensions
    body_height = height * float(actor.get("body_height_ratio", 0.62))
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=(0.0, 0.0, body_height / 2.0))
    body = bpy.context.active_object
    body.scale = (length, width, body_height)
    parts = [body]
    cabin_height = height - body_height
    if cabin_height > MINIMUM_CABIN_HEIGHT_METERS:
        cabin_length = length * float(actor.get("cabin_length_ratio", 0.55))
        bpy.ops.mesh.primitive_cube_add(
            size=1.0,
            location=(0.0, (length - cabin_length) * CABIN_FORWARD_BIAS,
                      body_height + cabin_height / 2.0),
        )
        cabin = bpy.context.active_object
        cabin.scale = (
            cabin_length, width * float(actor.get("cabin_width_ratio", 0.92)), cabin_height,
        )
        parts.append(cabin)
    merged = _join_parts_onto_the_ground(parts)
    bpy.ops.object.transform_apply(location=True, scale=True)
    return _detach_mesh(merged, mesh_name)


def build_mesh_library() -> dict[str, bpy.types.Mesh]:
    """Everything built up front, which is only the articulated rig.

    Rigid proxies are built lazily on first use and then shared, because their shape
    depends on the class that turned up. A video of pedestrians never builds a bus, and
    a video with forty cars builds one car.
    """
    return {"person": humanoid.build_humanoid_mesh(skeleton_for_height(1.75))}


# --------------------------------------------------------------------------
# Camera, lighting, shadow catcher
# --------------------------------------------------------------------------

def build_camera(scene: bpy.types.Scene, camera_contract: dict) -> bpy.types.Object:
    camera_data = bpy.data.cameras.new("FOR3D_Camera")
    camera_data.lens = float(camera_contract.get("focal_length_mm", DEFAULT_FOCAL_LENGTH_MM))
    # Pinned horizontal rather than left on AUTO: the host derives focal length from a
    # horizontal field of view, and AUTO would silently fit the vertical axis instead on
    # portrait footage, changing the framing the crop rectangle was computed for.
    camera_data.sensor_fit = "HORIZONTAL"
    camera_data.sensor_width = float(
        camera_contract.get("sensor_width_mm", DEFAULT_SENSOR_WIDTH_MM)
    )
    camera_data.shift_x = 0.0
    camera_data.shift_y = 0.0
    camera = bpy.data.objects.new("FOR3D_Camera", camera_data)
    camera.location = tuple(camera_contract.get("position", DEFAULT_CAMERA_LOCATION))
    rotation_degrees = camera_contract.get("rotation_degrees", DEFAULT_CAMERA_ROTATION_DEGREES)
    camera.rotation_euler = tuple(math.radians(float(angle)) for angle in rotation_degrees)
    _ensure_collection(SHELL_COLLECTION_NAME).objects.link(camera)
    scene.camera = camera
    return camera


def build_key_light() -> bpy.types.Object:
    light_data = bpy.data.lights.new("FOR3D_Sun", type="SUN")
    light_data.energy = SUN_ENERGY
    light = bpy.data.objects.new("FOR3D_Sun", light_data)
    light.location = SUN_LOCATION
    light.rotation_euler = SUN_ROTATION_RADIANS
    _ensure_collection(SHELL_COLLECTION_NAME).objects.link(light)
    return light


def build_ambient_world(scene: bpy.types.Scene) -> bpy.types.World:
    """A dim grey world so unlit sides of actors are shaded, not black."""
    world = bpy.data.worlds.get("FOR3D_World") or bpy.data.worlds.new("FOR3D_World")
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    if background is not None:
        background.inputs["Color"].default_value = WORLD_AMBIENT_COLOR
        background.inputs["Strength"].default_value = WORLD_AMBIENT_STRENGTH
    scene.world = world
    return world


def build_shadow_catcher(engine: str) -> bpy.types.Object | None:
    """Invisible ground that receives contact shadows only.

    §5.1: nothing of the environment is rendered. The catcher exists so the shadow
    pass has a surface, and is holdout everywhere else. Returns None on engines that do
    not implement shadow catching, where the plane would render as opaque ground.
    """
    if engine not in SHADOW_CATCHER_ENGINES:
        return None
    bpy.ops.mesh.primitive_plane_add(size=SHADOW_CATCHER_SIZE_METERS, location=(0.0, 0.0, 0.0))
    catcher = bpy.context.active_object
    catcher.name = SHADOW_CATCHER_NAME
    if not getattr(catcher, "is_shadow_catcher", None):
        # The attribute is absent on a build without Cycles support; an opaque plane is
        # worse than no shadow, so it is removed rather than left in the scene.
        bpy.data.objects.remove(catcher, do_unlink=True)
        return None
    catcher.is_shadow_catcher = True
    for collection in list(catcher.users_collection):
        collection.objects.unlink(catcher)
    _ensure_collection(SHELL_COLLECTION_NAME).objects.link(catcher)
    return catcher


# --------------------------------------------------------------------------
# Shell assembly
# --------------------------------------------------------------------------

TEMPLATE_COLLECTION_NAME = "FOR3D_Templates"
_TEMPLATES: dict[str, bpy.types.Object] = {}


def template_collection() -> bpy.types.Collection:
    """Holds the prebuilt assets. Only copies of them are ever rendered.

    The templates stay in the view layer rather than being excluded from it, because an
    excluded collection is dropped from the depsgraph and a template that was never
    evaluated cannot be copied into a working rig.
    """
    return _ensure_collection(TEMPLATE_COLLECTION_NAME)


def _set_rendered(instance: bpy.types.Object, rendered: bool) -> None:
    """Show or hide an object and its children from the camera.

    `hide_viewport` is not enough and hiding the collection is not enough: neither
    affects rendering. Without `hide_render` every prebuilt asset — including a 10.5
    metre bus parked at the world origin — is drawn into every frame, which composites
    as a solid block over the plate.
    """
    instance.hide_render = not rendered
    for child in instance.children:
        child.hide_render = not rendered


def load_actor_library(library_path: str, asset_names: list[str]) -> dict:
    """Append prebuilt actor geometry instead of generating it.

    This is what takes model building off the render path. Geometry is authored once by
    `backend/tools/build_actor_library.py` and appended here as templates; every actor is then
    a copy that shares the appended mesh. A far heavier, more detailed model therefore
    costs the same per gap as a crude one — the cost moved to build time.

    Returns a summary rather than raising on failure: a library that cannot be appended
    is a reason to generate geometry the old way, not a reason to lose the job.
    """
    _TEMPLATES.clear()
    path = Path(library_path)
    if not path.is_file():
        return {"loaded": False, "reason": "missing", "assets": []}
    wanted = set(asset_names)
    try:
        with bpy.data.libraries.load(str(path), link=False) as (source, target):
            target.objects = [name for name in source.objects if name in wanted]
    except (OSError, RuntimeError) as error:
        return {"loaded": False, "reason": str(error), "assets": []}
    collection = template_collection()
    appended = []
    for name in wanted:
        candidate = bpy.data.objects.get(name)
        if candidate is None:
            continue
        if candidate.name not in collection.objects:
            for existing in list(candidate.users_collection):
                existing.objects.unlink(candidate)
            collection.objects.link(candidate)
        _set_rendered(candidate, False)
        _TEMPLATES[name] = candidate
        appended.append(name)
    return {"loaded": bool(appended), "reason": "ok", "assets": sorted(appended)}


def _copy_template(template: bpy.types.Object, name: str) -> bpy.types.Object:
    """Copy an appended template, sharing its data so nothing is duplicated.

    An armature object's pose belongs to the object, so each actor needs its own copy of
    the object while pointing at the same armature datablock. The skinned body is copied
    the same way and its modifier repointed, or every actor would be driven by the
    template's rig and they would all move identically.
    """
    collection = actor_collection()
    instance = template.copy()
    instance.name = name
    collection.objects.link(instance)
    for child in template.children:
        body = child.copy()
        body.name = f"{name}_body"
        collection.objects.link(body)
        body.parent = instance
        body.matrix_parent_inverse = child.matrix_parent_inverse.copy()
        for modifier in body.modifiers:
            if modifier.type == "ARMATURE":
                modifier.object = instance
    # The copy inherits the template's hidden state and must be shown again.
    _set_rendered(instance, True)
    bpy.context.view_layer.update()
    return instance


def select_engine(scene: bpy.types.Scene, requested: str) -> str:
    """Set the render engine, falling back to Cycles when the request is unavailable.

    Chosen before anything is built because the shell's contents depend on it — see
    `build_shadow_catcher`.
    """
    try:
        scene.render.engine = requested
    except (TypeError, ValueError):
        scene.render.engine = "CYCLES"
    return scene.render.engine


def build_shell(job_manifest: dict) -> dict:
    """Construct everything reused across every gap. Returns a summary for the host."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    engine = select_engine(
        scene, str(job_manifest.get("render", {}).get("engine", "BLENDER_EEVEE_NEXT")),
    )
    resolution = job_manifest.get("resolution", DEFAULT_RESOLUTION)
    scene.render.resolution_x = int(resolution[0])
    scene.render.resolution_y = int(resolution[1])
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = True
    _ensure_collection(SHELL_COLLECTION_NAME)
    actor_collection()
    build_camera(scene, job_manifest.get("camera", {}))
    build_key_light()
    build_ambient_world(scene)
    catcher = build_shadow_catcher(engine)
    material = build_shared_actor_material()
    actor_library = job_manifest.get("actor_library") or {}
    library_report = load_actor_library(
        str(actor_library.get("path", "")), list(actor_library.get("assets", [])),
    )
    # Only generate geometry for what the prebuilt library did not supply.
    library = {} if library_report["loaded"] else build_mesh_library()
    return {
        "shell_built": True,
        "engine": engine,
        "resolution": [scene.render.resolution_x, scene.render.resolution_y],
        "mesh_library": sorted(library),
        "actor_library": library_report,
        "shared_material": material.name,
        "shadow_catcher": catcher is not None,
        "census": datablock_census(),
    }


# --------------------------------------------------------------------------
# Per-gap actor instancing
# --------------------------------------------------------------------------

def _rigid_mesh_for_actor(actor: dict) -> bpy.types.Mesh:
    """A mesh for one non-articulated class, built on demand and then shared.

    Keyed by class name, so a video containing only people never builds a bus, and a
    video with forty cars builds one car.
    """
    class_name = str(actor.get("class_name", "unknown"))
    mesh_name = f"{PROXY_MESH_PREFIX}_{class_name.replace(' ', '_')}"
    existing = bpy.data.meshes.get(mesh_name)
    if existing is not None:
        return existing
    proxy = str(actor.get("proxy", "box"))
    dimensions = [float(value) for value in actor.get("dimensions", (0.5, 0.5, 0.6))]
    if proxy == "vehicle":
        return build_vehicle_mesh(mesh_name, dimensions, actor)
    if proxy == "cylinder":
        return build_cylinder_mesh(mesh_name, dimensions)
    return build_box_mesh(mesh_name, dimensions)


def instance_actor(actor: dict, material: bpy.types.Material) -> bpy.types.Object:
    """Create one actor sharing library data — no geometry is duplicated.

    Articulated classes get a rig object plus a skinned body; everything else is a single
    rigid object. Both return the object that carries the actor's world transform, so
    the caller does not have to care which it got.
    """
    collection = actor_collection()
    colour = actor.get("color", [0.5, 0.5, 0.5])
    template = _TEMPLATES.get(str(actor.get("asset_name", "")))
    if template is not None:
        instance = _copy_template(template, str(actor["id"]))
        instance.color = (float(colour[0]), float(colour[1]), float(colour[2]), 1.0)
        for child in instance.children:
            child.color = instance.color
        return instance
    if str(actor.get("proxy")) == "humanoid":
        rig_object, body = humanoid.instance_humanoid(
            str(actor["id"]),
            humanoid.build_humanoid_mesh(_actor_skeleton(actor)),
            humanoid.build_humanoid_armature(_actor_skeleton(actor)),
            material,
            collection,
        )
        humanoid.add_subdivision(body)
        body.color = (float(colour[0]), float(colour[1]), float(colour[2]), 1.0)
        rig_object.color = body.color
        return rig_object
    instance = bpy.data.objects.new(str(actor["id"]), _rigid_mesh_for_actor(actor))
    instance.color = (float(colour[0]), float(colour[1]), float(colour[2]), 1.0)
    if not instance.data.materials:
        instance.data.materials.append(material)
    collection.objects.link(instance)
    return instance


def _actor_skeleton(actor: dict):
    height = float(actor.get("dimensions", (0.5, 0.34, 1.75))[2])
    return skeleton_for_height(height)


def apply_actor_keyframes(instance: bpy.types.Object, keyframes: list[dict]) -> int:
    """Bind the validated path. Coordinates come from the storyboard, never from here.

    Frames are floats: the host maps source time onto the sparse render timeline and a
    waypoint rarely lands exactly on a sample. Blender accepts subframe keyframes, so
    the path is bound at its true time rather than quantised to the nearest render.
    """
    for keyframe in keyframes:
        frame = float(keyframe["frame"])
        instance.location = tuple(float(value) for value in keyframe["location"])
        instance.keyframe_insert(data_path="location", frame=frame)
        heading = keyframe.get("heading_degrees")
        if heading is not None:
            instance.rotation_euler = (0.0, 0.0, math.radians(float(heading)))
            instance.keyframe_insert(data_path="rotation_euler", frame=frame)
        pose = keyframe.get("pose")
        if pose is None:
            continue
        if instance.pose is None:
            # The host asked for an articulated actor and got rigid geometry. Rendering
            # it anyway would produce a figure that slides without moving its legs.
            raise ShellStateError(
                f"{instance.name} was sent a pose but carries no armature"
            )
        rotations = pose.get("rotations", {})
        humanoid.apply_pose(instance, rotations, pose.get("root_offset", (0.0, 0.0, 0.0)))
        humanoid.keyframe_pose(instance, frame, rotations)
    _set_linear_interpolation(instance)
    return len(keyframes)


def _set_linear_interpolation(instance: bpy.types.Object) -> None:
    """Move between waypoints at a constant rate.

    Blender's default Bezier handles ease in and out of every keyframe, which would add
    acceleration the evidence does not support — a figure that repeatedly slows to a
    halt at each predicted waypoint and speeds up between them.
    """
    for animation in (instance.animation_data, getattr(instance.data, "animation_data", None)):
        if animation is None or animation.action is None:
            continue
        for curve in animation.action.fcurves:
            for point in curve.keyframe_points:
                point.interpolation = "LINEAR"


def prepare_gap(gap_specification: dict) -> dict:
    """Instance and animate the actors for one gap against the existing shell."""
    material = build_shared_actor_material()
    instanced: list[str] = []
    for actor in gap_specification.get("actors", []):
        instance = instance_actor(actor, material)
        apply_actor_keyframes(instance, actor.get("keyframes", []))
        instanced.append(instance.name)
    return {
        "gap_index": gap_specification.get("gap_index"),
        "actor_ids": instanced,
        "census": datablock_census(),
    }


# --------------------------------------------------------------------------
# Reset and leak detection — §6.6
# --------------------------------------------------------------------------

def clear_actors() -> dict:
    """Remove every per-gap object and assert the collection really is empty.

    A silent state leak between gaps corrupts output, which is worse than crashing,
    so this asserts rather than trusting the removal loop.
    """
    collection = actor_collection()
    for instance in list(collection.objects):
        if instance.animation_data is not None:
            instance.animation_data_clear()
        bpy.data.objects.remove(instance, do_unlink=True)
    remaining = list(collection.objects)
    if remaining:
        raise ShellStateError(
            f"Actor collection still holds {len(remaining)} objects after reset: "
            f"{[item.name for item in remaining]}"
        )
    _purge_orphan_actions()
    return {"cleared": True, "census": datablock_census()}


def _purge_orphan_actions() -> None:
    """Actions outlive their objects and would otherwise accumulate across gaps."""
    for action in list(bpy.data.actions):
        if action.users == 0:
            bpy.data.actions.remove(action)


def datablock_census() -> dict:
    """Counts the host watches for monotonic growth to trigger a recycle (§6.6)."""
    return {
        "objects": len(bpy.data.objects),
        "meshes": len(bpy.data.meshes),
        "materials": len(bpy.data.materials),
        "actions": len(bpy.data.actions),
        "actors": len(actor_collection().objects),
    }


def load_json_contract(path: Path) -> dict:
    """Blender reads validated JSON contracts and nothing else (§3)."""
    import json

    with Path(path).open("r", encoding="utf-8") as contract_file:
        contract = json.load(contract_file)
    if not isinstance(contract, dict):
        raise ValueError(f"Contract at {path} must be a JSON object")
    return contract
