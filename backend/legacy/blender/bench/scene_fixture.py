"""Representative benchmark scene: the entity classes v3 actually renders.

Two people and one vehicle, framed so their union bounding box covers roughly the
share of frame a real three-entity gap does. Kept separate from the measurement
loop in `m0_probe` so each file stays inside the `rules.md` size budget, and so the
ROI projection here can be lifted into production for M2 unchanged.

Imports `bpy`; only ever executed inside Blender.
"""

from pathlib import Path

import bpy


FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# Actors are placed to occupy roughly the union screen area a real 3-entity gap
# does (Implementation_plan.md §4 assumes ~35%), so ROI timings reflect production framing.
ROI_PADDING_NORMALIZED = 0.04
ROI_FULL_FRAME_AREA_THRESHOLD = 0.60

CAMERA_FOCAL_LENGTH_MM = 35.0
CAMERA_LOCATION = (0.0, -12.0, 4.5)
CAMERA_PITCH_RADIANS = 1.187

PERSON_HEIGHT_METERS = 1.75
PERSON_RADIUS_METERS = 0.22
VEHICLE_DIMENSIONS_METERS = (4.3, 1.8, 1.45)
ACTOR_STEP_METERS_PER_FRAME = 0.18

MOTION_LIBRARY_RELATIVE_PATH = Path("assets/animation/humanoid_motion_library.blend")
MOTION_LIBRARY_COLLECTION = "FOR3D_Humanoid"


def reset_scene() -> bpy.types.Scene:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.resolution_x = FRAME_WIDTH
    scene.render.resolution_y = FRAME_HEIGHT
    scene.render.resolution_percentage = 100
    return scene


def build_camera(scene: bpy.types.Scene) -> bpy.types.Object:
    """A pinhole camera at roughly the height and pitch of street CCTV."""
    camera_data = bpy.data.cameras.new("BenchCamera")
    camera_data.lens = CAMERA_FOCAL_LENGTH_MM
    camera = bpy.data.objects.new("BenchCamera", camera_data)
    camera.location = CAMERA_LOCATION
    camera.rotation_euler = (CAMERA_PITCH_RADIANS, 0.0, 0.0)
    scene.collection.objects.link(camera)
    scene.camera = camera
    return camera


def build_key_light(scene: bpy.types.Scene) -> None:
    light_data = bpy.data.lights.new("BenchSun", type="SUN")
    light_data.energy = 3.0
    light = bpy.data.objects.new("BenchSun", light_data)
    light.location = (4.0, -6.0, 9.0)
    light.rotation_euler = (0.6, 0.2, 0.3)
    scene.collection.objects.link(light)


def build_shared_actor_material() -> bpy.types.Material:
    """One material for every actor — §6.4's shared-shader strategy, measured here."""
    material = bpy.data.materials.new("BenchActorShared")
    material.use_nodes = True
    principled = material.node_tree.nodes.get("Principled BSDF")
    if principled is not None:
        principled.inputs["Base Color"].default_value = (0.25, 0.28, 0.35, 1.0)
        principled.inputs["Roughness"].default_value = 0.7
    return material


def _person_mesh() -> bpy.types.Mesh:
    """Fallback humanoid when the production motion library is unavailable."""
    bpy.ops.mesh.primitive_cylinder_add(
        vertices=16, radius=PERSON_RADIUS_METERS, depth=PERSON_HEIGHT_METERS,
    )
    body = bpy.context.active_object
    mesh = body.data
    mesh.name = "BenchPersonMesh"
    bpy.data.objects.remove(body, do_unlink=True)
    return mesh


def _vehicle_mesh() -> bpy.types.Mesh:
    bpy.ops.mesh.primitive_cube_add(size=1.0)
    vehicle = bpy.context.active_object
    vehicle.scale = VEHICLE_DIMENSIONS_METERS
    bpy.ops.object.transform_apply(scale=True)
    mesh = vehicle.data
    mesh.name = "BenchVehicleMesh"
    bpy.data.objects.remove(vehicle, do_unlink=True)
    return mesh


def _link_primitive(
    scene: bpy.types.Scene,
    name: str,
    mesh: bpy.types.Mesh,
    location: tuple[float, float, float],
    material: bpy.types.Material,
) -> bpy.types.Object:
    actor = bpy.data.objects.new(name, mesh)
    actor.location = location
    actor.data.materials.append(material)
    scene.collection.objects.link(actor)
    return actor


def append_motion_library_actor(project_root: Path) -> bpy.types.Object | None:
    """Use the real rigged humanoid when present so timings reflect real geometry."""
    library_path = project_root / MOTION_LIBRARY_RELATIVE_PATH
    if not library_path.is_file():
        return None
    try:
        with bpy.data.libraries.load(str(library_path), link=False) as (source, target):
            if MOTION_LIBRARY_COLLECTION not in source.collections:
                return None
            target.collections = [MOTION_LIBRARY_COLLECTION]
    except (OSError, RuntimeError):
        return None
    collection = bpy.data.collections.get(MOTION_LIBRARY_COLLECTION)
    if collection is None:
        return None
    instance = bpy.data.objects.new(f"{MOTION_LIBRARY_COLLECTION}_instance", None)
    instance.instance_type = "COLLECTION"
    instance.instance_collection = collection
    bpy.context.scene.collection.objects.link(instance)
    return instance


def build_actors(scene: bpy.types.Scene, project_root: Path) -> list[bpy.types.Object]:
    """Two people and one vehicle — the §2 supported entity classes."""
    material = build_shared_actor_material()
    actors: list[bpy.types.Object] = []
    library_actor = append_motion_library_actor(project_root)
    person_mesh = _person_mesh()
    person_positions = [(-2.4, 2.0, 0.875), (1.1, 4.5, 0.875)]
    if library_actor is not None:
        library_actor.location = person_positions[0]
        actors.append(library_actor)
        person_positions = person_positions[1:]
    for index, position in enumerate(person_positions):
        actors.append(_link_primitive(scene, f"BenchPerson{index}", person_mesh, position, material))
    actors.append(_link_primitive(scene, "BenchVehicle", _vehicle_mesh(), (4.2, 8.0, 0.72), material))
    return actors


def build_environment(scene: bpy.types.Scene) -> None:
    """Ground plane plus lit world — what v2 rendered and v3 deliberately does not."""
    bpy.ops.mesh.primitive_plane_add(size=80.0, location=(0.0, 6.0, 0.0))
    ground = bpy.context.active_object
    ground.name = "BenchGround"
    ground_material = bpy.data.materials.new("BenchGround")
    ground_material.use_nodes = True
    ground.data.materials.append(ground_material)
    world = bpy.data.worlds.new("BenchWorld")
    world.use_nodes = True
    scene.world = world
    scene.render.film_transparent = False


def animate_actors(actors: list[bpy.types.Object], frame_count: int) -> None:
    """Move actors so successive frames are genuinely different renders."""
    for index, actor in enumerate(actors):
        start_x = actor.location.x
        direction = 1 if index % 2 == 0 else -1
        for frame in range(1, frame_count + 1):
            actor.location.x = start_x + (frame - 1) * ACTOR_STEP_METERS_PER_FRAME * direction
            actor.keyframe_insert(data_path="location", frame=frame)


# --------------------------------------------------------------------------
# ROI projection — the same algorithm M2 uses in production
# --------------------------------------------------------------------------

def _as_vector(corner: object) -> object:
    from mathutils import Vector

    return Vector((corner[0], corner[1], corner[2]))


def _actor_camera_coordinates(
    scene: bpy.types.Scene,
    camera: bpy.types.Object,
    actor: bpy.types.Object,
    depsgraph: bpy.types.Depsgraph,
) -> list[tuple[float, float]]:
    from bpy_extras.object_utils import world_to_camera_view

    evaluated = actor.evaluated_get(depsgraph)
    matrix = evaluated.matrix_world
    corners = getattr(evaluated, "bound_box", None)
    if not corners:
        projected = world_to_camera_view(scene, camera, matrix.translation)
        return [(projected.x, projected.y)]
    points = []
    for corner in corners:
        projected = world_to_camera_view(scene, camera, matrix @ _as_vector(corner))
        points.append((projected.x, projected.y))
    return points


def projected_actor_region(
    scene: bpy.types.Scene,
    camera: bpy.types.Object,
    actors: list[bpy.types.Object],
) -> tuple[float, float, float, float] | None:
    """Union of actor bounding boxes in normalized camera space, padded."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    coordinates: list[tuple[float, float]] = []
    for actor in actors:
        coordinates.extend(_actor_camera_coordinates(scene, camera, actor, depsgraph))
    if not coordinates:
        return None
    minimum_x = max(0.0, min(point[0] for point in coordinates) - ROI_PADDING_NORMALIZED)
    maximum_x = min(1.0, max(point[0] for point in coordinates) + ROI_PADDING_NORMALIZED)
    minimum_y = max(0.0, min(point[1] for point in coordinates) - ROI_PADDING_NORMALIZED)
    maximum_y = min(1.0, max(point[1] for point in coordinates) + ROI_PADDING_NORMALIZED)
    if maximum_x <= minimum_x or maximum_y <= minimum_y:
        return None
    return (minimum_x, minimum_y, maximum_x, maximum_y)


def region_area_fraction(region: tuple[float, float, float, float]) -> float:
    return (region[2] - region[0]) * (region[3] - region[1])


def apply_render_region(
    scene: bpy.types.Scene,
    region: tuple[float, float, float, float] | None,
) -> None:
    if region is None:
        scene.render.use_border = False
        scene.render.use_crop_to_border = False
        return
    scene.render.use_border = True
    scene.render.use_crop_to_border = True
    scene.render.border_min_x, scene.render.border_min_y = region[0], region[1]
    scene.render.border_max_x, scene.render.border_max_y = region[2], region[3]
