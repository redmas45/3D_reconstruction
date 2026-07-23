import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector


MINIMUM_PROJECTED_HEIGHT_FRACTION = 0.005
MINIMUM_TARGET_HEIGHT_FRACTION = 0.012
MAXIMUM_HUMAN_HEIGHT_FRACTION = 0.34
MAXIMUM_VEHICLE_HEIGHT_FRACTION = 0.30
MINIMUM_ENTITY_SCALE = 0.35
MAXIMUM_ENTITY_SCALE = 1.80
HUMAN_KINDS = {"person"}


def align_entity_scale(
    parts: dict,
    entity: dict,
    scene: bpy.types.Scene,
    camera: bpy.types.Object,
) -> float:
    target_height = _target_height_fraction(entity)
    projected_height = _projected_height_fraction(parts, entity, scene, camera)
    scale = bounded_visual_scale(target_height, projected_height)
    parts["root"].scale = (scale, scale, scale)
    return scale


def bounded_visual_scale(target_height: float, projected_height: float) -> float:
    if projected_height < MINIMUM_PROJECTED_HEIGHT_FRACTION:
        return 1.0
    requested_scale = target_height / projected_height
    return max(MINIMUM_ENTITY_SCALE, min(MAXIMUM_ENTITY_SCALE, requested_scale))


def _target_height_fraction(entity: dict) -> float:
    observed_height = float(entity.get("visual_anchor", {}).get("height_fraction", 0.0))
    maximum_height = (
        MAXIMUM_HUMAN_HEIGHT_FRACTION
        if entity.get("kind") in HUMAN_KINDS else MAXIMUM_VEHICLE_HEIGHT_FRACTION
    )
    return max(MINIMUM_TARGET_HEIGHT_FRACTION, min(maximum_height, observed_height))


def _projected_height_fraction(
    parts: dict,
    entity: dict,
    scene: bpy.types.Scene,
    camera: bpy.types.Object,
) -> float:
    first_world = entity["path_prediction"]["waypoints"][0]["world"]
    ground = Vector((float(first_world[0]), float(first_world[1]), 0.0))
    height = max(0.1, float(parts["visual_height_meters"]))
    top = ground + Vector((0.0, 0.0, height))
    ground_projection = world_to_camera_view(scene, camera, ground)
    top_projection = world_to_camera_view(scene, camera, top)
    return abs(float(top_projection.y) - float(ground_projection.y))
