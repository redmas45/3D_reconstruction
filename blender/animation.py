import math

from mathutils import Vector


ANIMATION_SAMPLE_STEP = 2
MAXIMUM_WALK_SWING_RADIANS = 0.58


def animate_entity(parts: dict, entity: dict, frame_count: int) -> None:
    root = parts["root"]
    world_points = [Vector(item["world"]) for item in entity["path_prediction"]["waypoints"]]
    previous_point = world_points[0]
    for local_frame in range(1, frame_count + 1, ANIMATION_SAMPLE_STEP):
        parameter = (local_frame - 1) / max(1, frame_count - 1)
        position = catmull_rom_position(world_points, parameter)
        root.location = position
        direction = position - previous_point
        if direction.length > 0.001:
            root.rotation_euler[2] = math.atan2(-direction.x, direction.y)
        root.keyframe_insert("location", frame=local_frame)
        root.keyframe_insert("rotation_euler", frame=local_frame)
        _animate_articulation(parts, entity, local_frame)
        previous_point = position
    if (frame_count - 1) % ANIMATION_SAMPLE_STEP:
        _keyframe_final_pose(parts, entity, frame_count, world_points)
    _animate_lifecycle(parts, entity, frame_count)


def catmull_rom_position(points: list[Vector], parameter: float) -> Vector:
    p0, p1, p2 = points[0], points[1], points[2]
    if parameter <= 0.5:
        return _catmull_segment(p0, p0, p1, p2, parameter * 2.0)
    return _catmull_segment(p0, p1, p2, p2, (parameter - 0.5) * 2.0)


def _catmull_segment(p0: Vector, p1: Vector, p2: Vector, p3: Vector, parameter: float) -> Vector:
    parameter_squared = parameter * parameter
    parameter_cubed = parameter_squared * parameter
    return 0.5 * (
        (2.0 * p1)
        + (-p0 + p2) * parameter
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * parameter_squared
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * parameter_cubed
    )


def _animate_articulation(parts: dict, entity: dict, frame: int) -> None:
    if entity["animation"]["state"] == "idle":
        return
    speed = max(0.4, entity["animation"]["speed_meters_per_second"])
    phase = entity["animation"]["phase_offset"] * math.tau
    swing = math.sin(frame * 0.16 * speed + phase) * MAXIMUM_WALK_SWING_RADIANS
    if parts["legs"]:
        _animate_walk_cycle(parts, frame, swing)
    for wheel in parts["wheels"]:
        wheel.rotation_euler[1] = frame * speed * 0.32
        wheel.keyframe_insert("rotation_euler", frame=frame)


def _animate_walk_cycle(parts: dict, frame: int, swing: float) -> None:
    for index, leg in enumerate(parts["legs"]):
        leg.rotation_euler[0] = swing if index == 0 else -swing
        leg.keyframe_insert("rotation_euler", frame=frame)
    for index, arm in enumerate(parts["arms"]):
        arm.rotation_euler[0] = -swing * 0.72 if index == 0 else swing * 0.72
        arm.keyframe_insert("rotation_euler", frame=frame)


def _keyframe_final_pose(
    parts: dict, entity: dict, frame_count: int, world_points: list[Vector],
) -> None:
    root = parts["root"]
    root.location = world_points[-1]
    root.keyframe_insert("location", frame=frame_count)
    _animate_articulation(parts, entity, frame_count)


def _animate_lifecycle(parts: dict, entity: dict, frame_count: int) -> None:
    lifecycle = entity["lifecycle"]
    if lifecycle == "continuous":
        return
    transition_frame = max(2, round(frame_count * 0.22))
    if lifecycle == "enters":
        keyframes = [(1, 0.0), (transition_frame, 1.0)]
    elif lifecycle == "uncertain":
        keyframes = [(1, 1.0), (frame_count, 0.35)]
    else:
        keyframes = [(frame_count - transition_frame, 1.0), (frame_count, 0.0)]
    for material in parts.get("materials", []):
        _keyframe_material_alpha(material, keyframes)


def _keyframe_material_alpha(
    material: object, keyframes: list[tuple[int, float]],
) -> None:
    shader = material.node_tree.nodes.get("Principled BSDF") if material.use_nodes else None
    if shader is None:
        return
    if hasattr(material, "surface_render_method"):
        material.surface_render_method = "DITHERED"
    alpha_input = shader.inputs["Alpha"]
    base_alpha = float(alpha_input.default_value)
    for frame, opacity_factor in keyframes:
        alpha_input.default_value = base_alpha * opacity_factor
        alpha_input.keyframe_insert("default_value", frame=frame)
