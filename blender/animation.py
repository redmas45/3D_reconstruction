import math

from mathutils import Vector


ANIMATION_SAMPLE_STEP = 2
MAXIMUM_WALK_SWING_RADIANS = 0.58
HUMAN_STRIDE_LENGTH_METERS = 1.35
MAXIMUM_STEERING_RADIANS = math.radians(32.0)
MINIMUM_DIRECTION_DISTANCE_METERS = 0.001


def animate_entity(
    parts: dict,
    entity: dict,
    frame_count: int,
    fps: float,
) -> None:
    world_points = [
        _grounded_position(Vector(item["world"]))
        for item in entity["path_prediction"]["waypoints"]
    ]
    sample_frames = _sample_frames(frame_count)
    positions = [
        catmull_rom_position(
            world_points, (frame_index - 1) / max(1, frame_count - 1),
        )
        for frame_index in sample_frames
    ]
    _animate_path(parts, entity, sample_frames, positions, fps)
    _animate_lifecycle(parts, entity, frame_count)


def catmull_rom_position(points: list[Vector], parameter: float) -> Vector:
    if len(points) < 3:
        raise ValueError("Catmull-Rom paths require at least three points")
    bounded_parameter = max(0.0, min(1.0, parameter))
    segment_count = len(points) - 1
    scaled_parameter = bounded_parameter * segment_count
    segment_index = min(segment_count - 1, int(scaled_parameter))
    local_parameter = scaled_parameter - segment_index
    p0 = points[max(0, segment_index - 1)]
    p1 = points[segment_index]
    p2 = points[segment_index + 1]
    p3 = points[min(len(points) - 1, segment_index + 2)]
    return _grounded_position(
        _catmull_segment(p0, p1, p2, p3, local_parameter),
    )


def _animate_path(
    parts: dict,
    entity: dict,
    sample_frames: list[int],
    positions: list[Vector],
    fps: float,
) -> None:
    root = parts["root"]
    distance_travelled = 0.0
    previous_heading: float | None = None
    for index, (frame_index, position) in enumerate(zip(sample_frames, positions)):
        previous_position = positions[max(0, index - 1)]
        movement = position - previous_position
        distance_travelled += movement.length
        heading = _bounded_heading(
            previous_heading, movement, entity, frame_index, sample_frames, fps,
        )
        root.location = position
        if heading is not None:
            root.rotation_euler[2] = heading
        _keyframe_root(root, frame_index)
        _animate_articulation(
            parts, entity, frame_index, distance_travelled, previous_heading, heading,
        )
        previous_heading = heading if heading is not None else previous_heading


def _bounded_heading(
    previous_heading: float | None,
    movement: Vector,
    entity: dict,
    frame_index: int,
    sample_frames: list[int],
    fps: float,
) -> float | None:
    if movement.length <= MINIMUM_DIRECTION_DISTANCE_METERS:
        return previous_heading
    requested_heading = math.atan2(-movement.x, movement.y)
    if previous_heading is None:
        return requested_heading
    previous_frame = sample_frames[max(0, sample_frames.index(frame_index) - 1)]
    elapsed_seconds = max(1.0 / fps, (frame_index - previous_frame) / fps)
    turn_rate = float(entity.get("kinematics", {}).get(
        "maximum_turn_rate_degrees_per_second", 45.0,
    ))
    maximum_delta = math.radians(turn_rate) * elapsed_seconds
    return previous_heading + _clamp(
        _angle_delta(previous_heading, requested_heading),
        -maximum_delta,
        maximum_delta,
    )


def _animate_articulation(
    parts: dict,
    entity: dict,
    frame_index: int,
    distance_travelled: float,
    previous_heading: float | None,
    heading: float | None,
) -> None:
    if entity["animation"]["state"] == "idle":
        return
    if parts.get("legs"):
        _animate_walk_cycle(parts, entity, frame_index, distance_travelled)
    _animate_wheels(parts, frame_index, distance_travelled, previous_heading, heading)


def _animate_walk_cycle(
    parts: dict,
    entity: dict,
    frame_index: int,
    distance_travelled: float,
) -> None:
    phase_offset = float(entity["animation"]["phase_offset"]) * math.tau
    phase = distance_travelled / HUMAN_STRIDE_LENGTH_METERS * math.tau + phase_offset
    swing = math.sin(phase) * MAXIMUM_WALK_SWING_RADIANS
    for index, leg in enumerate(parts["legs"]):
        leg.rotation_euler[0] = swing if index == 0 else -swing
        leg.keyframe_insert("rotation_euler", frame=frame_index)
    for index, arm in enumerate(parts["arms"]):
        arm.rotation_euler[0] = -swing * 0.72 if index == 0 else swing * 0.72
        arm.keyframe_insert("rotation_euler", frame=frame_index)


def _animate_wheels(
    parts: dict,
    frame_index: int,
    distance_travelled: float,
    previous_heading: float | None,
    heading: float | None,
) -> None:
    wheel_radius = parts.get("wheel_radius")
    if not parts.get("wheels") or not wheel_radius:
        return
    rotation = distance_travelled / float(wheel_radius)
    steering = _steering_angle(previous_heading, heading)
    steering_wheels = set(parts.get("steering_wheels", []))
    for wheel in parts["wheels"]:
        wheel.rotation_euler[0] = rotation
        wheel.rotation_euler[2] = steering if wheel in steering_wheels else 0.0
        wheel.keyframe_insert("rotation_euler", frame=frame_index)


def _steering_angle(
    previous_heading: float | None,
    heading: float | None,
) -> float:
    if previous_heading is None or heading is None:
        return 0.0
    return _clamp(
        _angle_delta(previous_heading, heading) * 2.0,
        -MAXIMUM_STEERING_RADIANS,
        MAXIMUM_STEERING_RADIANS,
    )


def _sample_frames(frame_count: int) -> list[int]:
    frames = list(range(1, frame_count + 1, ANIMATION_SAMPLE_STEP))
    if not frames or frames[-1] != frame_count:
        frames.append(frame_count)
    return frames


def _grounded_position(position: Vector) -> Vector:
    grounded = position.copy()
    grounded.z = max(0.0, float(grounded.z))
    return grounded


def _keyframe_root(root: object, frame_index: int) -> None:
    root.keyframe_insert("location", frame=frame_index)
    root.keyframe_insert("rotation_euler", frame=frame_index)


def _angle_delta(first: float, second: float) -> float:
    return (second - first + math.pi) % math.tau - math.pi


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _catmull_segment(
    p0: Vector,
    p1: Vector,
    p2: Vector,
    p3: Vector,
    parameter: float,
) -> Vector:
    parameter_squared = parameter * parameter
    parameter_cubed = parameter_squared * parameter
    return 0.5 * (
        (2.0 * p1)
        + (-p0 + p2) * parameter
        + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * parameter_squared
        + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * parameter_cubed
    )


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
    material: object,
    keyframes: list[tuple[int, float]],
) -> None:
    shader = material.node_tree.nodes.get("Principled BSDF") if material.use_nodes else None
    if shader is None:
        return
    if hasattr(material, "surface_render_method"):
        material.surface_render_method = "DITHERED"
    alpha_input = shader.inputs["Alpha"]
    base_alpha = float(alpha_input.default_value)
    for frame_index, opacity_factor in keyframes:
        alpha_input.default_value = base_alpha * opacity_factor
        alpha_input.keyframe_insert("default_value", frame=frame_index)
