import math

from mathutils import Vector


ANIMATION_SAMPLE_STEP = 1
MAXIMUM_WALK_SWING_RADIANS = 0.46
MAXIMUM_KNEE_BEND_RADIANS = 0.68
BASE_ELBOW_BEND_RADIANS = 0.16
ADDITIONAL_ELBOW_BEND_RADIANS = 0.24
FOOT_LEVEL_COMPENSATION = 0.88
HUMAN_BODY_BOB_METERS = 0.025
HUMAN_BODY_SWAY_RADIANS = 0.028
UPPER_LEG_LENGTH_METERS = 0.46
LOWER_LEG_TO_SOLE_METERS = 0.52
MAXIMUM_STANCE_COMPENSATION_METERS = 0.10
HUMAN_STRIDE_LENGTH_METERS = 1.35
MAXIMUM_STEERING_RADIANS = math.radians(32.0)
MINIMUM_DIRECTION_DISTANCE_METERS = 0.001
MAXIMUM_PATH_OVERSHOOT_METERS = 0.20
PRESENTATION_FADE_SECONDS = 0.18
EDGE_OPACITY_FACTOR = 0.12


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
    _set_linear_object_interpolation(parts)
    _animate_lifecycle(parts, entity, frame_count, fps)


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
    interpolated = _catmull_segment(p0, p1, p2, p3, local_parameter)
    return _bounded_path_position(interpolated, points)


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
        if heading is not None and not parts.get("lock_facing_camera", False):
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
    _animate_body_weight(parts, frame_index, phase)
    _animate_legs(parts, frame_index, phase)
    _animate_arms(parts, frame_index, phase)


def _animate_body_weight(parts: dict, frame_index: int, phase: float) -> None:
    body_rig = parts.get("rig")
    if body_rig is None:
        return
    height_scale = float(parts.get("height_scale", 1.0))
    body_rig.location.z = (
        abs(math.sin(phase)) * HUMAN_BODY_BOB_METERS * height_scale
    )
    body_rig.rotation_euler[1] = math.sin(phase) * HUMAN_BODY_SWAY_RADIANS
    body_rig.keyframe_insert("location", frame=frame_index)
    body_rig.keyframe_insert("rotation_euler", frame=frame_index)


def _animate_legs(parts: dict, frame_index: int, phase: float) -> None:
    knees = parts.get("knees", [])
    feet = parts.get("feet", [])
    base_heights = parts.get("leg_base_heights", [])
    height_scale = float(parts.get("height_scale", 1.0))
    for index, (leg, knee, foot) in enumerate(zip(parts["legs"], knees, feet)):
        leg_phase = phase + index * math.pi
        swing = math.sin(leg_phase) * MAXIMUM_WALK_SWING_RADIANS
        knee_bend = (
            max(0.0, math.sin(leg_phase + math.pi / 3.0))
            * MAXIMUM_KNEE_BEND_RADIANS
        )
        leg.rotation_euler[0] = swing
        knee.rotation_euler[0] = knee_bend
        foot.rotation_euler[0] = -(swing + knee_bend) * FOOT_LEVEL_COMPENSATION
        if index < len(base_heights):
            leg.location.z = (
                float(base_heights[index])
                - _stance_vertical_compensation(
                    swing,
                    knee_bend,
                    height_scale,
                )
            )
            leg.keyframe_insert("location", frame=frame_index)
        for control in (leg, knee, foot):
            control.keyframe_insert("rotation_euler", frame=frame_index)


def _stance_vertical_compensation(
    swing: float,
    knee_bend: float,
    height_scale: float,
) -> float:
    rest_reach = (
        UPPER_LEG_LENGTH_METERS + LOWER_LEG_TO_SOLE_METERS
    ) * height_scale
    animated_reach = (
        UPPER_LEG_LENGTH_METERS * height_scale * math.cos(swing)
        + LOWER_LEG_TO_SOLE_METERS * height_scale * math.cos(swing + knee_bend)
    )
    stance_weight = 1.0 - min(1.0, knee_bend / MAXIMUM_KNEE_BEND_RADIANS)
    shortening = max(0.0, rest_reach - animated_reach)
    return min(
        MAXIMUM_STANCE_COMPENSATION_METERS * height_scale,
        shortening * stance_weight,
    )


def _animate_arms(parts: dict, frame_index: int, phase: float) -> None:
    elbows = parts.get("elbows", [])
    for index, (arm, elbow) in enumerate(zip(parts["arms"], elbows)):
        arm_phase = phase + index * math.pi
        arm.rotation_euler[0] = (
            -math.sin(arm_phase) * MAXIMUM_WALK_SWING_RADIANS * 0.68
        )
        elbow.rotation_euler[0] = (
            BASE_ELBOW_BEND_RADIANS
            + max(0.0, -math.sin(arm_phase)) * ADDITIONAL_ELBOW_BEND_RADIANS
        )
        arm.keyframe_insert("rotation_euler", frame=frame_index)
        elbow.keyframe_insert("rotation_euler", frame=frame_index)


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
    spin_axis = int(parts.get("wheel_spin_axis", 0))
    for wheel in parts["wheels"]:
        wheel.rotation_euler[spin_axis] = -rotation
        if spin_axis != 2:
            wheel.rotation_euler[2] = steering if wheel in steering_wheels else 0.0
        wheel.keyframe_insert("rotation_euler", frame=frame_index)


def _set_linear_object_interpolation(parts: dict) -> None:
    controls = [parts.get("root"), parts.get("rig")]
    for key in ("arms", "elbows", "legs", "knees", "feet", "wheels"):
        controls.extend(parts.get(key, []))
    for control in (item for item in controls if item is not None):
        action = getattr(getattr(control, "animation_data", None), "action", None)
        for curve in getattr(action, "fcurves", []):
            for keyframe in curve.keyframe_points:
                keyframe.interpolation = "LINEAR"


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


def _bounded_path_position(position: Vector, points: list[Vector]) -> Vector:
    bounded = position.copy()
    for axis in (0, 1):
        lower = min(point[axis] for point in points) - MAXIMUM_PATH_OVERSHOOT_METERS
        upper = max(point[axis] for point in points) + MAXIMUM_PATH_OVERSHOOT_METERS
        bounded[axis] = max(lower, min(upper, float(bounded[axis])))
    return _grounded_position(bounded)


def _animate_lifecycle(
    parts: dict,
    entity: dict,
    frame_count: int,
    fps: float,
) -> None:
    lifecycle = entity["lifecycle"]
    transition_frame = max(2, round(frame_count * 0.22))
    edge_frame = min(
        max(2, round(PRESENTATION_FADE_SECONDS * fps)),
        max(2, frame_count // 3),
    )
    if lifecycle == "continuous":
        keyframes = [(1, 1.0), (frame_count, 1.0)]
    elif lifecycle == "enters":
        keyframes = [
            (1, 0.0), (transition_frame, 1.0), (frame_count, 1.0),
        ]
    elif lifecycle == "uncertain":
        keyframes = [
            (1, EDGE_OPACITY_FACTOR), (edge_frame, 0.70), (frame_count, 0.20),
        ]
    else:
        keyframes = [
            (1, 1.0),
            (frame_count - transition_frame, 1.0),
            (frame_count, 0.0),
        ]
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
