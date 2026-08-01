import math


CAMERA_SENSOR_WIDTH_MILLIMETERS = 36.0
GROUND_IMAGE_MARGIN = 0.01
MINIMUM_GROUND_RAY_DOWNWARD_COMPONENT = 0.01
CAMERA_LOOK_DISTANCE_METERS = 10.0

# Below this forward depth perspective division is numerically meaningless, so a point
# is reported off-screen rather than projected to a wild coordinate.
MINIMUM_PROJECTION_DEPTH_METERS = 0.05


def camera_pose(
    camera_height_meters: float,
    horizon_normalized_y: float,
    horizontal_field_of_view_degrees: float,
    frame_width: int,
    frame_height: int,
) -> dict:
    horizontal_tangent = math.tan(
        math.radians(horizontal_field_of_view_degrees) / 2.0,
    )
    vertical_tangent = horizontal_tangent * frame_height / frame_width
    horizon_axis = (0.5 - horizon_normalized_y) * 2.0
    downward_pitch = math.atan(horizon_axis * vertical_tangent)
    forward = [0.0, math.cos(downward_pitch), -math.sin(downward_pitch)]
    focal_length = CAMERA_SENSOR_WIDTH_MILLIMETERS / (2.0 * horizontal_tangent)
    return {
        "focal_length_mm": round(focal_length, 4),
        "position": [0.0, 0.0, round(camera_height_meters, 4)],
        "look_at": [
            0.0,
            round(forward[1] * CAMERA_LOOK_DISTANCE_METERS, 4),
            round(
                camera_height_meters + forward[2] * CAMERA_LOOK_DISTANCE_METERS,
                4,
            ),
        ],
    }


def image_point_to_world(
    image_x: float,
    image_y: float,
    frame_width: int,
    frame_height: int,
    camera_contract: dict,
) -> list[float]:
    if camera_contract.get("projection_model") != "pinhole_ground_plane_v2":
        return _legacy_image_point_to_world(
            image_x, image_y, frame_width, frame_height, camera_contract,
        )
    mapping = camera_contract["ground_mapping"]
    bounded_x = _bounded_image_coordinate(image_x, frame_width)
    bounded_y = _bounded_ground_image_y(image_y, frame_height, mapping)
    ray = _ground_ray(
        bounded_x, bounded_y, frame_width, frame_height, camera_contract,
    )
    camera_height = float(camera_contract["position"][2])
    distance = camera_height / max(
        MINIMUM_GROUND_RAY_DOWNWARD_COMPONENT, -ray[2],
    )
    return [
        round(ray[0] * distance, 4),
        round(ray[1] * distance, 4),
        0.0,
    ]


def camera_basis(
    frame_width: int,
    frame_height: int,
    camera_contract: dict,
) -> dict:
    """Orthonormal camera axes plus the sensor tangents, shared by both directions.

    `_ground_ray` builds a ray from this basis and `world_point_to_image` inverts it.
    Deriving both from one function is what keeps the projection and its inverse from
    drifting apart, which would put composited actors in the wrong place (§5.2).
    """
    horizontal_tangent = math.tan(
        math.radians(float(camera_contract["field_of_view_degrees"])) / 2.0,
    )
    vertical_tangent = horizontal_tangent * frame_height / frame_width
    horizon_axis = (0.5 - float(camera_contract["horizon_normalized_y"])) * 2.0
    downward_pitch = math.atan(horizon_axis * vertical_tangent)
    return {
        "right": (1.0, 0.0, 0.0),
        "forward": (0.0, math.cos(downward_pitch), -math.sin(downward_pitch)),
        "upward": (0.0, math.sin(downward_pitch), math.cos(downward_pitch)),
        "horizontal_tangent": horizontal_tangent,
        "vertical_tangent": vertical_tangent,
    }


def world_point_to_image(
    world_point: list[float] | tuple[float, float, float],
    frame_width: int,
    frame_height: int,
    camera_contract: dict,
) -> tuple[float, float] | None:
    """Project a world point to pixel coordinates.

    Returns None when the point is at or behind the camera plane, where perspective
    division is undefined — callers must treat that as "not on screen" rather than
    clamping it to an edge, which would drag the render region across the frame.
    """
    basis = camera_basis(frame_width, frame_height, camera_contract)
    camera_position = [float(value) for value in camera_contract["position"]]
    offset = tuple(float(world_point[index]) - camera_position[index] for index in range(3))
    depth = _dot(offset, basis["forward"])
    if depth <= MINIMUM_PROJECTION_DEPTH_METERS:
        return None
    normalized_x = _dot(offset, basis["right"]) / (depth * basis["horizontal_tangent"])
    normalized_y = _dot(offset, basis["upward"]) / (depth * basis["vertical_tangent"])
    return (
        (normalized_x / 2.0 + 0.5) * frame_width,
        (0.5 - normalized_y / 2.0) * frame_height,
    )


def supports_projection(camera_contract: dict) -> bool:
    """Whether `world_point_to_image` can be evaluated for this contract.

    The legacy ground mapping is depth-table based and has no pinhole parameters, so
    world points cannot be projected forward from it. Callers use this to decide
    whether actor placement is possible at all rather than discovering it as a KeyError
    halfway through a render.
    """
    if camera_contract.get("projection_model") != "pinhole_ground_plane_v2":
        return False
    required = ("field_of_view_degrees", "horizon_normalized_y", "position")
    if any(camera_contract.get(name) is None for name in required):
        return False
    position = camera_contract["position"]
    return isinstance(position, (list, tuple)) and len(position) >= 3


def blender_camera_parameters(
    frame_width: int,
    frame_height: int,
    camera_contract: dict,
) -> dict:
    """Blender camera settings equivalent to this contract's projection.

    This is the hinge of the whole actor path. `world_point_to_image` decides where an
    actor's crop rectangle goes; Blender decides where the actor is actually drawn. If
    the two cameras disagree by even a couple of degrees the actor lands outside its own
    crop and the composite is empty. So both are derived from `camera_basis` here rather
    than being configured independently.

    Blender's camera looks down its local -Z with local +Y up. Rotating by `rx` about X
    sends -Z to `(0, sin rx, -cos rx)`; matching that to the basis forward vector
    `(0, cos p, -sin p)` gives `rx = 90 deg - p`, and the same rotation carries local +Y
    onto the basis up vector, so the roll is fixed too.
    """
    basis = camera_basis(frame_width, frame_height, camera_contract)
    downward_pitch = math.atan2(-basis["forward"][2], basis["forward"][1])
    return {
        "focal_length_mm": round(
            CAMERA_SENSOR_WIDTH_MILLIMETERS / (2.0 * basis["horizontal_tangent"]), 6,
        ),
        "sensor_width_mm": CAMERA_SENSOR_WIDTH_MILLIMETERS,
        "position": [float(value) for value in camera_contract["position"][:3]],
        "rotation_degrees": [
            round(90.0 - math.degrees(downward_pitch), 6), 0.0, 0.0,
        ],
    }


def _dot(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return sum(left[index] * right[index] for index in range(3))


def _ground_ray(
    image_x: float,
    image_y: float,
    frame_width: int,
    frame_height: int,
    camera_contract: dict,
) -> tuple[float, float, float]:
    horizontal_tangent = math.tan(
        math.radians(float(camera_contract["field_of_view_degrees"])) / 2.0,
    )
    vertical_tangent = horizontal_tangent * frame_height / frame_width
    horizon_axis = (0.5 - float(camera_contract["horizon_normalized_y"])) * 2.0
    downward_pitch = math.atan(horizon_axis * vertical_tangent)
    normalized_x = (image_x / frame_width - 0.5) * 2.0
    normalized_y = (0.5 - image_y / frame_height) * 2.0
    forward = (0.0, math.cos(downward_pitch), -math.sin(downward_pitch))
    upward = (0.0, math.sin(downward_pitch), math.cos(downward_pitch))
    return (
        normalized_x * horizontal_tangent,
        forward[1] + upward[1] * normalized_y * vertical_tangent,
        forward[2] + upward[2] * normalized_y * vertical_tangent,
    )


def _legacy_image_point_to_world(
    image_x: float,
    image_y: float,
    frame_width: int,
    frame_height: int,
    camera_contract: dict,
) -> list[float]:
    mapping = camera_contract["ground_mapping"]
    normalized_x = (image_x / frame_width) - 0.5
    normalized_y = image_y / frame_height
    denominator = max(0.01, mapping["near_y"] - mapping["far_y"])
    depth_ratio = _clamp((mapping["near_y"] - normalized_y) / denominator)
    depth = mapping["near_depth_meters"] + depth_ratio * (
        mapping["far_depth_meters"] - mapping["near_depth_meters"]
    )
    horizontal_span = 5.5 + depth * 0.42
    return [round(normalized_x * horizontal_span, 4), round(depth, 4), 0.0]


def _bounded_image_coordinate(value: float, dimension: int) -> float:
    margin = dimension * GROUND_IMAGE_MARGIN
    return max(margin, min(dimension - margin, float(value)))


def _bounded_ground_image_y(
    image_y: float,
    frame_height: int,
    mapping: dict,
) -> float:
    lower = float(mapping["far_y"]) * frame_height
    upper = float(mapping["near_y"]) * frame_height
    return max(lower, min(upper, float(image_y)))


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
