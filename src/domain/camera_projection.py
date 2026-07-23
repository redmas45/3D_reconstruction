import math


CAMERA_SENSOR_WIDTH_MILLIMETERS = 36.0
GROUND_IMAGE_MARGIN = 0.01
MINIMUM_GROUND_RAY_DOWNWARD_COMPONENT = 0.01
CAMERA_LOOK_DISTANCE_METERS = 10.0


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
