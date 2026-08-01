"""The actor path's load-bearing assumption, checked independently.

`world_point_to_image` decides where an actor's crop rectangle goes. Blender decides
where the actor is drawn. If the two cameras disagree the actor lands outside its own
crop and the composite is empty — a failure that produces plausible-looking blank output
rather than an error.

So these tests re-implement Blender's own projection from first principles — Euler
rotation, sensor size, perspective divide — and assert it lands on the same pixel. That
is a real check: it would fail if the rotation convention, the sensor fit, or the focal
length derivation were wrong in either direction.
"""

import math
import sys
from pathlib import Path

import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[4] / "backend"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from domain.camera_projection import (
    blender_camera_parameters,
    supports_projection,
    world_point_to_image,
)

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# The manifest rounds focal length and rotation to six decimals so the JSON stays
# readable, which costs a few millionths of a pixel. A thousandth of a pixel is still
# orders of magnitude tighter than any real convention error — a flipped axis or a
# vertical sensor fit lands tens to hundreds of pixels away.
PIXEL_TOLERANCE = 1e-3


def _contract(field_of_view=54.0, horizon=0.42, height=3.0):
    return {
        "projection_model": "pinhole_ground_plane_v2",
        "field_of_view_degrees": field_of_view,
        "horizon_normalized_y": horizon,
        "position": [0.0, 0.0, height],
        "ground_mapping": {"near_y": 0.98, "far_y": 0.44},
    }


def blender_projection(
    world_point, frame_width, frame_height, camera,
):
    """Where Blender puts this point, derived from its documented camera model.

    Deliberately independent of `camera_basis`: rotation is applied as a raw Euler-X
    matrix and the sensor mapping is written out longhand, so agreement means the two
    models really match rather than sharing a bug.
    """
    rotation_x = math.radians(camera["rotation_degrees"][0])
    position = camera["position"]
    offset = [float(world_point[axis]) - float(position[axis]) for axis in range(3)]
    # Inverse of a rotation about X is a rotation by -rx.
    camera_x = offset[0]
    camera_y = offset[1] * math.cos(rotation_x) + offset[2] * math.sin(rotation_x)
    camera_z = -offset[1] * math.sin(rotation_x) + offset[2] * math.cos(rotation_x)
    depth = -camera_z  # Blender cameras look down local -Z.
    if depth <= 0:
        return None
    sensor_half_width = camera["sensor_width_mm"] / 2.0
    focal = camera["focal_length_mm"]
    horizontal = (focal * camera_x / depth) / sensor_half_width
    # sensor_fit HORIZONTAL: the vertical extent follows the frame aspect.
    vertical = (focal * camera_y / depth) / (sensor_half_width * frame_height / frame_width)
    return (
        (horizontal + 1.0) / 2.0 * frame_width,
        (1.0 - vertical) / 2.0 * frame_height,
    )


WORLD_POINTS = [
    (0.0, 12.0, 0.0),
    (0.0, 12.0, 1.75),
    (-3.5, 25.0, 0.9),
    (4.2, 8.0, 0.0),
    (1.0, 40.0, 2.0),
    (-8.0, 60.0, 3.2),
]


class TestBlenderAgreesWithTheProjection:
    @pytest.mark.parametrize("world_point", WORLD_POINTS)
    def test_same_pixel_as_the_region_projection(self, world_point):
        contract = _contract()
        camera = blender_camera_parameters(FRAME_WIDTH, FRAME_HEIGHT, contract)
        expected = world_point_to_image(world_point, FRAME_WIDTH, FRAME_HEIGHT, contract)
        assert expected is not None
        assert blender_projection(
            world_point, FRAME_WIDTH, FRAME_HEIGHT, camera,
        ) == pytest.approx(expected, abs=PIXEL_TOLERANCE)

    @pytest.mark.parametrize("field_of_view", [35.0, 54.0, 78.0, 100.0])
    def test_agreement_holds_across_fields_of_view(self, field_of_view):
        contract = _contract(field_of_view=field_of_view)
        camera = blender_camera_parameters(FRAME_WIDTH, FRAME_HEIGHT, contract)
        expected = world_point_to_image((2.0, 18.0, 1.0), FRAME_WIDTH, FRAME_HEIGHT, contract)
        assert blender_projection(
            (2.0, 18.0, 1.0), FRAME_WIDTH, FRAME_HEIGHT, camera,
        ) == pytest.approx(expected, abs=PIXEL_TOLERANCE)

    @pytest.mark.parametrize("horizon", [0.25, 0.42, 0.5, 0.68])
    def test_agreement_holds_across_camera_pitches(self, horizon):
        contract = _contract(horizon=horizon)
        camera = blender_camera_parameters(FRAME_WIDTH, FRAME_HEIGHT, contract)
        expected = world_point_to_image((-1.0, 22.0, 0.5), FRAME_WIDTH, FRAME_HEIGHT, contract)
        assert blender_projection(
            (-1.0, 22.0, 0.5), FRAME_WIDTH, FRAME_HEIGHT, camera,
        ) == pytest.approx(expected, abs=PIXEL_TOLERANCE)

    def test_agreement_holds_on_portrait_footage(self):
        """The reason the sensor fit is pinned horizontal rather than left on AUTO."""
        contract = _contract()
        camera = blender_camera_parameters(720, 1280, contract)
        expected = world_point_to_image((1.5, 15.0, 1.2), 720, 1280, contract)
        assert blender_projection(
            (1.5, 15.0, 1.2), 720, 1280, camera,
        ) == pytest.approx(expected, abs=PIXEL_TOLERANCE)


class TestDerivedValues:
    def test_a_level_camera_points_at_the_horizon(self):
        """Horizon at the frame centre means no pitch: 90 degrees in Blender's terms."""
        camera = blender_camera_parameters(FRAME_WIDTH, FRAME_HEIGHT, _contract(horizon=0.5))
        assert camera["rotation_degrees"] == pytest.approx([90.0, 0.0, 0.0])

    def test_a_high_horizon_tilts_the_camera_downward(self):
        camera = blender_camera_parameters(FRAME_WIDTH, FRAME_HEIGHT, _contract(horizon=0.25))
        assert camera["rotation_degrees"][0] < 90.0

    def test_a_low_horizon_tilts_the_camera_upward(self):
        camera = blender_camera_parameters(FRAME_WIDTH, FRAME_HEIGHT, _contract(horizon=0.75))
        assert camera["rotation_degrees"][0] > 90.0

    def test_roll_and_yaw_are_always_zero(self):
        camera = blender_camera_parameters(FRAME_WIDTH, FRAME_HEIGHT, _contract())
        assert camera["rotation_degrees"][1:] == [0.0, 0.0]

    def test_a_wider_field_of_view_gives_a_shorter_lens(self):
        narrow = blender_camera_parameters(FRAME_WIDTH, FRAME_HEIGHT, _contract(field_of_view=35.0))
        wide = blender_camera_parameters(FRAME_WIDTH, FRAME_HEIGHT, _contract(field_of_view=90.0))
        assert wide["focal_length_mm"] < narrow["focal_length_mm"]

    def test_camera_height_is_carried_from_the_contract(self):
        camera = blender_camera_parameters(FRAME_WIDTH, FRAME_HEIGHT, _contract(height=4.75))
        assert camera["position"] == [0.0, 0.0, 4.75]


class TestProjectionSupport:
    def test_a_calibrated_contract_is_supported(self):
        assert supports_projection(_contract()) is True

    def test_the_legacy_depth_table_model_is_not_supported(self):
        legacy = {
            "ground_mapping": {
                "near_y": 0.98, "far_y": 0.44,
                "near_depth_meters": 4.0, "far_depth_meters": 60.0,
            },
        }
        assert supports_projection(legacy) is False

    def test_a_contract_missing_its_field_of_view_is_not_supported(self):
        contract = _contract()
        del contract["field_of_view_degrees"]
        assert supports_projection(contract) is False

    def test_a_contract_with_a_short_position_is_not_supported(self):
        contract = _contract()
        contract["position"] = [0.0, 0.0]
        assert supports_projection(contract) is False
