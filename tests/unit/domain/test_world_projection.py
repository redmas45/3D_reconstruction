"""World-to-image projection, the inverse of the existing ground-ray projection.

M2 crops each render to the actors' projected bounding box (§5.2). If this projection
and `image_point_to_world` disagree, composited actors land in the wrong place — so
the round-trip is tested directly rather than each direction in isolation.
"""

import sys
from pathlib import Path

import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[3] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from domain.camera_projection import (
    camera_basis,
    image_point_to_world,
    world_point_to_image,
)

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
ROUND_TRIP_TOLERANCE_PIXELS = 0.5


def _camera(horizon: float = 0.42, height: float = 3.0) -> dict:
    return {
        "projection_model": "pinhole_ground_plane_v2",
        "field_of_view_degrees": 54.0,
        "horizon_normalized_y": horizon,
        "position": [0.0, 0.0, height],
        "ground_mapping": {"near_y": 0.98, "far_y": horizon + 0.02},
    }


class TestCameraBasisIsOrthonormal:
    """The inverse projection is only valid if the basis really is orthonormal."""

    @pytest.mark.parametrize("horizon", [0.30, 0.42, 0.55])
    def test_axes_are_unit_length(self, horizon):
        basis = camera_basis(FRAME_WIDTH, FRAME_HEIGHT, _camera(horizon))
        for axis in ("right", "forward", "upward"):
            length = sum(component ** 2 for component in basis[axis]) ** 0.5
            assert length == pytest.approx(1.0, abs=1e-9)

    @pytest.mark.parametrize("horizon", [0.30, 0.42, 0.55])
    def test_axes_are_mutually_perpendicular(self, horizon):
        basis = camera_basis(FRAME_WIDTH, FRAME_HEIGHT, _camera(horizon))
        pairs = (("right", "forward"), ("right", "upward"), ("forward", "upward"))
        for first, second in pairs:
            dot = sum(basis[first][i] * basis[second][i] for i in range(3))
            assert dot == pytest.approx(0.0, abs=1e-9)


class TestRoundTrip:
    @pytest.mark.parametrize(
        "image_x,image_y",
        [
            (640.0, 700.0),
            (200.0, 650.0),
            (1100.0, 600.0),
            (640.0, 500.0),
            (320.0, 560.0),
        ],
    )
    def test_ground_point_survives_image_to_world_to_image(self, image_x, image_y):
        camera = _camera()
        world = image_point_to_world(image_x, image_y, FRAME_WIDTH, FRAME_HEIGHT, camera)
        projected = world_point_to_image(world, FRAME_WIDTH, FRAME_HEIGHT, camera)
        assert projected is not None
        assert projected[0] == pytest.approx(image_x, abs=ROUND_TRIP_TOLERANCE_PIXELS)
        assert projected[1] == pytest.approx(image_y, abs=ROUND_TRIP_TOLERANCE_PIXELS)

    @pytest.mark.parametrize("horizon", [0.35, 0.42, 0.50])
    def test_round_trip_holds_across_horizons(self, horizon):
        camera = _camera(horizon)
        world = image_point_to_world(640.0, 680.0, FRAME_WIDTH, FRAME_HEIGHT, camera)
        projected = world_point_to_image(world, FRAME_WIDTH, FRAME_HEIGHT, camera)
        assert projected is not None
        assert projected[0] == pytest.approx(640.0, abs=ROUND_TRIP_TOLERANCE_PIXELS)
        assert projected[1] == pytest.approx(680.0, abs=ROUND_TRIP_TOLERANCE_PIXELS)


class TestProjectionGeometry:
    def test_point_on_the_optical_axis_lands_at_the_frame_centre(self):
        camera = _camera()
        basis = camera_basis(FRAME_WIDTH, FRAME_HEIGHT, camera)
        forward = basis["forward"]
        world = [
            camera["position"][index] + forward[index] * 12.0 for index in range(3)
        ]
        projected = world_point_to_image(world, FRAME_WIDTH, FRAME_HEIGHT, camera)
        assert projected is not None
        assert projected[0] == pytest.approx(FRAME_WIDTH / 2.0, abs=1e-6)
        assert projected[1] == pytest.approx(FRAME_HEIGHT / 2.0, abs=1e-6)

    def test_raising_a_point_moves_it_up_the_frame(self):
        """A standing actor's head must project above their feet."""
        camera = _camera()
        feet = world_point_to_image([0.0, 12.0, 0.0], FRAME_WIDTH, FRAME_HEIGHT, camera)
        head = world_point_to_image([0.0, 12.0, 1.75], FRAME_WIDTH, FRAME_HEIGHT, camera)
        assert feet is not None and head is not None
        assert head[1] < feet[1]

    def test_more_distant_actors_are_smaller(self):
        camera = _camera()
        near = _projected_height(camera, distance=8.0)
        far = _projected_height(camera, distance=24.0)
        assert far < near

    def test_points_left_and_right_of_the_axis_land_on_the_correct_side(self):
        camera = _camera()
        left = world_point_to_image([-4.0, 12.0, 0.0], FRAME_WIDTH, FRAME_HEIGHT, camera)
        right = world_point_to_image([4.0, 12.0, 0.0], FRAME_WIDTH, FRAME_HEIGHT, camera)
        assert left is not None and right is not None
        assert left[0] < FRAME_WIDTH / 2.0 < right[0]


class TestPointsBehindTheCamera:
    def test_point_behind_the_camera_is_not_projected(self):
        camera = _camera()
        assert world_point_to_image([0.0, -20.0, 0.0], FRAME_WIDTH, FRAME_HEIGHT, camera) is None

    def test_point_at_the_camera_plane_is_not_projected(self):
        camera = _camera()
        assert world_point_to_image([0.0, 0.0, 3.0], FRAME_WIDTH, FRAME_HEIGHT, camera) is None


def _projected_height(camera: dict, distance: float) -> float:
    feet = world_point_to_image([0.0, distance, 0.0], FRAME_WIDTH, FRAME_HEIGHT, camera)
    head = world_point_to_image([0.0, distance, 1.75], FRAME_WIDTH, FRAME_HEIGHT, camera)
    assert feet is not None and head is not None
    return feet[1] - head[1]
