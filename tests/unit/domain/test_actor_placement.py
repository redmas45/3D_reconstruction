"""Placing an entity in the frame, and picking which photograph of it to draw.

Two failures this guards against, both of which look fine in a still and terrible in
motion. Drawing the same sighting on every frame gives a figure that glides along
perfectly rigid. Drawing a sighting that faced the other way gives one that moonwalks.
"""

import sys
from pathlib import Path

import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[3] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from domain.actor_placement import (
    MINIMUM_DRAWN_HEIGHT_PIXELS,
    Observation,
    Placement,
    choose_observation,
    observation_velocities,
    placement_for_frame,
)

WIDTH, HEIGHT = 1280, 720
CAMERA = {
    "projection_model": "pinhole_ground_plane_v2",
    "field_of_view_degrees": 58.0,
    "horizon_normalized_y": 0.39,
    "position": [0.0, -12.0, 2.4],
}


def _entity(positions: dict[int, list[float]]) -> dict:
    return {
        "id": "person_1",
        "kind": "person",
        "path_prediction": {
            "waypoints": [
                {"frame": frame, "world": position}
                for frame, position in sorted(positions.items())
            ],
        },
    }


class TestPlacement:
    def test_an_entity_further_away_is_drawn_smaller(self):
        near = placement_for_frame(
            _entity({0: [0.0, 2.0, 0.0], 100: [0.0, 2.0, 0.0]}), 50, WIDTH, HEIGHT, CAMERA,
        )
        far = placement_for_frame(
            _entity({0: [0.0, 20.0, 0.0], 100: [0.0, 20.0, 0.0]}), 50, WIDTH, HEIGHT, CAMERA,
        )
        assert near is not None and far is not None
        assert near.pixel_height > far.pixel_height

    def test_an_entity_to_the_right_is_drawn_to_the_right(self):
        centre = placement_for_frame(
            _entity({0: [0.0, 10.0, 0.0], 100: [0.0, 10.0, 0.0]}), 50, WIDTH, HEIGHT, CAMERA,
        )
        right = placement_for_frame(
            _entity({0: [4.0, 10.0, 0.0], 100: [4.0, 10.0, 0.0]}), 50, WIDTH, HEIGHT, CAMERA,
        )
        assert right.centre_x > centre.centre_x

    def test_a_walking_entity_reports_the_direction_it_moves_on_screen(self):
        entity = _entity({0: [-4.0, 10.0, 0.0], 100: [4.0, 10.0, 0.0]})
        placement = placement_for_frame(entity, 50, WIDTH, HEIGHT, CAMERA)
        assert placement.velocity_x > 0

    def test_a_camera_without_pinhole_parameters_cannot_place_anything(self):
        legacy = {"projection_model": "pinhole_ground_plane"}
        entity = _entity({0: [0.0, 10.0, 0.0], 100: [0.0, 10.0, 0.0]})
        assert placement_for_frame(entity, 50, WIDTH, HEIGHT, legacy) is None

    def test_an_entity_with_no_path_at_that_frame_is_not_placed(self):
        assert placement_for_frame(
            {"id": "x", "kind": "person", "path_prediction": {"waypoints": []}},
            50, WIDTH, HEIGHT, CAMERA,
        ) is None

    def test_a_figure_too_small_to_draw_is_reported_as_such(self):
        assert not Placement(100.0, 300.0, MINIMUM_DRAWN_HEIGHT_PIXELS - 1, 0.0, 0.0).is_drawable

    def test_a_figure_large_enough_is_drawable(self):
        assert Placement(100.0, 300.0, MINIMUM_DRAWN_HEIGHT_PIXELS + 1, 0.0, 0.0).is_drawable


def _observations(count: int, step: float, height: float = 100.0) -> list[Observation]:
    return [
        Observation(
            source_frame=index * 5,
            left=100.0 + index * step,
            top=200.0,
            right=140.0 + index * step,
            bottom=200.0 + height,
        )
        for index in range(count)
    ]


class TestObservationVelocities:
    def test_a_subject_moving_right_reports_rightward_motion(self):
        velocities = observation_velocities(_observations(4, step=10.0))
        assert all(velocity[0] > 0 for velocity in velocities)

    def test_a_subject_moving_left_reports_leftward_motion(self):
        velocities = observation_velocities(_observations(4, step=-10.0))
        assert all(velocity[0] < 0 for velocity in velocities)

    def test_a_single_sighting_has_no_measurable_motion(self):
        assert observation_velocities(_observations(1, step=10.0)) == [(0.0, 0.0)]

    def test_every_sighting_gets_a_velocity(self):
        observations = _observations(5, step=8.0)
        assert len(observation_velocities(observations)) == len(observations)


class TestChoosingAnObservation:
    def test_a_sighting_moving_the_same_way_is_preferred(self):
        """Drawing a subject that was walking left onto a subject walking right is the
        clearest possible tell that the footage was assembled."""
        observations = _observations(3, step=12.0) + _observations(3, step=-12.0)
        velocities = observation_velocities(_observations(3, step=12.0)) + \
            observation_velocities(_observations(3, step=-12.0))
        rightward = Placement(600.0, 500.0, 100.0, velocity_x=9.0, velocity_y=0.0)
        chosen = choose_observation(observations, velocities, rightward)
        assert velocities[chosen][0] > 0

    def test_a_sighting_near_the_drawn_size_is_preferred_over_a_distant_one(self):
        observations = [
            Observation(0, 100.0, 200.0, 140.0, 240.0),    # 40px tall
            Observation(5, 100.0, 200.0, 140.0, 300.0),    # 100px tall
        ]
        velocities = [(0.0, 0.0), (0.0, 0.0)]
        placement = Placement(600.0, 500.0, 100.0, 0.0, 0.0)
        assert choose_observation(observations, velocities, placement) == 1

    def test_the_previous_sighting_is_avoided_so_the_figure_is_not_frozen(self):
        observations = _observations(4, step=10.0)
        velocities = observation_velocities(observations)
        placement = Placement(600.0, 500.0, 100.0, 8.0, 0.0)
        first = choose_observation(observations, velocities, placement)
        second = choose_observation(observations, velocities, placement, previous_index=first)
        assert second != first

    def test_no_usable_sighting_returns_nothing(self):
        """Every sighting is far too small to be blown up to the drawn size."""
        observations = [Observation(0, 100.0, 200.0, 110.0, 205.0)]
        assert choose_observation(observations, [(0.0, 0.0)], Placement(600.0, 500.0, 400.0, 0.0, 0.0)) is None

    def test_an_empty_bank_returns_nothing(self):
        assert choose_observation([], [], Placement(600.0, 500.0, 100.0, 0.0, 0.0)) is None
