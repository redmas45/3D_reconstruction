import sys
from pathlib import Path

import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[3] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from domain.render_region import (
    FULL_FRAME_REGION,
    RenderRegion,
    entity_bounds,
    entity_screen_span,
    gap_render_region,
    world_position_at_frame,
)

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

CAMERA = {
    "projection_model": "pinhole_ground_plane_v2",
    "field_of_view_degrees": 54.0,
    "horizon_normalized_y": 0.42,
    "position": [0.0, 0.0, 3.0],
    "ground_mapping": {"near_y": 0.98, "far_y": 0.44},
}


def _entity(kind="person", waypoints=None):
    return {
        "id": f"{kind}_1",
        "kind": kind,
        "path_prediction": {
            "waypoints": waypoints or [
                {"role": "start", "frame": 100, "world": [-2.0, 12.0, 0.0]},
                {"role": "inferred_midpoint", "frame": 150, "world": [0.0, 12.0, 0.0]},
                {"role": "predicted_end", "frame": 200, "world": [2.0, 12.0, 0.0]},
            ],
        },
    }


class TestWorldPositionInterpolation:
    def test_returns_the_first_waypoint_before_the_path_starts(self):
        assert world_position_at_frame(_entity(), 50) == [-2.0, 12.0, 0.0]

    def test_returns_the_last_waypoint_after_the_path_ends(self):
        assert world_position_at_frame(_entity(), 900) == [2.0, 12.0, 0.0]

    def test_interpolates_between_waypoints(self):
        position = world_position_at_frame(_entity(), 125)
        assert position == pytest.approx([-1.0, 12.0, 0.0])

    def test_lands_exactly_on_a_waypoint_frame(self):
        assert world_position_at_frame(_entity(), 150) == pytest.approx([0.0, 12.0, 0.0])

    def test_unordered_waypoints_are_sorted_before_use(self):
        entity = _entity(waypoints=[
            {"frame": 200, "world": [2.0, 12.0, 0.0]},
            {"frame": 100, "world": [-2.0, 12.0, 0.0]},
            {"frame": 150, "world": [0.0, 12.0, 0.0]},
        ])
        assert world_position_at_frame(entity, 125) == pytest.approx([-1.0, 12.0, 0.0])

    def test_entity_without_waypoints_has_no_position(self):
        assert world_position_at_frame({"path_prediction": {"waypoints": []}}, 100) is None

    def test_entity_without_path_prediction_has_no_position(self):
        assert world_position_at_frame({}, 100) is None


class TestEntityBounds:
    @pytest.mark.parametrize("kind", ["person", "car", "truck", "bus", "bicycle", "motorcycle"])
    def test_every_renderable_class_has_bounds(self, kind):
        length, width, height = entity_bounds(kind)
        assert length > 0 and width > 0 and height > 0

    def test_unknown_class_falls_back_rather_than_raising(self):
        assert entity_bounds("unicorn") == entity_bounds("__missing__")

    def test_a_bus_is_larger_than_a_person(self):
        assert entity_bounds("bus")[2] > entity_bounds("person")[2]


class TestEntityScreenSpan:
    def test_span_is_within_the_normalized_frame(self):
        span = entity_screen_span(_entity(), 150, FRAME_WIDTH, FRAME_HEIGHT, CAMERA)
        assert span is not None
        assert all(-1.0 <= value <= 2.0 for value in span)
        assert span[0] < span[2] and span[1] < span[3]

    def test_a_bus_covers_more_screen_than_a_person_at_the_same_place(self):
        person = entity_screen_span(_entity("person"), 150, FRAME_WIDTH, FRAME_HEIGHT, CAMERA)
        bus = entity_screen_span(_entity("bus"), 150, FRAME_WIDTH, FRAME_HEIGHT, CAMERA)
        assert person is not None and bus is not None
        person_area = (person[2] - person[0]) * (person[3] - person[1])
        bus_area = (bus[2] - bus[0]) * (bus[3] - bus[1])
        assert bus_area > person_area

    def test_entity_behind_the_camera_has_no_span(self):
        entity = _entity(waypoints=[
            {"frame": 100, "world": [0.0, -30.0, 0.0]},
            {"frame": 150, "world": [0.0, -25.0, 0.0]},
            {"frame": 200, "world": [0.0, -20.0, 0.0]},
        ])
        assert entity_screen_span(entity, 150, FRAME_WIDTH, FRAME_HEIGHT, CAMERA) is None


class TestGapRenderRegion:
    def test_region_covers_less_than_the_full_frame_for_one_distant_actor(self):
        entity = _entity(waypoints=[
            {"frame": 100, "world": [0.0, 22.0, 0.0]},
            {"frame": 150, "world": [0.5, 22.0, 0.0]},
            {"frame": 200, "world": [1.0, 22.0, 0.0]},
        ])
        region = gap_render_region([entity], 150, FRAME_WIDTH, FRAME_HEIGHT, CAMERA)
        assert region.coverage < 0.6
        assert not region.is_full_frame

    def test_region_grows_to_contain_two_separated_actors(self):
        left = _entity(waypoints=[{"frame": f, "world": [-6.0, 20.0, 0.0]} for f in (100, 150, 200)])
        right = _entity(waypoints=[{"frame": f, "world": [6.0, 20.0, 0.0]} for f in (100, 150, 200)])
        single = gap_render_region([left], 150, FRAME_WIDTH, FRAME_HEIGHT, CAMERA)
        both = gap_render_region([left, right], 150, FRAME_WIDTH, FRAME_HEIGHT, CAMERA)
        assert both.coverage > single.coverage

    def test_no_entities_falls_back_to_the_full_frame(self):
        assert gap_render_region([], 150, FRAME_WIDTH, FRAME_HEIGHT, CAMERA) == FULL_FRAME_REGION

    def test_entities_behind_the_camera_fall_back_to_the_full_frame(self):
        entity = _entity(waypoints=[{"frame": f, "world": [0.0, -20.0, 0.0]} for f in (100, 150, 200)])
        region = gap_render_region([entity], 150, FRAME_WIDTH, FRAME_HEIGHT, CAMERA)
        assert region == FULL_FRAME_REGION

    def test_a_very_close_actor_falls_back_to_the_full_frame(self):
        """Past the coverage threshold cropping stops being worth the bookkeeping."""
        entity = _entity("bus", waypoints=[
            {"frame": f, "world": [0.0, 5.0, 0.0]} for f in (100, 150, 200)
        ])
        region = gap_render_region([entity], 150, FRAME_WIDTH, FRAME_HEIGHT, CAMERA)
        assert region == FULL_FRAME_REGION

    def test_region_is_clamped_to_the_frame(self):
        entity = _entity(waypoints=[{"frame": f, "world": [-8.0, 9.0, 0.0]} for f in (100, 150, 200)])
        region = gap_render_region([entity], 150, FRAME_WIDTH, FRAME_HEIGHT, CAMERA)
        assert region.minimum_x >= 0.0 and region.minimum_y >= 0.0
        assert region.maximum_x <= 1.0 and region.maximum_y <= 1.0


class TestPixelBoxConversion:
    def test_full_frame_maps_to_the_whole_image(self):
        assert FULL_FRAME_REGION.pixel_box(FRAME_WIDTH, FRAME_HEIGHT) == (
            0, 0, FRAME_WIDTH, FRAME_HEIGHT,
        )

    def test_vertical_axis_is_flipped_exactly_once(self):
        """Blender's border origin is bottom-left; the compositor's is top-left."""
        region = RenderRegion(0.0, 0.0, 1.0, 0.5)
        left, top, right, bottom = region.pixel_box(FRAME_WIDTH, FRAME_HEIGHT)
        assert (left, right) == (0, FRAME_WIDTH)
        # The bottom half in Blender space is the lower half of the image.
        assert top == FRAME_HEIGHT // 2
        assert bottom == FRAME_HEIGHT

    def test_box_is_never_degenerate(self):
        region = RenderRegion(0.5, 0.5, 0.5, 0.5)
        left, top, right, bottom = region.pixel_box(FRAME_WIDTH, FRAME_HEIGHT)
        assert right > left or right >= 1
        assert bottom > top or bottom >= 1

    def test_box_stays_inside_the_image(self):
        region = RenderRegion(0.0, 0.0, 1.0, 1.0)
        left, top, right, bottom = region.pixel_box(FRAME_WIDTH, FRAME_HEIGHT)
        assert 0 <= left < right <= FRAME_WIDTH
        assert 0 <= top < bottom <= FRAME_HEIGHT
