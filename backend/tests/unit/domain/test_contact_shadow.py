import sys
from pathlib import Path

import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[4] / "backend"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from domain.contact_shadow import (
    ShadowEllipse,
    footprint_shadow,
    gap_contact_shadows,
)

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720


def _camera():
    return {
        "projection_model": "pinhole_ground_plane_v2",
        "field_of_view_degrees": 54.0,
        "horizon_normalized_y": 0.38,
        "position": [0.0, 0.0, 2.6],
        "ground_mapping": {"near_y": 0.98, "far_y": 0.40},
    }


def _entity(entity_id="p1", kind="person"):
    return {"id": entity_id, "kind": kind}


class TestFootprintShadow:
    def test_a_nearby_actor_casts_a_shadow(self):
        shadow = footprint_shadow((0.0, 8.0, 0.0), "person", FRAME_WIDTH, FRAME_HEIGHT, _camera())
        assert shadow is not None and shadow.is_visible

    def test_the_shadow_sits_under_the_actor(self):
        from domain.camera_projection import world_point_to_image

        shadow = footprint_shadow((1.0, 9.0, 0.0), "person", FRAME_WIDTH, FRAME_HEIGHT, _camera())
        feet = world_point_to_image((1.0, 9.0, 0.0), FRAME_WIDTH, FRAME_HEIGHT, _camera())
        assert shadow.center_x == pytest.approx(feet[0], abs=2.0)
        assert shadow.center_y == pytest.approx(feet[1], abs=shadow.radius_y + 2.0)

    def test_perspective_flattens_the_shadow(self):
        """A ground ellipse seen from a low camera is wider than it is tall."""
        shadow = footprint_shadow((0.0, 14.0, 0.0), "person", FRAME_WIDTH, FRAME_HEIGHT, _camera())
        assert shadow.radius_x > shadow.radius_y

    def test_a_distant_actor_casts_a_smaller_shadow(self):
        near = footprint_shadow((0.0, 6.0, 0.0), "person", FRAME_WIDTH, FRAME_HEIGHT, _camera())
        far = footprint_shadow((0.0, 40.0, 0.0), "person", FRAME_WIDTH, FRAME_HEIGHT, _camera())
        assert far.radius_x < near.radius_x

    def test_a_bus_casts_a_larger_shadow_than_a_person(self):
        person = footprint_shadow((0.0, 12.0, 0.0), "person", FRAME_WIDTH, FRAME_HEIGHT, _camera())
        bus = footprint_shadow((0.0, 12.0, 0.0), "bus", FRAME_WIDTH, FRAME_HEIGHT, _camera())
        assert bus.radius_x > person.radius_x * 3

    def test_an_actor_behind_the_camera_casts_nothing(self):
        assert footprint_shadow(
            (0.0, -20.0, 0.0), "person", FRAME_WIDTH, FRAME_HEIGHT, _camera(),
        ) is None

    def test_an_unprojectable_camera_casts_nothing(self):
        legacy = {"ground_mapping": {"near_y": 0.9, "far_y": 0.4}}
        assert footprint_shadow((0.0, 8.0, 0.0), "person", FRAME_WIDTH, FRAME_HEIGHT, legacy) is None

    def test_a_far_enough_actor_is_dropped_as_sub_pixel(self):
        shadow = footprint_shadow(
            (0.0, 4000.0, 0.0), "person", FRAME_WIDTH, FRAME_HEIGHT, _camera(),
        )
        assert shadow is None or not shadow.is_visible

    def test_blur_never_collapses_to_a_hard_edge(self):
        shadow = footprint_shadow((0.0, 30.0, 0.0), "person", FRAME_WIDTH, FRAME_HEIGHT, _camera())
        assert shadow.blur_pixels >= 3.0


class TestGapShadows:
    def test_one_shadow_per_positioned_actor(self):
        shadows = gap_contact_shadows(
            [_entity("a"), _entity("b")],
            {"a": (0.0, 9.0, 0.0), "b": (2.0, 11.0, 0.0)},
            FRAME_WIDTH, FRAME_HEIGHT, _camera(),
        )
        assert len(shadows) == 2

    def test_an_actor_without_a_position_is_skipped(self):
        shadows = gap_contact_shadows(
            [_entity("a"), _entity("b")], {"a": (0.0, 9.0, 0.0)},
            FRAME_WIDTH, FRAME_HEIGHT, _camera(),
        )
        assert len(shadows) == 1

    def test_no_actors_means_no_shadows(self):
        assert gap_contact_shadows([], {}, FRAME_WIDTH, FRAME_HEIGHT, _camera()) == []

    def test_shadows_serialise_for_the_report(self):
        shadow = ShadowEllipse(10.0, 20.0, 5.0, 2.0)
        assert shadow.as_dict()["radii"] == [5.0, 2.0]
