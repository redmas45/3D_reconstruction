"""Drawing a gap's actors from photographs into the same layers Blender would write.

The contract this must honour is exact: one frame-sized RGBA PNG per sparse sample,
named `frame_NNNNNN.png`. Everything downstream — plate compositing, optical-flow
expansion, encoding — was built against that and is not aware of where the pixels came
from.
"""

import sys
from pathlib import Path

import cv2
import numpy
import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[4] / "backend"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from application.exemplar_gap_renderer import (
    coverage,
    draw_cutout,
    entities_with_banks,
    render_exemplar_layers,
)
from application.exemplar_library import ExemplarBank
from domain.actor_placement import Observation, Placement

WIDTH, HEIGHT = 320, 240
CAMERA = {
    "projection_model": "pinhole_ground_plane_v2",
    "field_of_view_degrees": 58.0,
    "horizon_normalized_y": 0.39,
    "position": [0.0, -12.0, 2.4],
}


class _Sparse:
    def __init__(self, render_index: int, source_frame: int):
        self.render_index = render_index
        self.source_frame = source_frame
        self.region = None
        self.shadows = ()


def _cutout(colour=(200, 50, 50), height=60, width=24) -> numpy.ndarray:
    cutout = numpy.zeros((height, width, 4), numpy.uint8)
    cutout[:, :, :3] = colour
    cutout[:, :, 3] = 255
    return cutout


def _bank(entity_id: str, count: int = 4, colour=(200, 50, 50)) -> ExemplarBank:
    observations = tuple(
        Observation(index * 5, 10.0, 10.0, 34.0, 70.0) for index in range(count)
    )
    return ExemplarBank(
        entity_id=entity_id,
        observations=observations,
        cutouts=tuple(_cutout(colour) for _ in range(count)),
        velocities=tuple((5.0, 0.0) for _ in range(count)),
    )


def _entity(entity_id: str, y_depth: float) -> dict:
    return {
        "id": entity_id,
        "kind": "person",
        "path_prediction": {
            "waypoints": [
                {"frame": 0, "world": [-2.0, y_depth, 0.0]},
                {"frame": 100, "world": [2.0, y_depth, 0.0]},
            ],
        },
    }


def _plan(entities: list[dict]) -> dict:
    return {"gap_index": 0, "camera": CAMERA, "entities": entities}


class TestDrawing:
    def test_a_cut_out_lands_on_the_layer(self):
        layer = numpy.zeros((HEIGHT, WIDTH, 4), numpy.uint8)
        assert draw_cutout(layer, _cutout(), Placement(160.0, 200.0, 60.0, 0.0, 0.0))
        assert layer[:, :, 3].max() == 255

    def test_a_cut_out_entirely_off_frame_draws_nothing(self):
        layer = numpy.zeros((HEIGHT, WIDTH, 4), numpy.uint8)
        assert not draw_cutout(layer, _cutout(), Placement(-500.0, 200.0, 60.0, 0.0, 0.0))
        assert layer[:, :, 3].max() == 0

    def test_a_cut_out_is_drawn_standing_on_its_foot_position(self):
        layer = numpy.zeros((HEIGHT, WIDTH, 4), numpy.uint8)
        draw_cutout(layer, _cutout(), Placement(160.0, 200.0, 60.0, 0.0, 0.0))
        column = layer[:, 160, 3]
        assert column[199] > 0, "nothing at the feet"
        assert column[210] == 0, "drawn below its own feet"

    def test_a_cut_out_is_scaled_to_the_requested_height(self):
        layer = numpy.zeros((HEIGHT, WIDTH, 4), numpy.uint8)
        draw_cutout(layer, _cutout(height=60), Placement(160.0, 200.0, 30.0, 0.0, 0.0))
        rows = numpy.nonzero(layer[:, 160, 3])[0]
        assert 26 <= (rows.max() - rows.min() + 1) <= 34

    def test_a_degenerate_target_size_draws_nothing(self):
        layer = numpy.zeros((HEIGHT, WIDTH, 4), numpy.uint8)
        assert not draw_cutout(layer, _cutout(), Placement(160.0, 200.0, 0.5, 0.0, 0.0))


class TestCoverage:
    def test_full_coverage_when_every_entity_has_footage(self):
        plan = _plan([_entity("a", 8.0), _entity("b", 12.0)])
        assert coverage(plan, {"a": _bank("a"), "b": _bank("b")}) == 1.0

    def test_partial_coverage_is_reported(self):
        plan = _plan([_entity("a", 8.0), _entity("b", 12.0)])
        assert coverage(plan, {"a": _bank("a")}) == 0.5

    def test_a_gap_with_no_entities_counts_as_covered(self):
        assert coverage(_plan([]), {}) == 1.0

    def test_only_entities_with_footage_are_listed(self):
        plan = _plan([_entity("a", 8.0), _entity("b", 12.0)])
        listed = entities_with_banks(plan, {"b": _bank("b")})
        assert [entity["id"] for entity in listed] == ["b"]


class TestLayerRendering:
    def test_one_layer_is_written_per_sparse_sample(self, tmp_path):
        sparse = [_Sparse(index + 1, 10 + index * 4) for index in range(5)]
        render_exemplar_layers(
            _plan([_entity("a", 8.0)]), sparse, {"a": _bank("a")},
            WIDTH, HEIGHT, tmp_path,
        )
        assert sorted(item.name for item in tmp_path.glob("*.png")) == [
            f"frame_{index:06d}.png" for index in range(1, 6)
        ]

    def test_layers_are_frame_sized_rgba(self, tmp_path):
        sparse = [_Sparse(1, 10)]
        render_exemplar_layers(
            _plan([_entity("a", 8.0)]), sparse, {"a": _bank("a")},
            WIDTH, HEIGHT, tmp_path,
        )
        layer = cv2.imread(str(tmp_path / "frame_000001.png"), cv2.IMREAD_UNCHANGED)
        assert layer.shape == (HEIGHT, WIDTH, 4)

    def test_the_actor_is_actually_drawn(self, tmp_path):
        sparse = [_Sparse(1, 50)]
        report = render_exemplar_layers(
            _plan([_entity("a", 8.0)]), sparse, {"a": _bank("a")},
            WIDTH, HEIGHT, tmp_path,
        )
        assert report["actor_draws"] == 1
        assert report["frames_with_no_actor"] == 0

    def test_a_nearer_actor_is_drawn_over_a_further_one(self, tmp_path):
        """Without a depth rule a crowd reads as a collage. Foot position supplies it."""
        near, far = (150, 30, 30), (30, 200, 30)
        plan = _plan([_entity("far", 24.0), _entity("near", 4.0)])
        sparse = [_Sparse(1, 50)]
        render_exemplar_layers(
            plan, sparse,
            {"far": _bank("far", colour=far), "near": _bank("near", colour=near)},
            WIDTH, HEIGHT, tmp_path,
        )
        layer = cv2.imread(str(tmp_path / "frame_000001.png"), cv2.IMREAD_UNCHANGED)
        drawn = layer[layer[:, :, 3] > 0][:, :3]
        # The near actor is much larger on screen, so its colour must dominate.
        near_pixels = numpy.all(numpy.abs(drawn.astype(int) - list(near)) < 30, axis=1).sum()
        far_pixels = numpy.all(numpy.abs(drawn.astype(int) - list(far)) < 30, axis=1).sum()
        assert near_pixels > far_pixels

    def test_the_same_sighting_is_not_reused_on_consecutive_frames(self, tmp_path):
        """A figure drawn from one photograph for a whole gap glides along frozen — the
        clearest tell that a composite is not real footage. Each cut-out here carries a
        distinct colour, so which one was drawn is readable straight off the layer."""
        sparse = [_Sparse(index + 1, 10 + index * 6) for index in range(6)]
        observations = tuple(
            Observation(index * 5, 10.0, 10.0, 34.0, 70.0) for index in range(5)
        )
        bank = ExemplarBank(
            entity_id="a",
            observations=observations,
            cutouts=tuple(
                _cutout(colour=(40 + index * 40, 30, 30)) for index in range(5)
            ),
            velocities=tuple((5.0, 0.0) for _ in range(5)),
        )
        render_exemplar_layers(
            _plan([_entity("a", 8.0)]), sparse, {"a": bank}, WIDTH, HEIGHT, tmp_path,
        )
        drawn = []
        for index in range(1, 7):
            layer = cv2.imread(
                str(tmp_path / f"frame_{index:06d}.png"), cv2.IMREAD_UNCHANGED,
            )
            opaque = layer[layer[:, :, 3] > 128]
            drawn.append(int(numpy.median(opaque[:, 0])) if len(opaque) else -1)
        assert all(
            earlier != later for earlier, later in zip(drawn, drawn[1:])
        ), f"the same photograph was drawn twice in a row: {drawn}"

    def test_an_entity_with_no_path_leaves_an_empty_layer(self, tmp_path):
        plan = _plan([{"id": "a", "kind": "person", "path_prediction": {"waypoints": []}}])
        report = render_exemplar_layers(
            plan, [_Sparse(1, 50)], {"a": _bank("a")}, WIDTH, HEIGHT, tmp_path,
        )
        assert report["frames_with_no_actor"] == 1
        layer = cv2.imread(str(tmp_path / "frame_000001.png"), cv2.IMREAD_UNCHANGED)
        assert layer[:, :, 3].max() == 0

    def test_no_partial_file_is_left_behind(self, tmp_path):
        render_exemplar_layers(
            _plan([_entity("a", 8.0)]), [_Sparse(1, 50)], {"a": _bank("a")},
            WIDTH, HEIGHT, tmp_path,
        )
        assert list(tmp_path.glob("*.writing.png")) == []
