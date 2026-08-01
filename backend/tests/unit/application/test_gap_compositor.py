import sys
from pathlib import Path

import numpy
import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[4] / "backend"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from application.gap_compositor import (
    CompositeError,
    alpha_over,
    apply_shadow,
    composite_gap_frame,
    depth_test_alpha,
    match_grade,
    region_slice,
)
from domain.render_region import FULL_FRAME_REGION, RenderRegion

PLATE_HEIGHT = 80
PLATE_WIDTH = 120
PLATE_COLOUR = (60, 70, 80)


def _plate(colour=PLATE_COLOUR):
    return numpy.full((PLATE_HEIGHT, PLATE_WIDTH, 3), colour, dtype=numpy.uint8)


def _actor_layer(shape, rgb=(200, 30, 30), alpha_value=255, covered=None):
    height, width = shape
    layer = numpy.zeros((height, width, 4), dtype=numpy.uint8)
    region = covered or (slice(None), slice(None))
    layer[region[0], region[1], :3] = rgb
    layer[region[0], region[1], 3] = alpha_value
    return layer


class TestRegionSlice:
    def test_full_frame_covers_the_whole_plate(self):
        rows, columns = region_slice(_plate(), FULL_FRAME_REGION)
        assert (rows.start, rows.stop) == (0, PLATE_HEIGHT)
        assert (columns.start, columns.stop) == (0, PLATE_WIDTH)

    def test_partial_region_selects_a_sub_rectangle(self):
        rows, columns = region_slice(_plate(), RenderRegion(0.25, 0.25, 0.75, 0.75))
        assert rows.stop - rows.start == PLATE_HEIGHT // 2
        assert columns.stop - columns.start == PLATE_WIDTH // 2


class TestAlphaOver:
    def test_opaque_foreground_replaces_the_background(self):
        background = _plate()
        foreground = numpy.full_like(background, 255)
        alpha = numpy.full(background.shape[:2], 255, dtype=numpy.uint8)
        assert (alpha_over(background, foreground, alpha) == 255).all()

    def test_transparent_foreground_leaves_the_background(self):
        background = _plate()
        foreground = numpy.full_like(background, 255)
        alpha = numpy.zeros(background.shape[:2], dtype=numpy.uint8)
        assert (alpha_over(background, foreground, alpha) == background).all()

    def test_half_alpha_blends_both(self):
        background = numpy.zeros((4, 4, 3), dtype=numpy.uint8)
        foreground = numpy.full((4, 4, 3), 200, dtype=numpy.uint8)
        alpha = numpy.full((4, 4), 128, dtype=numpy.uint8)
        blended = alpha_over(background, foreground, alpha)
        assert blended.max() == pytest.approx(100, abs=2)


class TestShadow:
    def test_shadow_darkens_the_plate(self):
        plate_region = _plate()
        shadow = numpy.full(plate_region.shape[:2], 255, dtype=numpy.uint8)
        assert apply_shadow(plate_region, shadow).mean() < plate_region.mean()

    def test_no_shadow_leaves_the_plate_unchanged(self):
        plate_region = _plate()
        shadow = numpy.zeros(plate_region.shape[:2], dtype=numpy.uint8)
        assert (apply_shadow(plate_region, shadow) == plate_region).all()

    def test_shadow_is_multiplicative_not_flat_black(self):
        """Plate texture must survive under a shadow rather than being replaced."""
        plate_region = numpy.zeros((10, 10, 3), dtype=numpy.uint8)
        plate_region[:, :5] = 200
        shadow = numpy.full((10, 10), 255, dtype=numpy.uint8)
        shadowed = apply_shadow(plate_region, shadow)
        assert shadowed[:, :5].mean() > shadowed[:, 5:].mean()

    def test_three_channel_shadow_is_accepted(self):
        plate_region = _plate()
        shadow = numpy.full((PLATE_HEIGHT, PLATE_WIDTH, 3), 255, dtype=numpy.uint8)
        assert apply_shadow(plate_region, shadow).mean() < plate_region.mean()


class TestGradeMatching:
    def test_actor_moves_toward_the_plate_colour(self):
        plate_region = numpy.full((20, 20, 3), (60, 70, 80), dtype=numpy.uint8)
        actor_rgb = numpy.full((20, 20, 3), (240, 240, 240), dtype=numpy.uint8)
        alpha = numpy.full((20, 20), 255, dtype=numpy.uint8)
        graded = match_grade(actor_rgb, alpha, plate_region)
        assert graded.mean() < actor_rgb.mean()

    def test_grading_never_fully_matches_the_plate(self):
        """Actors legitimately differ from the background; full correction is wrong."""
        plate_region = numpy.full((20, 20, 3), (60, 70, 80), dtype=numpy.uint8)
        actor_rgb = numpy.full((20, 20, 3), (240, 240, 240), dtype=numpy.uint8)
        alpha = numpy.full((20, 20), 255, dtype=numpy.uint8)
        graded = match_grade(actor_rgb, alpha, plate_region)
        assert graded.mean() > plate_region.mean()

    def test_transparent_pixels_are_not_graded(self):
        plate_region = numpy.full((10, 10, 3), 60, dtype=numpy.uint8)
        actor_rgb = numpy.zeros((10, 10, 3), dtype=numpy.uint8)
        actor_rgb[2:5, 2:5] = 240
        alpha = numpy.zeros((10, 10), dtype=numpy.uint8)
        alpha[2:5, 2:5] = 255
        graded = match_grade(actor_rgb, alpha, plate_region)
        assert (graded[0, 0] == 0).all()

    def test_fully_transparent_layer_is_returned_unchanged(self):
        plate_region = numpy.full((10, 10, 3), 60, dtype=numpy.uint8)
        actor_rgb = numpy.full((10, 10, 3), 200, dtype=numpy.uint8)
        alpha = numpy.zeros((10, 10), dtype=numpy.uint8)
        assert (match_grade(actor_rgb, alpha, plate_region) == actor_rgb).all()

    def test_flat_plate_still_produces_a_valid_result(self):
        plate_region = numpy.full((10, 10, 3), 60, dtype=numpy.uint8)
        actor_rgb = numpy.full((10, 10, 3), 200, dtype=numpy.uint8)
        alpha = numpy.full((10, 10), 255, dtype=numpy.uint8)
        graded = match_grade(actor_rgb, alpha, plate_region)
        assert graded.dtype == numpy.uint8
        assert (graded >= 0).all() and (graded <= 255).all()

    def test_zero_strength_is_a_no_op(self):
        plate_region = numpy.full((10, 10, 3), 60, dtype=numpy.uint8)
        actor_rgb = numpy.full((10, 10, 3), 200, dtype=numpy.uint8)
        alpha = numpy.full((10, 10), 255, dtype=numpy.uint8)
        graded = match_grade(actor_rgb, alpha, plate_region, strength=0.0)
        assert (graded == actor_rgb).all()


class TestDepthTesting:
    def test_actor_behind_the_scene_is_hidden(self):
        alpha = numpy.full((5, 5), 255, dtype=numpy.uint8)
        actor_depth = numpy.full((5, 5), 20.0, dtype=numpy.float32)
        plate_depth = numpy.full((5, 5), 10.0, dtype=numpy.float32)
        assert (depth_test_alpha(alpha, actor_depth, plate_depth) == 0).all()

    def test_actor_in_front_is_kept(self):
        alpha = numpy.full((5, 5), 255, dtype=numpy.uint8)
        actor_depth = numpy.full((5, 5), 5.0, dtype=numpy.float32)
        plate_depth = numpy.full((5, 5), 10.0, dtype=numpy.float32)
        assert (depth_test_alpha(alpha, actor_depth, plate_depth) == 255).all()

    def test_partial_occlusion_hides_only_the_occluded_pixels(self):
        alpha = numpy.full((4, 4), 255, dtype=numpy.uint8)
        actor_depth = numpy.full((4, 4), 5.0, dtype=numpy.float32)
        plate_depth = numpy.full((4, 4), 10.0, dtype=numpy.float32)
        plate_depth[:2] = 1.0
        tested = depth_test_alpha(alpha, actor_depth, plate_depth)
        assert (tested[:2] == 0).all()
        assert (tested[2:] == 255).all()

    def test_absent_depth_is_a_documented_downgrade_not_an_error(self):
        alpha = numpy.full((5, 5), 255, dtype=numpy.uint8)
        assert (depth_test_alpha(alpha, None, None) == 255).all()

    def test_mismatched_depth_shapes_are_rejected(self):
        alpha = numpy.full((5, 5), 255, dtype=numpy.uint8)
        with pytest.raises(CompositeError, match="identical shapes"):
            depth_test_alpha(alpha, numpy.zeros((5, 5)), numpy.zeros((6, 6)))


class TestCompositeGapFrame:
    def test_opaque_actor_appears_in_the_output(self):
        plate = _plate()
        actor = _actor_layer((PLATE_HEIGHT, PLATE_WIDTH))
        composed = composite_gap_frame(plate, actor, FULL_FRAME_REGION)
        assert not numpy.array_equal(composed, plate)

    def test_the_plate_is_never_mutated(self):
        plate = _plate()
        original = plate.copy()
        composite_gap_frame(plate, _actor_layer((PLATE_HEIGHT, PLATE_WIDTH)), FULL_FRAME_REGION)
        assert numpy.array_equal(plate, original)

    def test_fully_transparent_actor_leaves_the_plate_intact(self):
        plate = _plate()
        actor = numpy.zeros((PLATE_HEIGHT, PLATE_WIDTH, 4), dtype=numpy.uint8)
        composed = composite_gap_frame(plate, actor, FULL_FRAME_REGION)
        assert numpy.array_equal(composed, plate)

    def test_a_layer_only_changes_the_plate_where_it_has_alpha(self):
        """Layers are frame-sized and transparent outside the rendered border."""
        plate = _plate()
        region = RenderRegion(0.25, 0.25, 0.75, 0.75)
        rows, columns = region_slice(plate, region)
        layer = numpy.zeros((PLATE_HEIGHT, PLATE_WIDTH, 4), dtype=numpy.uint8)
        layer[rows, columns] = _actor_layer(
            (rows.stop - rows.start, columns.stop - columns.start),
        )
        composed = composite_gap_frame(plate, layer, region)
        assert not numpy.array_equal(composed[rows, columns], plate[rows, columns])
        assert numpy.array_equal(composed[0:5, 0:5], plate[0:5, 0:5])

    def test_misaligned_actor_layer_is_rejected_loudly(self):
        """A silent misalignment would put actors in the wrong part of the frame."""
        plate = _plate()
        region = RenderRegion(0.25, 0.25, 0.75, 0.75)
        with pytest.raises(CompositeError, match="misaligned"):
            composite_gap_frame(plate, _actor_layer((10, 10)), region)

    def test_actor_layer_without_alpha_is_rejected(self):
        plate = _plate()
        rgb_only = numpy.zeros((PLATE_HEIGHT, PLATE_WIDTH, 3), dtype=numpy.uint8)
        with pytest.raises(CompositeError, match="4 channels"):
            composite_gap_frame(plate, rgb_only, FULL_FRAME_REGION)

    def test_shadow_darkens_the_area_around_the_actor(self):
        plate = _plate()
        actor = numpy.zeros((PLATE_HEIGHT, PLATE_WIDTH, 4), dtype=numpy.uint8)
        shadow = numpy.full((PLATE_HEIGHT, PLATE_WIDTH), 255, dtype=numpy.uint8)
        composed = composite_gap_frame(plate, actor, FULL_FRAME_REGION, shadow_layer=shadow)
        assert composed.mean() < plate.mean()

    def test_output_matches_the_plate_dimensions(self):
        plate = _plate()
        composed = composite_gap_frame(
            plate, _actor_layer((PLATE_HEIGHT, PLATE_WIDTH)), FULL_FRAME_REGION,
        )
        assert composed.shape == plate.shape
        assert composed.dtype == numpy.uint8
