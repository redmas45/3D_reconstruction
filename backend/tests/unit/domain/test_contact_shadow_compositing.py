"""Shadow drawing, grain matching and edge softening in the compositor.

These are the three cheap levers that decide whether a rendered figure sits in the plate
or floats on top of it. Each is small enough to be tempting to skip and obvious enough
to notice when it is missing.
"""

import sys
from pathlib import Path

import numpy
import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[4] / "backend"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from application.gap_compositor import (
    apply_grain,
    composite_gap_frame,
    draw_contact_shadows,
    plate_grain_sigma,
    soften_alpha,
)
from domain.contact_shadow import ShadowEllipse
from domain.render_region import FULL_FRAME_REGION

HEIGHT = 120
WIDTH = 160


def _plate(value: int = 140) -> numpy.ndarray:
    return numpy.full((HEIGHT, WIDTH, 3), value, dtype=numpy.uint8)


def _actor(alpha_box=(40, 60, 70, 90)) -> numpy.ndarray:
    layer = numpy.zeros((HEIGHT, WIDTH, 4), dtype=numpy.uint8)
    top, bottom, left, right = alpha_box
    layer[top:bottom, left:right, :3] = 90
    layer[top:bottom, left:right, 3] = 255
    return layer


class TestContactShadows:
    def test_a_shadow_darkens_the_ground_beneath_it(self):
        shadowed = draw_contact_shadows(_plate(), [ShadowEllipse(80, 90, 20, 8)])
        assert shadowed[90, 80].mean() < 140

    def test_the_plate_is_untouched_away_from_the_shadow(self):
        shadowed = draw_contact_shadows(_plate(), [ShadowEllipse(80, 90, 20, 8)])
        assert (shadowed[5, 5] == 140).all()

    def test_no_shadows_leaves_the_plate_exactly_as_it_was(self):
        plate = _plate()
        assert draw_contact_shadows(plate, []) is plate

    def test_darkening_is_multiplicative_so_texture_survives(self):
        """A shadow removes light; it must not paint flat grey over the paving."""
        plate = _plate()
        plate[:, ::2] = 200  # vertical stripes standing in for ground texture
        shadowed = draw_contact_shadows(plate, [ShadowEllipse(80, 60, 40, 30)])
        centre = shadowed[60, 70:90]
        assert centre.std() > 5, "the shadow flattened the plate's own texture"

    def test_a_sub_pixel_shadow_is_skipped(self):
        plate = _plate()
        assert (draw_contact_shadows(plate, [ShadowEllipse(80, 90, 0.2, 0.1)]) == plate).all()

    def test_overlapping_shadows_do_not_compound_into_black(self):
        shadowed = draw_contact_shadows(
            _plate(), [ShadowEllipse(80, 60, 25, 20), ShadowEllipse(82, 62, 25, 20)],
        )
        assert shadowed.min() > 30

    def test_a_shadow_is_soft_edged(self):
        shadowed = draw_contact_shadows(_plate(), [ShadowEllipse(80, 60, 25, 18)])
        row = shadowed[60, :, 0].astype(int)
        transitions = numpy.abs(numpy.diff(row))
        assert transitions.max() < 40, "the shadow has a hard edge"


class TestGrain:
    def test_grain_is_measured_from_the_plate(self):
        noisy = _plate().astype(numpy.float32)
        noisy += numpy.random.default_rng(0).normal(0, 6, noisy.shape)
        measured = plate_grain_sigma(numpy.clip(noisy, 0, 255).astype(numpy.uint8))
        assert 3.0 < measured < 9.0

    def test_a_clean_plate_measures_almost_no_grain(self):
        assert plate_grain_sigma(_plate()) < 1.0

    def test_grain_only_lands_where_the_actor_is(self):
        frame = _plate(100)
        alpha = numpy.zeros((HEIGHT, WIDTH), dtype=numpy.uint8)
        alpha[40:60, 40:60] = 255
        grained = apply_grain(frame, alpha, sigma=6.0, seed=1)
        assert (grained[0:20, 0:20] == 100).all()
        assert grained[40:60, 40:60].std() > 1.0

    def test_a_clean_plate_adds_no_grain(self):
        frame = _plate(100)
        alpha = numpy.full((HEIGHT, WIDTH), 255, dtype=numpy.uint8)
        assert (apply_grain(frame, alpha, sigma=0.1, seed=1) == frame).all()

    def test_grain_is_reproducible_for_a_given_seed(self):
        frame, alpha = _plate(100), numpy.full((HEIGHT, WIDTH), 255, dtype=numpy.uint8)
        assert (apply_grain(frame, alpha, 6.0, 7) == apply_grain(frame, alpha, 6.0, 7)).all()

    def test_grain_differs_between_frames(self):
        """Static noise reads as a texture stuck to the actor rather than as sensor grain."""
        frame, alpha = _plate(100), numpy.full((HEIGHT, WIDTH), 255, dtype=numpy.uint8)
        assert not (apply_grain(frame, alpha, 6.0, 1) == apply_grain(frame, alpha, 6.0, 2)).all()


class TestEdgeSoftening:
    def test_a_hard_matte_edge_is_softened(self):
        alpha = numpy.zeros((HEIGHT, WIDTH), dtype=numpy.uint8)
        alpha[:, 80:] = 255
        softened = soften_alpha(alpha)
        assert 0 < softened[60, 79] < 255

    def test_the_interior_stays_fully_opaque(self):
        alpha = numpy.zeros((HEIGHT, WIDTH), dtype=numpy.uint8)
        alpha[20:100, 20:140] = 255
        assert soften_alpha(alpha)[60, 80] == 255


class TestCompositeIntegration:
    def test_shadows_and_grain_reach_the_composited_frame(self):
        composed = composite_gap_frame(
            _plate(), _actor(), FULL_FRAME_REGION,
            contact_shadows=[ShadowEllipse(65, 92, 18, 6)], grain_seed=3,
        )
        # The actor covers rows 40-60, columns 70-90; the shadow sits below it.
        assert composed[92, 65].mean() < 140, "the shadow did not darken the ground"
        assert composed[50, 80].mean() < 140, "the actor did not land on the plate"

    def test_a_frame_sized_layer_mismatch_is_still_refused(self):
        from application.gap_compositor import CompositeError

        with pytest.raises(CompositeError, match="misaligned"):
            composite_gap_frame(
                _plate(), numpy.zeros((10, 10, 4), dtype=numpy.uint8), FULL_FRAME_REGION,
            )

    def test_the_plate_is_never_mutated(self):
        plate = _plate()
        original = plate.copy()
        composite_gap_frame(
            plate, _actor(), FULL_FRAME_REGION,
            contact_shadows=[ShadowEllipse(65, 92, 18, 6)],
        )
        assert (plate == original).all()
