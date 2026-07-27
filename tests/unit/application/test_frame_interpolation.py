"""Motion-compensated expansion of sparse renders back to the source frame rate.

The defect these guard against is judder: a 12 fps render repeated to 30 fps has the
exact right duration and looks like a slideshow next to the real footage either side
of it. The measurable signature is that all the change is concentrated in a few frames
and the rest are identical, which the smoothness tests here assert against directly.
"""

import sys
from pathlib import Path

import numpy
import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[3] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from application.actor_gap_renderer import ActorRenderError, expand_to_source_frames

HEIGHT = 96
WIDTH = 128


def _frame_with_square_at(left: int) -> numpy.ndarray:
    """A moving bright square on a textured background — something flow can track."""
    frame = numpy.tile(
        (numpy.arange(WIDTH, dtype=numpy.uint8) * 2 % 90 + 40)[None, :, None], (HEIGHT, 1, 3),
    ).astype(numpy.uint8)
    frame[36:60, left:left + 24] = 235
    return frame


def _moving_samples(count: int = 5, step: int = 14) -> list[numpy.ndarray]:
    return [_frame_with_square_at(8 + index * step) for index in range(count)]


def _square_centre(frame: numpy.ndarray) -> float:
    columns = numpy.where(frame[48, :, 0] > 200)[0]
    return float(columns.mean()) if len(columns) else float("nan")


def _step_sizes(frames: list[numpy.ndarray]) -> list[float]:
    return [
        float(numpy.abs(later.astype(numpy.int16) - earlier.astype(numpy.int16)).mean())
        for earlier, later in zip(frames, frames[1:])
    ]


class TestContract:
    def test_the_exact_source_frame_count_is_restored(self):
        assert len(expand_to_source_frames(_moving_samples(), 150)) == 150

    def test_the_first_and_last_samples_are_reproduced_exactly(self):
        samples = _moving_samples()
        expanded = expand_to_source_frames(samples, 60)
        assert numpy.array_equal(expanded[0], samples[0])
        assert numpy.array_equal(expanded[-1], samples[-1])

    def test_frames_keep_the_plate_shape(self):
        expanded = expand_to_source_frames(_moving_samples(), 40)
        assert all(frame.shape == (HEIGHT, WIDTH, 3) for frame in expanded)

    def test_a_single_sample_expands_without_interpolating(self):
        assert len(expand_to_source_frames([_frame_with_square_at(10)], 30)) == 30

    def test_empty_input_is_rejected(self):
        with pytest.raises(ActorRenderError, match="No composited frames"):
            expand_to_source_frames([], 10)

    def test_non_positive_frame_count_is_rejected(self):
        with pytest.raises(ActorRenderError, match="must be positive"):
            expand_to_source_frames(_moving_samples(), 0)


class TestSmoothness:
    def test_motion_is_spread_across_frames_rather_than_jumping(self):
        """The judder signature: nearest-sample leaves most frames identical.

        Interpolation should leave no completely static frames in a sequence where the
        subject is moving throughout.
        """
        expanded = expand_to_source_frames(_moving_samples(), 40)
        static = sum(1 for step in _step_sizes(expanded) if step < 0.01)
        assert static == 0, f"{static} of {len(expanded) - 1} frame steps showed no change"

    def test_nearest_sample_expansion_does_show_that_judder(self):
        """The control: without interpolation the same sequence is mostly frozen."""
        expanded = expand_to_source_frames(_moving_samples(), 40, interpolate=False)
        static = sum(1 for step in _step_sizes(expanded) if step < 0.01)
        assert static > len(expanded) // 2

    def test_step_sizes_are_far_more_even_than_nearest_sample(self):
        smooth = _step_sizes(expand_to_source_frames(_moving_samples(), 40))
        stepped = _step_sizes(expand_to_source_frames(_moving_samples(), 40, interpolate=False))
        assert numpy.std(smooth) < numpy.std(stepped) / 2.0

    def test_the_subject_advances_without_going_backwards(self):
        """Sub-pixel wobble is measurement noise; a real reversal is a visible stutter.

        The tolerance is one pixel because the centroid is read off a single row of a
        bilinearly resampled image, which is itself only accurate to about that.
        """
        centres = [
            _square_centre(frame) for frame in expand_to_source_frames(_moving_samples(), 40)
        ]
        reversals = [
            earlier - later
            for earlier, later in zip(centres, centres[1:]) if later < earlier - 1.0
        ]
        assert reversals == []
        assert centres[-1] > centres[0]

    def test_an_interpolated_frame_lands_between_its_neighbours(self):
        """Not a cross-fade: the subject must be at an intermediate *position*."""
        samples = [_frame_with_square_at(8), _frame_with_square_at(24)]
        middle = expand_to_source_frames(samples, 3)[1]
        assert 8 + 12 < _square_centre(middle) < 24 + 12


class TestUntrustworthyMotion:
    """Displacement beyond what flow can resolve must degrade to judder, not to smear."""

    def test_a_large_jump_falls_back_to_a_real_sample(self):
        samples = [_frame_with_square_at(8), _frame_with_square_at(96)]
        middle = expand_to_source_frames(samples, 3)[1]
        centre = _square_centre(middle)
        assert centre == pytest.approx(8 + 12, abs=1.0) or centre == pytest.approx(96 + 12, abs=1.0)

    def test_the_subject_is_never_dissolved(self):
        """The failure this guard exists for: a smeared frame with no subject at all."""
        samples = [_frame_with_square_at(8), _frame_with_square_at(96)]
        for frame in expand_to_source_frames(samples, 9):
            assert frame.max() > 200, "the subject vanished from an interpolated frame"

    def test_small_motion_is_still_interpolated(self):
        samples = [_frame_with_square_at(8), _frame_with_square_at(20)]
        centres = {_square_centre(f) for f in expand_to_source_frames(samples, 7)}
        assert len(centres) > 2


class TestNoInvention:
    def test_expansion_never_moves_past_the_last_sample(self):
        samples = _moving_samples()
        expanded = expand_to_source_frames(samples, 50)
        assert _square_centre(expanded[-1]) == pytest.approx(_square_centre(samples[-1]), abs=1.0)

    def test_expansion_never_starts_before_the_first_sample(self):
        samples = _moving_samples()
        expanded = expand_to_source_frames(samples, 50)
        assert _square_centre(expanded[0]) == pytest.approx(_square_centre(samples[0]), abs=1.0)

    def test_a_static_subject_stays_static(self):
        """Flow on identical frames must not hallucinate movement."""
        still = [_frame_with_square_at(40) for _ in range(4)]
        expanded = expand_to_source_frames(still, 20)
        assert max(_step_sizes(expanded)) < 0.5

    def test_expansion_is_reproducible(self):
        samples = _moving_samples()
        first = expand_to_source_frames(samples, 25)
        second = expand_to_source_frames(samples, 25)
        assert all(numpy.array_equal(a, b) for a, b in zip(first, second))
