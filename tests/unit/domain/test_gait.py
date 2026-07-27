"""The walk cycle, checked as motion rather than as numbers.

Each test names a way a gait can look wrong to a viewer — sliding feet, both legs
forward at once, a knee bending backwards, a standing figure marching on the spot — and
asserts it does not happen. That is the level a viewer judges it at.
"""

import math
import sys
from pathlib import Path

import pytest

SOURCE_ROOT = Path(__file__).resolve().parents[3] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from domain.gait import (
    IDLE_SPEED_METERS_PER_SECOND,
    STRIDE_LIMITS_METERS,
    cycle_phase,
    is_walking,
    stride_length,
    walk_pose,
)

WALKING_SPEED = 1.4


class TestStride:
    def test_faster_walking_lengthens_the_stride(self):
        assert stride_length(2.2) > stride_length(0.8)

    def test_stride_stays_within_plausible_human_bounds(self):
        for speed in (0.0, 0.5, 1.4, 3.0, 40.0):
            assert STRIDE_LIMITS_METERS[0] <= stride_length(speed) <= STRIDE_LIMITS_METERS[1]

    def test_a_nonsensical_negative_speed_still_gives_a_usable_stride(self):
        assert stride_length(-5.0) >= STRIDE_LIMITS_METERS[0]


class TestPhaseLocking:
    def test_phase_advances_with_distance(self):
        assert cycle_phase(0.4, WALKING_SPEED) != cycle_phase(0.0, WALKING_SPEED)

    def test_one_stride_returns_to_the_same_phase(self):
        stride = stride_length(WALKING_SPEED)
        assert cycle_phase(stride, WALKING_SPEED) == pytest.approx(
            cycle_phase(0.0, WALKING_SPEED), abs=1e-9,
        )

    def test_phase_is_locked_to_distance_not_to_time(self):
        """The anti-foot-sliding guarantee: same distance, same pose, any frame rate.

        A gait driven by elapsed time would give these two different phases, and the
        feet would skate across the ground whenever the sample spacing changed.
        """
        coarse = walk_pose(3.0, WALKING_SPEED, elapsed_seconds=2.0)
        fine = walk_pose(3.0, WALKING_SPEED, elapsed_seconds=9.0)
        assert coarse.rotations["thigh.L"] == fine.rotations["thigh.L"]

    def test_phase_always_lies_in_the_unit_interval(self):
        for distance in (0.0, 0.3, 7.5, 123.4):
            assert 0.0 <= cycle_phase(distance, WALKING_SPEED) < 1.0


class TestWalkingPose:
    def test_the_legs_are_never_in_the_same_place(self):
        """Both legs swinging together is the most obvious possible gait failure."""
        for distance in (0.0, 0.2, 0.5, 0.9, 1.3, 2.1):
            pose = walk_pose(distance, WALKING_SPEED)
            assert pose.rotations["thigh.L"][0] != pytest.approx(
                pose.rotations["thigh.R"][0], abs=1e-6,
            ) or abs(pose.rotations["thigh.L"][0]) < 1e-6

    def test_the_legs_are_exactly_out_of_phase(self):
        pose = walk_pose(0.35, WALKING_SPEED)
        assert pose.rotations["thigh.L"][0] == pytest.approx(
            -pose.rotations["thigh.R"][0], abs=1e-9,
        )

    def test_knees_only_ever_bend_one_way(self):
        """A knee is a hinge. A negative flexion is a backwards-breaking leg."""
        for step in range(60):
            pose = walk_pose(step * 0.05, WALKING_SPEED)
            assert pose.rotations["shin.L"][0] >= 0.0
            assert pose.rotations["shin.R"][0] >= 0.0

    def test_arms_swing_opposite_to_the_leg_on_the_same_side(self):
        pose = walk_pose(0.0, WALKING_SPEED)
        assert pose.rotations["thigh.L"][0] * pose.rotations["upper_arm.L"][0] < 0.0

    def test_faster_walking_swings_the_legs_further(self):
        slow = abs(walk_pose(0.0, 0.6).rotations["thigh.L"][0])
        fast = abs(walk_pose(0.0, 2.4).rotations["thigh.L"][0])
        assert fast > slow

    def test_the_pelvis_dips_rather_than_rising(self):
        offsets = [walk_pose(step * 0.05, WALKING_SPEED).root_offset[2] for step in range(40)]
        assert max(offsets) <= 0.0
        assert min(offsets) < 0.0

    def test_the_pelvis_dips_twice_per_stride(self):
        stride = stride_length(WALKING_SPEED)
        samples = 96
        offsets = [
            walk_pose(index * stride / samples, WALKING_SPEED).root_offset[2]
            for index in range(samples)
        ]
        dips = sum(
            1 for index in range(samples)
            if offsets[index] < offsets[index - 1] and offsets[index] < offsets[(index + 1) % samples]
        )
        assert dips == 2

    def test_the_cycle_is_continuous_across_the_wrap(self):
        """A discontinuity at the loop point is a visible hitch every stride."""
        stride = stride_length(WALKING_SPEED)
        before = walk_pose(stride - 1e-4, WALKING_SPEED).rotations["thigh.L"][0]
        after = walk_pose(stride + 1e-4, WALKING_SPEED).rotations["thigh.L"][0]
        assert before == pytest.approx(after, abs=1e-3)

    def test_every_posed_bone_is_finite(self):
        pose = walk_pose(1.7, 3.5)
        assert all(
            math.isfinite(angle) for rotation in pose.rotations.values() for angle in rotation
        )


class TestStandingStill:
    def test_a_stationary_figure_does_not_walk(self):
        pose = walk_pose(0.0, 0.0)
        assert pose.rotations["thigh.L"] == (0.0, 0.0, 0.0)
        assert pose.rotations["thigh.R"] == (0.0, 0.0, 0.0)

    def test_a_barely_moving_figure_does_not_walk(self):
        pose = walk_pose(0.01, IDLE_SPEED_METERS_PER_SECOND - 0.01)
        assert pose.rotations["thigh.L"][0] == 0.0

    def test_the_idle_threshold_is_where_walking_begins(self):
        assert not is_walking(IDLE_SPEED_METERS_PER_SECOND - 0.001)
        assert is_walking(IDLE_SPEED_METERS_PER_SECOND)

    def test_a_standing_figure_still_sways_over_time(self):
        """Perfectly frozen reads as a paused video, not as a person standing."""
        first = walk_pose(0.0, 0.0, elapsed_seconds=0.0).rotations["pelvis"]
        later = walk_pose(0.0, 0.0, elapsed_seconds=1.2).rotations["pelvis"]
        assert first != later

    def test_a_standing_figure_does_not_bob(self):
        assert walk_pose(0.0, 0.0).root_offset == (0.0, 0.0, 0.0)

    def test_arms_hang_slightly_bent_at_rest(self):
        pose = walk_pose(0.0, 0.0)
        assert pose.rotations["forearm.L"][0] > 0.0


class TestSerialisation:
    def test_the_pose_serialises_to_plain_json_types(self):
        payload = walk_pose(1.0, WALKING_SPEED).as_dict()
        assert isinstance(payload["rotations"]["thigh.L"], list)
        assert len(payload["root_offset"]) == 3

    def test_serialised_angles_are_rounded_but_not_flattened(self):
        payload = walk_pose(0.3, WALKING_SPEED).as_dict()
        assert payload["rotations"]["thigh.L"][0] != 0.0
