"""Turns a measured speed and distance travelled into joint angles.

Kept out of Blender deliberately. Blender applies rotations; it does not decide them.
That means the whole gait is unit-testable without launching a render, which matters
because a limb swinging the wrong way is obvious to a viewer and invisible to a
structural assertion.

**What this is and is not.** The evidence supports a position and a heading — YOLO pose
is sampled only at gap boundaries, not through the hidden interval. So the cycle here is
*synthesised* from the measured ground speed, not observed. It is presentation: it makes
the figure read as a walking person rather than a sliding statue. It is not a claim
about how the subject moved their limbs, and the report labels it as such.

The gait is phase-locked to **distance travelled**, not to time. A figure whose feet
slide because its stride is driven by a clock rather than by the ground is the single
most recognisable animation error there is, and locking to distance makes it impossible.
"""

import math
from dataclasses import dataclass, field


# Below this the subject is standing, not walking, and the cycle is frozen.
IDLE_SPEED_METERS_PER_SECOND = 0.15
# Above this the cycle switches to a run: longer stride, higher knee, bigger arm swing.
RUN_SPEED_METERS_PER_SECOND = 2.6

# Stride is the distance covered by one full cycle, i.e. two steps. Humans lengthen
# their stride with speed rather than only stepping faster, so this is a fit, not a
# constant, bounded to keep implausible speeds from producing implausible strides.
STRIDE_BASE_METERS = 0.62
STRIDE_PER_SPEED_SECONDS = 0.72
STRIDE_LIMITS_METERS = (0.55, 2.40)

THIGH_SWING_BASE_RADIANS = 0.20
THIGH_SWING_PER_SPEED = 0.17
THIGH_SWING_LIMITS = (0.14, 0.80)

KNEE_BEND_BASE_RADIANS = 0.42
KNEE_BEND_PER_SPEED = 0.34
KNEE_BEND_LIMITS = (0.30, 1.45)
# Knee flexion peaks after the leg passes under the body, not at the extremes.
KNEE_PHASE_OFFSET = 0.62

ARM_SWING_RATIO = 0.62
# Arms hang slightly bent and slightly out from the body even at rest.
ELBOW_REST_RADIANS = 0.16
ELBOW_SWING_RATIO = 0.30
SHOULDER_SPREAD_RADIANS = 0.09

ANKLE_SWING_RATIO = 0.35

# The pelvis dips twice per cycle, lowest when the legs are furthest apart.
PELVIS_BOB_PER_SPEED = 0.013
PELVIS_BOB_LIMITS = (0.004, 0.045)
# And rolls slightly toward the supporting leg.
PELVIS_ROLL_PER_SPEED = 0.020
PELVIS_ROLL_LIMIT = 0.075

# A walking figure leans very slightly into its direction of travel.
LEAN_PER_SPEED_RADIANS = 0.022
LEAN_LIMIT_RADIANS = 0.13

# Idle figures are not statues; a small sway keeps them from reading as frozen.
IDLE_SWAY_RADIANS = 0.012
IDLE_SWAY_CYCLES_PER_SECOND = 0.28


def _clamp(value: float, limits: tuple[float, float]) -> float:
    return max(limits[0], min(limits[1], value))


@dataclass(frozen=True)
class JointPose:
    """Euler XYZ rotation in radians per bone, plus a root offset in metres."""

    rotations: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    root_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def as_dict(self) -> dict:
        return {
            "rotations": {
                name: [round(angle, 5) for angle in rotation]
                for name, rotation in self.rotations.items()
            },
            "root_offset": [round(value, 5) for value in self.root_offset],
        }


def stride_length(speed_meters_per_second: float) -> float:
    """Distance covered by one full two-step cycle at this speed."""
    return _clamp(
        STRIDE_BASE_METERS + STRIDE_PER_SPEED_SECONDS * max(0.0, speed_meters_per_second),
        STRIDE_LIMITS_METERS,
    )


def cycle_phase(distance_meters: float, speed_meters_per_second: float) -> float:
    """Where in the walk cycle the figure is, from how far it has travelled.

    Phase-locking to distance is what prevents foot sliding: the feet advance exactly as
    fast as the body does, whatever the frame rate or the render sample spacing.
    """
    return (float(distance_meters) / stride_length(speed_meters_per_second)) % 1.0


def is_walking(speed_meters_per_second: float) -> bool:
    return float(speed_meters_per_second) >= IDLE_SPEED_METERS_PER_SECOND


def _swing_amplitudes(speed: float) -> tuple[float, float, float]:
    thigh = _clamp(
        THIGH_SWING_BASE_RADIANS + THIGH_SWING_PER_SPEED * speed, THIGH_SWING_LIMITS,
    )
    knee = _clamp(KNEE_BEND_BASE_RADIANS + KNEE_BEND_PER_SPEED * speed, KNEE_BEND_LIMITS)
    return thigh, knee, thigh * ARM_SWING_RATIO


def _leg_pose(
    phase: float, thigh_swing: float, knee_bend: float, side_offset: float,
) -> dict[str, tuple[float, float, float]]:
    """One leg's thigh, shin and foot at this phase.

    The thigh swings as a cosine. The knee only ever bends one way — it is a hinge, and
    letting it go negative is the classic broken-leg artifact — so flexion is a
    rectified sine offset to peak during the swing phase.
    """
    angle = 2.0 * math.pi * (phase + side_offset)
    thigh = thigh_swing * math.cos(angle)
    flexion = knee_bend * max(0.0, math.sin(2.0 * math.pi * (phase + side_offset + KNEE_PHASE_OFFSET)))
    # The ankle counter-rotates so the sole stays roughly level through the step.
    ankle = -ANKLE_SWING_RATIO * (thigh + flexion)
    return {"thigh": (thigh, 0.0, 0.0), "shin": (flexion, 0.0, 0.0), "foot": (ankle, 0.0, 0.0)}


def _arm_pose(
    phase: float, arm_swing: float, side_offset: float, side_sign: float,
) -> dict[str, tuple[float, float, float]]:
    """One arm at this phase.

    The caller passes an offset already shifted half a cycle from the arm's own leg,
    which is what makes the left arm travel with the right one. The swing itself is a
    plain cosine — negating it here as well would cancel that shift out and put each arm
    back in step with the leg beneath it.
    """
    angle = 2.0 * math.pi * (phase + side_offset)
    shoulder = arm_swing * math.cos(angle)
    elbow = ELBOW_REST_RADIANS + ELBOW_SWING_RATIO * arm_swing * (1.0 - math.cos(angle))
    return {
        "upper_arm": (shoulder, 0.0, side_sign * SHOULDER_SPREAD_RADIANS),
        "forearm": (elbow, 0.0, 0.0),
    }


def walk_pose(
    distance_meters: float,
    speed_meters_per_second: float,
    elapsed_seconds: float = 0.0,
) -> JointPose:
    """Full-body pose for a figure that has travelled this far at this speed."""
    speed = max(0.0, float(speed_meters_per_second))
    if not is_walking(speed):
        return _idle_pose(elapsed_seconds)
    phase = cycle_phase(distance_meters, speed)
    thigh_swing, knee_bend, arm_swing = _swing_amplitudes(speed)
    rotations: dict[str, tuple[float, float, float]] = {}
    for suffix, offset, sign in ((".L", 0.0, 1.0), (".R", 0.5, -1.0)):
        for joint, rotation in _leg_pose(phase, thigh_swing, knee_bend, offset).items():
            rotations[joint + suffix] = rotation
        # The arm offset is shifted half a cycle from its own leg, so the left arm
        # travels with the right leg.
        for joint, rotation in _arm_pose(phase, arm_swing, offset + 0.5, sign).items():
            rotations[joint + suffix] = rotation
    angle = 2.0 * math.pi * phase
    bob = _clamp(PELVIS_BOB_PER_SPEED * speed, PELVIS_BOB_LIMITS)
    roll = min(PELVIS_ROLL_LIMIT, PELVIS_ROLL_PER_SPEED * speed) * math.sin(angle)
    lean = min(LEAN_LIMIT_RADIANS, LEAN_PER_SPEED_RADIANS * speed)
    rotations["pelvis"] = (0.0, roll, 0.0)
    rotations["chest"] = (-lean, -roll * 0.6, 0.0)
    rotations["head"] = (lean * 0.5, 0.0, 0.0)
    return JointPose(
        rotations=rotations,
        root_offset=(0.0, 0.0, -bob * abs(math.cos(angle))),
    )


def _idle_pose(elapsed_seconds: float) -> JointPose:
    """Standing, with a slow sway so the figure does not read as a frozen mannequin."""
    sway = IDLE_SWAY_RADIANS * math.sin(
        2.0 * math.pi * IDLE_SWAY_CYCLES_PER_SECOND * float(elapsed_seconds)
    )
    return JointPose(
        rotations={
            "thigh.L": (0.0, 0.0, 0.0),
            "thigh.R": (0.0, 0.0, 0.0),
            "shin.L": (0.0, 0.0, 0.0),
            "shin.R": (0.0, 0.0, 0.0),
            "foot.L": (0.0, 0.0, 0.0),
            "foot.R": (0.0, 0.0, 0.0),
            "upper_arm.L": (0.0, 0.0, SHOULDER_SPREAD_RADIANS),
            "upper_arm.R": (0.0, 0.0, -SHOULDER_SPREAD_RADIANS),
            "forearm.L": (ELBOW_REST_RADIANS, 0.0, 0.0),
            "forearm.R": (ELBOW_REST_RADIANS, 0.0, 0.0),
            "pelvis": (0.0, sway, 0.0),
            "chest": (0.0, -sway * 0.5, 0.0),
            "head": (0.0, sway * 0.4, 0.0),
        },
        root_offset=(0.0, 0.0, 0.0),
    )
