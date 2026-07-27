"""The actor skeleton, defined once and shared by the rig builder and the gait.

Blender builds bones from this and the gait poses them by name. Two definitions would
drift, and a drifted skeleton bends an elbow where a knee should be — so there is one,
and it lives here rather than in Blender because it is geometry, not scene construction.

Coordinate convention, which everything downstream depends on:

  * origin at the feet, so a plan waypoint at ground level places the figure standing
  * +Z up
  * **+Y is the direction the actor faces at heading zero**, which is away from the
    camera — the camera looks along +Y (see `camera_basis`)
  * +X is the actor's left

Proportions are canonical adult ratios for a 1.75 m figure and are scaled per actor.
"""

import math
from dataclasses import dataclass


REFERENCE_HEIGHT_METERS = 1.75

# Left-side bones are mirrored to the right automatically, so the table stays readable
# and the two sides cannot accidentally differ.
MIRRORED_SUFFIX = ".L"
MIRROR_SUFFIX = ".R"


@dataclass(frozen=True)
class BoneSpec:
    """One bone in the rest pose. `head` is the joint; `tail` is where it points."""

    name: str
    parent: str | None
    head: tuple[float, float, float]
    tail: tuple[float, float, float]

    @property
    def length(self) -> float:
        return math.dist(self.head, self.tail)

    def scaled(self, scale: float) -> "BoneSpec":
        return BoneSpec(
            self.name,
            self.parent,
            tuple(value * scale for value in self.head),
            tuple(value * scale for value in self.tail),
        )


# Rest pose in metres for a 1.75 m figure standing with arms down.
_CORE_BONES = (
    BoneSpec("pelvis", None, (0.0, 0.0, 0.94), (0.0, 0.0, 1.14)),
    BoneSpec("chest", "pelvis", (0.0, 0.0, 1.14), (0.0, 0.0, 1.42)),
    BoneSpec("neck", "chest", (0.0, 0.0, 1.42), (0.0, 0.0, 1.54)),
    BoneSpec("head", "neck", (0.0, 0.0, 1.54), (0.0, 0.0, 1.75)),
)

_LEFT_BONES = (
    # Shoulders are set wide enough that the arms hang clear of the torso rather than
    # inside it — an arm embedded in the chest reads as a figure with no arms at all.
    BoneSpec("clavicle.L", "chest", (0.03, 0.0, 1.39), (0.19, 0.0, 1.41)),
    BoneSpec("upper_arm.L", "clavicle.L", (0.21, 0.0, 1.40), (0.23, 0.0, 1.13)),
    BoneSpec("forearm.L", "upper_arm.L", (0.23, 0.0, 1.13), (0.24, 0.0, 0.88)),
    BoneSpec("hand.L", "forearm.L", (0.24, 0.0, 0.88), (0.24, 0.0, 0.77)),
    BoneSpec("thigh.L", "pelvis", (0.09, 0.0, 0.92), (0.09, 0.0, 0.50)),
    BoneSpec("shin.L", "thigh.L", (0.09, 0.0, 0.50), (0.09, 0.0, 0.09)),
    BoneSpec("foot.L", "shin.L", (0.09, 0.0, 0.09), (0.09, 0.14, 0.03)),
)


def _mirrored(bone: BoneSpec) -> BoneSpec:
    def flip(point):
        return (-point[0], point[1], point[2])

    parent = bone.parent
    if parent is not None and parent.endswith(MIRRORED_SUFFIX):
        parent = parent[: -len(MIRRORED_SUFFIX)] + MIRROR_SUFFIX
    return BoneSpec(
        bone.name[: -len(MIRRORED_SUFFIX)] + MIRROR_SUFFIX,
        parent,
        flip(bone.head),
        flip(bone.tail),
    )


SKELETON: tuple[BoneSpec, ...] = (
    _CORE_BONES + _LEFT_BONES + tuple(_mirrored(bone) for bone in _LEFT_BONES)
)

BONE_NAMES = tuple(bone.name for bone in SKELETON)
POSED_BONE_NAMES = tuple(
    name for name in BONE_NAMES if name.startswith((
        "thigh", "shin", "foot", "upper_arm", "forearm", "chest", "pelvis", "head",
    ))
)


def skeleton_for_height(height_meters: float) -> tuple[BoneSpec, ...]:
    """Scale the canonical skeleton to an actor's estimated height."""
    scale = float(height_meters) / REFERENCE_HEIGHT_METERS
    return tuple(bone.scaled(scale) for bone in SKELETON)


def bone_by_name(skeleton: tuple[BoneSpec, ...] = SKELETON) -> dict[str, BoneSpec]:
    return {bone.name: bone for bone in skeleton}


def validate_skeleton(skeleton: tuple[BoneSpec, ...] = SKELETON) -> None:
    """Every parent must exist, no bone may be zero length, nothing below the ground."""
    known: set[str] = set()
    for bone in skeleton:
        if bone.parent is not None and bone.parent not in known:
            raise ValueError(f"Bone {bone.name} names an unknown parent {bone.parent}")
        if bone.length <= 0.0:
            raise ValueError(f"Bone {bone.name} has zero length")
        if min(bone.head[2], bone.tail[2]) < 0.0:
            raise ValueError(f"Bone {bone.name} extends below the ground plane")
        known.add(bone.name)
