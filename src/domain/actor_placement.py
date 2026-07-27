"""Where an entity sits in the picture, and which observation of it to draw there.

The reconstruction already decides *where* an entity is in the world at every moment of
a gap. Projecting that through the same camera the plate was calibrated for gives a
footprint in pixels: where the feet land, and how tall the figure should be. That is all
the geometry a composited actor needs.

Which picture to draw is the other half. Every entity in a gap was observed, repeatedly,
in the visible footage on either side of it — walking, at many scales, from several
angles. Those observations are real photographs of the real subject under the real light
of the real scene, so using them sidesteps everything that makes a synthetic figure read
as synthetic: clothing, skin, hair, sensor grain, motion blur, and the fact that the
camera and the renderer never quite agree about light.

Matching is done in image space rather than by inferring a 3-D facing direction. A
monocular heading estimate for a person forty pixels tall is mostly noise, whereas which
way they are moving across the frame, and how big they are, are both measured directly.
So an observation is chosen by how closely its apparent motion and apparent size match
what the plan predicts for the frame being filled.
"""

import math
from dataclasses import dataclass

from domain.camera_projection import supports_projection, world_point_to_image
from domain.render_region import entity_bounds, world_position_at_frame


# How much each term counts when picking an observation.
DIRECTION_WEIGHT = 1.0
SCALE_WEIGHT = 0.45
# Penalty for reusing the observation drawn on the previous sample. Without it the same
# photograph is chosen for every frame of a gap and the figure glides along frozen, which
# is the single clearest tell that a composite is not real footage.
REPEAT_PENALTY = 0.8
# An observation smaller than this fraction of the size it must be drawn at is upscaled
# past the point where it holds together.
MINIMUM_SCALE_RATIO = 0.35
MAXIMUM_SCALE_RATIO = 4.0
# Below this the entity is too small on screen to be worth compositing at all.
MINIMUM_DRAWN_HEIGHT_PIXELS = 8.0


@dataclass(frozen=True)
class Placement:
    """Where one entity should be drawn on one frame."""

    centre_x: float
    foot_y: float
    pixel_height: float
    velocity_x: float
    velocity_y: float

    @property
    def is_drawable(self) -> bool:
        return self.pixel_height >= MINIMUM_DRAWN_HEIGHT_PIXELS


@dataclass(frozen=True)
class Observation:
    """One real sighting of an entity in visible footage."""

    source_frame: int
    left: float
    top: float
    right: float
    bottom: float

    @property
    def pixel_height(self) -> float:
        return max(1.0, self.bottom - self.top)

    @property
    def centre_x(self) -> float:
        return (self.left + self.right) / 2.0


def _project(entity: dict, frame: int, width: int, height: int, camera: dict):
    """Foot point and pixel height for an entity at one frame, or None."""
    ground = world_position_at_frame(entity, frame)
    if ground is None:
        return None
    foot = world_point_to_image(ground, width, height, camera)
    if foot is None:
        return None
    _, _, entity_height = entity_bounds(str(entity.get("kind", "person")))
    head_world = [float(ground[0]), float(ground[1]), float(ground[2]) + entity_height]
    head = world_point_to_image(head_world, width, height, camera)
    if head is None:
        return None
    return foot, abs(foot[1] - head[1])


def placement_for_frame(
    entity: dict, frame: int, width: int, height: int, camera: dict,
) -> Placement | None:
    """Project one entity onto one frame of the gap.

    Velocity is measured from the projected position a frame either side rather than
    from the world path, so it is directly comparable with the apparent motion measured
    from a detection box.
    """
    if not supports_projection(camera):
        return None
    current = _project(entity, frame, width, height, camera)
    if current is None:
        return None
    (centre_x, foot_y), pixel_height = current
    before = _project(entity, frame - 1, width, height, camera)
    after = _project(entity, frame + 1, width, height, camera)
    if before is not None and after is not None:
        velocity = (
            (after[0][0] - before[0][0]) / 2.0, (after[0][1] - before[0][1]) / 2.0,
        )
    elif after is not None:
        velocity = (after[0][0] - centre_x, after[0][1] - foot_y)
    elif before is not None:
        velocity = (centre_x - before[0][0], foot_y - before[0][1])
    else:
        velocity = (0.0, 0.0)
    return Placement(
        centre_x=centre_x, foot_y=foot_y, pixel_height=pixel_height,
        velocity_x=velocity[0], velocity_y=velocity[1],
    )


def observation_velocities(
    observations: list[Observation],
) -> list[tuple[float, float]]:
    """Apparent motion at each sighting, from its neighbours in time."""
    velocities: list[tuple[float, float]] = []
    for index, observation in enumerate(observations):
        previous = observations[index - 1] if index > 0 else None
        following = observations[index + 1] if index + 1 < len(observations) else None
        if previous is not None and following is not None:
            span = max(1, following.source_frame - previous.source_frame)
            velocities.append((
                (following.centre_x - previous.centre_x) / span,
                (following.bottom - previous.bottom) / span,
            ))
        elif following is not None:
            span = max(1, following.source_frame - observation.source_frame)
            velocities.append((
                (following.centre_x - observation.centre_x) / span,
                (following.bottom - observation.bottom) / span,
            ))
        elif previous is not None:
            span = max(1, observation.source_frame - previous.source_frame)
            velocities.append((
                (observation.centre_x - previous.centre_x) / span,
                (observation.bottom - previous.bottom) / span,
            ))
        else:
            velocities.append((0.0, 0.0))
    return velocities


def _direction_agreement(
    first: tuple[float, float], second: tuple[float, float],
) -> float:
    """Cosine similarity of two image-space velocities, mapped to 0..1.

    A stationary observation matches anything equally: with no motion there is no facing
    to disagree about, so it scores neutrally rather than being ranked last.
    """
    first_magnitude = math.hypot(*first)
    second_magnitude = math.hypot(*second)
    if first_magnitude < 1e-6 or second_magnitude < 1e-6:
        return 0.5
    cosine = (first[0] * second[0] + first[1] * second[1]) / (
        first_magnitude * second_magnitude
    )
    return (max(-1.0, min(1.0, cosine)) + 1.0) / 2.0


def _scale_agreement(observed_height: float, wanted_height: float) -> float:
    """How close an observation is to the size it must be drawn at.

    Prefers the sighting needing least resampling: a figure captured at roughly the size
    it will be drawn keeps its detail, where one blown up from a distant sighting turns
    to mush.
    """
    if observed_height <= 0.0 or wanted_height <= 0.0:
        return 0.0
    ratio = observed_height / wanted_height
    if not MINIMUM_SCALE_RATIO <= ratio <= MAXIMUM_SCALE_RATIO:
        return 0.0
    return 1.0 / (1.0 + abs(math.log(ratio)))


def choose_observation(
    observations: list[Observation],
    velocities: list[tuple[float, float]],
    placement: Placement,
    previous_index: int | None = None,
) -> int | None:
    """Index of the sighting to draw for this frame, or None if none is usable."""
    best_index, best_score = None, 0.0
    for index, observation in enumerate(observations):
        scale_score = _scale_agreement(observation.pixel_height, placement.pixel_height)
        if scale_score <= 0.0:
            continue
        score = (
            DIRECTION_WEIGHT * _direction_agreement(
                velocities[index], (placement.velocity_x, placement.velocity_y),
            )
            + SCALE_WEIGHT * scale_score
        )
        if index == previous_index:
            score -= REPEAT_PENALTY
        if best_index is None or score > best_score:
            best_index, best_score = index, score
    return best_index
