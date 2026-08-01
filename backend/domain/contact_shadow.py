"""Where each actor's contact shadow lands on the ground.

A composited figure with no shadow floats, whatever its geometry does — it is the first
thing a viewer's eye flags and the standard first note in VFX integration. Blender's own
shadow catcher is a Cycles feature, so under EEVEE there is nothing to catch shadows on
and the actor arrives with none.

Rather than switch engines for one effect, the shadow is computed here from geometry we
already have: the actor's ground position, its footprint from the proxy catalog, and the
same camera that placed it. Sampling the footprint ellipse on the ground plane and
projecting those points gives the shadow's true perspective shape — it stretches and
flattens with distance exactly as the ground does, which a fixed oval would not.

Pure geometry: no pixels, no `bpy`.
"""

import math

from domain.actor_proxies import proxy_for
from domain.camera_projection import supports_projection, world_point_to_image


# Points sampled around the footprint. Twelve is enough for a smooth ellipse once the
# projected bounding box is taken.
FOOTPRINT_SAMPLES = 12
# The shadow spreads slightly wider than the object that casts it, because the light is
# not a point and the contact is not a knife edge.
FOOTPRINT_SPREAD = 1.15
# Shadows are never sharper than this many pixels, or they read as a decal.
MINIMUM_BLUR_PIXELS = 3.0
BLUR_FRACTION_OF_RADIUS = 0.42
# A contact shadow is darkest under the object and fades outward; this is the peak.
DEFAULT_SHADOW_STRENGTH = 0.42
# Beyond this the actor is so far away the shadow is sub-pixel and only adds noise.
MINIMUM_SHADOW_RADIUS_PIXELS = 1.5


class ShadowEllipse:
    """A projected ground shadow, in image pixels with a top-left origin."""

    def __init__(
        self,
        center_x: float,
        center_y: float,
        radius_x: float,
        radius_y: float,
        strength: float = DEFAULT_SHADOW_STRENGTH,
    ) -> None:
        self.center_x = center_x
        self.center_y = center_y
        self.radius_x = radius_x
        self.radius_y = radius_y
        self.strength = strength

    @property
    def blur_pixels(self) -> float:
        return max(MINIMUM_BLUR_PIXELS, min(self.radius_x, self.radius_y) * BLUR_FRACTION_OF_RADIUS)

    @property
    def is_visible(self) -> bool:
        return min(self.radius_x, self.radius_y) >= MINIMUM_SHADOW_RADIUS_PIXELS

    def as_dict(self) -> dict:
        return {
            "center": [round(self.center_x, 2), round(self.center_y, 2)],
            "radii": [round(self.radius_x, 2), round(self.radius_y, 2)],
            "strength": round(self.strength, 4),
        }

    def __eq__(self, other: object) -> bool:
        return isinstance(other, ShadowEllipse) and self.as_dict() == other.as_dict()

    def __repr__(self) -> str:
        return (
            f"ShadowEllipse(({self.center_x:.1f}, {self.center_y:.1f}), "
            f"r=({self.radius_x:.1f}, {self.radius_y:.1f}))"
        )


def footprint_shadow(
    ground_position,
    class_name: str,
    frame_width: int,
    frame_height: int,
    camera_contract: dict,
    strength: float = DEFAULT_SHADOW_STRENGTH,
) -> ShadowEllipse | None:
    """Project one actor's footprint onto the image, or None when it cannot be seen."""
    if not supports_projection(camera_contract):
        return None
    specification = proxy_for(class_name)
    half_length, half_width = specification.half_extents
    radius = max(half_length, half_width) * FOOTPRINT_SPREAD
    projected = []
    for index in range(FOOTPRINT_SAMPLES):
        angle = 2.0 * math.pi * index / FOOTPRINT_SAMPLES
        point = (
            float(ground_position[0]) + radius * math.cos(angle),
            float(ground_position[1]) + radius * math.sin(angle),
            0.0,
        )
        image_point = world_point_to_image(point, frame_width, frame_height, camera_contract)
        if image_point is not None:
            projected.append(image_point)
    if len(projected) < 3:
        return None
    left = min(point[0] for point in projected)
    right = max(point[0] for point in projected)
    top = min(point[1] for point in projected)
    bottom = max(point[1] for point in projected)
    return ShadowEllipse(
        center_x=(left + right) / 2.0,
        center_y=(top + bottom) / 2.0,
        radius_x=(right - left) / 2.0,
        radius_y=(bottom - top) / 2.0,
        strength=strength,
    )


def gap_contact_shadows(
    entities: list[dict],
    positions: dict,
    frame_width: int,
    frame_height: int,
    camera_contract: dict,
    strength: float = DEFAULT_SHADOW_STRENGTH,
) -> list[ShadowEllipse]:
    """Every visible actor's shadow for one frame.

    `positions` maps entity id to its ground position, so the caller supplies exactly the
    positions it placed the actors at rather than this module re-deriving them and
    risking a different answer.
    """
    shadows = []
    for entity in entities:
        position = positions.get(str(entity.get("id")))
        if position is None:
            continue
        shadow = footprint_shadow(
            position, str(entity.get("kind", "")), frame_width, frame_height,
            camera_contract, strength,
        )
        if shadow is not None and shadow.is_visible:
            shadows.append(shadow)
    return shadows
