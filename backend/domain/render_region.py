"""Screen region occupied by a gap's actors (Implementation_plan.md §5.2).

M2 renders only the detected entity classes, cropped to the rectangle they actually
occupy. On the M0 benchmark that crop alone accounted for most of the speed-up, because
Blender stops shading a mostly-empty frame.

The region is part of the render contract, not an internal detail: the compositor uses
it to place the actor layer back onto the plate. An off-by-one here puts actors in the
wrong part of the frame, so the arithmetic lives in one tested place.

Pure geometry — no `bpy`, no I/O.
"""

from domain.actor_proxies import bounding_half_extents
from domain.camera_projection import supports_projection, world_point_to_image


# The crop must contain the whole actor however it is oriented, and an object rotated to
# face its direction of travel sweeps its longer axis into the shorter one. Using the
# larger half-extent on both horizontal axes covers every heading without needing to know
# it. A slightly generous region costs a little render time; a tight one clips a bumper.
BOUNDS_SAFETY_MARGIN = 1.12

# Padding as a fraction of frame size, so soft shadows and outlines are not clipped.
REGION_PADDING_FRACTION = 0.04

# Above this coverage the crop stops paying for itself and the bookkeeping risk is not
# worth it, so the renderer falls back to the full frame (§5.2).
FULL_FRAME_COVERAGE_THRESHOLD = 0.60

MINIMUM_REGION_SPAN = 1e-4


class RenderRegion:
    """Normalized crop rectangle, in Blender's bottom-left origin convention."""

    def __init__(
        self, minimum_x: float, minimum_y: float, maximum_x: float, maximum_y: float,
    ) -> None:
        self.minimum_x = minimum_x
        self.minimum_y = minimum_y
        self.maximum_x = maximum_x
        self.maximum_y = maximum_y

    @property
    def coverage(self) -> float:
        return (self.maximum_x - self.minimum_x) * (self.maximum_y - self.minimum_y)

    @property
    def is_full_frame(self) -> bool:
        return (
            self.minimum_x <= 0.0 and self.minimum_y <= 0.0
            and self.maximum_x >= 1.0 and self.maximum_y >= 1.0
        )

    def pixel_box(self, frame_width: int, frame_height: int) -> tuple[int, int, int, int]:
        """Crop in image pixels with a top-left origin, as the compositor needs it.

        Blender's border origin is bottom-left, so the vertical axis is flipped here
        exactly once — the single place that conversion happens.
        """
        left = int(round(self.minimum_x * frame_width))
        right = int(round(self.maximum_x * frame_width))
        top = int(round((1.0 - self.maximum_y) * frame_height))
        bottom = int(round((1.0 - self.minimum_y) * frame_height))
        return (
            max(0, min(frame_width - 1, left)),
            max(0, min(frame_height - 1, top)),
            max(1, min(frame_width, right)),
            max(1, min(frame_height, bottom)),
        )

    def as_dict(self) -> dict:
        return {
            "minimum_x": round(self.minimum_x, 6),
            "minimum_y": round(self.minimum_y, 6),
            "maximum_x": round(self.maximum_x, 6),
            "maximum_y": round(self.maximum_y, 6),
            "coverage": round(self.coverage, 6),
        }

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RenderRegion):
            return NotImplemented
        return self.as_dict() == other.as_dict()

    def __repr__(self) -> str:
        return (
            f"RenderRegion({self.minimum_x:.4f}, {self.minimum_y:.4f}, "
            f"{self.maximum_x:.4f}, {self.maximum_y:.4f})"
        )


FULL_FRAME_REGION = RenderRegion(0.0, 0.0, 1.0, 1.0)


def entity_bounds(kind: str) -> tuple[float, float, float]:
    """Half-length, half-width and top height that must fit inside the crop."""
    half_length, half_width, top = bounding_half_extents(kind)
    horizontal = max(half_length, half_width) * BOUNDS_SAFETY_MARGIN
    return horizontal, horizontal, top * BOUNDS_SAFETY_MARGIN


def world_position_at_frame(entity: dict, frame: int) -> list[float] | None:
    """Interpolate the validated path. Never extrapolates beyond its waypoints."""
    waypoints = entity.get("path_prediction", {}).get("waypoints") or []
    ordered = sorted(
        (point for point in waypoints if "frame" in point and "world" in point),
        key=lambda point: int(point["frame"]),
    )
    if not ordered:
        return None
    if frame <= int(ordered[0]["frame"]):
        return [float(value) for value in ordered[0]["world"]]
    if frame >= int(ordered[-1]["frame"]):
        return [float(value) for value in ordered[-1]["world"]]
    return _interpolate_between_waypoints(ordered, frame)


def _interpolate_between_waypoints(ordered: list[dict], frame: int) -> list[float]:
    for earlier, later in zip(ordered, ordered[1:]):
        earlier_frame = int(earlier["frame"])
        later_frame = int(later["frame"])
        if not earlier_frame <= frame <= later_frame:
            continue
        span = max(1, later_frame - earlier_frame)
        ratio = (frame - earlier_frame) / span
        return [
            float(earlier["world"][axis])
            + (float(later["world"][axis]) - float(earlier["world"][axis])) * ratio
            for axis in range(3)
        ]
    return [float(value) for value in ordered[-1]["world"]]


def _bounding_corners(position: list[float], kind: str) -> list[tuple[float, float, float]]:
    half_length, half_width, height = entity_bounds(kind)
    return [
        (position[0] + x_offset, position[1] + y_offset, position[2] + z_offset)
        for x_offset in (-half_length, half_length)
        for y_offset in (-half_width, half_width)
        for z_offset in (0.0, height)
    ]


def entity_screen_span(
    entity: dict,
    frame: int,
    frame_width: int,
    frame_height: int,
    camera_contract: dict,
) -> tuple[float, float, float, float] | None:
    """Normalized screen box for one entity, or None when it is not in front of the camera."""
    if not supports_projection(camera_contract):
        return None
    position = world_position_at_frame(entity, frame)
    if position is None:
        return None
    projected = [
        world_point_to_image(corner, frame_width, frame_height, camera_contract)
        for corner in _bounding_corners(position, entity.get("kind", ""))
    ]
    visible = [point for point in projected if point is not None]
    if not visible:
        return None
    return (
        min(point[0] for point in visible) / frame_width,
        # Screen y grows downward; the region uses Blender's bottom-left origin.
        1.0 - max(point[1] for point in visible) / frame_height,
        max(point[0] for point in visible) / frame_width,
        1.0 - min(point[1] for point in visible) / frame_height,
    )


def gap_render_region(
    entities: list[dict],
    frame: int,
    frame_width: int,
    frame_height: int,
    camera_contract: dict,
    padding_fraction: float = REGION_PADDING_FRACTION,
    coverage_threshold: float = FULL_FRAME_COVERAGE_THRESHOLD,
) -> RenderRegion:
    """Union of every entity's screen box, padded and clamped to the frame.

    Falls back to the full frame when nothing projects or when the union is large
    enough that cropping no longer saves meaningful work.
    """
    spans = [
        span for span in (
            entity_screen_span(entity, frame, frame_width, frame_height, camera_contract)
            for entity in entities
        )
        if span is not None
    ]
    if not spans:
        return FULL_FRAME_REGION
    region = RenderRegion(
        max(0.0, min(span[0] for span in spans) - padding_fraction),
        max(0.0, min(span[1] for span in spans) - padding_fraction),
        min(1.0, max(span[2] for span in spans) + padding_fraction),
        min(1.0, max(span[3] for span in spans) + padding_fraction),
    )
    if (
        region.maximum_x - region.minimum_x < MINIMUM_REGION_SPAN
        or region.maximum_y - region.minimum_y < MINIMUM_REGION_SPAN
        or region.coverage > coverage_threshold
    ):
        return FULL_FRAME_REGION
    return region
