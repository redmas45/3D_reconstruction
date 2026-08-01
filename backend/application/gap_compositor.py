"""Composites rendered actors onto the recovered plate (§5.4).

Order matters and is fixed: plate, then shadow, then graded actors, then 2D overlays.
Shadow before actors because a contact shadow belongs under the feet that cast it;
grading before the alpha-over because the actor must be corrected to the plate, never
the other way round.

**Grade matching** (§5.4 step 3) is the cheapest believability lever in the system and
the old renderer had nothing like it. A rendered actor carries the lighting of its
synthetic key light; the plate carries the real scene's exposure and colour cast. Left
uncorrected the actor reads as pasted on no matter how good the geometry is. Matching
the actor's luminance and colour statistics to the plate region it occupies costs a few
milliseconds of numpy and does more for believability than extra render samples.

Runs on CPU while the GPU renders the next frame, so it is off the critical path.
"""

import cv2
import numpy

from domain.contact_shadow import ShadowEllipse
from domain.render_region import RenderRegion


# How strongly the actor layer is pulled toward the plate's colour statistics. Full
# correction looks wrong: actors legitimately differ from the average background.
GRADE_STRENGTH = 0.55
# Below this the plate region is too uniform for its statistics to mean anything.
MINIMUM_GRADE_DEVIATION = 1e-3
GRADE_GAIN_LIMITS = (0.75, 1.35)

SHADOW_STRENGTH = 0.55
ALPHA_MAXIMUM = 255.0

# Grain is matched at slightly under the plate's own level: overshooting is worse than
# undershooting, because visible noise on the actor alone draws the eye straight to it.
GRAIN_MATCH_RATIO = 0.85
GRAIN_MEASURE_BLUR_SIGMA = 1.2
MINIMUM_GRAIN_SIGMA = 0.6
MAXIMUM_GRAIN_SIGMA = 12.0
# One pixel of matte softening, to match lens and codec softness in the plate.
EDGE_SOFTEN_SIGMA = 0.9

RGB_CHANNELS = 3
RGBA_CHANNELS = 4


class CompositeError(ValueError):
    """Layers could not be composited as supplied."""


def _validate_layer(layer: numpy.ndarray, expected_channels: int, name: str) -> None:
    if layer.ndim != 3 or layer.shape[2] != expected_channels:
        raise CompositeError(
            f"{name} must have {expected_channels} channels, got shape {layer.shape}"
        )


def region_slice(
    plate: numpy.ndarray, region: RenderRegion,
) -> tuple[slice, slice]:
    height, width = plate.shape[:2]
    left, top, right, bottom = region.pixel_box(width, height)
    return slice(top, bottom), slice(left, right)


def match_grade(
    actor_rgb: numpy.ndarray,
    actor_alpha: numpy.ndarray,
    plate_region: numpy.ndarray,
    strength: float = GRADE_STRENGTH,
) -> numpy.ndarray:
    """Pull the actor's per-channel statistics toward the plate's.

    Only pixels the actor actually covers contribute to its statistics — including the
    transparent surround would bias every measurement toward zero.
    """
    coverage = actor_alpha > 0
    if not coverage.any():
        return actor_rgb
    graded = actor_rgb.astype(numpy.float32).copy()
    plate_pixels = plate_region.reshape(-1, RGB_CHANNELS).astype(numpy.float32)
    for channel in range(RGB_CHANNELS):
        graded[..., channel] = _match_channel(
            graded[..., channel], coverage, plate_pixels[:, channel], strength,
        )
    return numpy.clip(graded, 0, 255).astype(numpy.uint8)


def _match_channel(
    channel_values: numpy.ndarray,
    coverage: numpy.ndarray,
    plate_channel: numpy.ndarray,
    strength: float,
) -> numpy.ndarray:
    covered = channel_values[coverage]
    actor_mean = float(covered.mean())
    actor_deviation = float(covered.std())
    plate_mean = float(plate_channel.mean())
    plate_deviation = float(plate_channel.std())
    if actor_deviation < MINIMUM_GRADE_DEVIATION or plate_deviation < MINIMUM_GRADE_DEVIATION:
        # Too flat to infer a gain from; shift the level only.
        offset = (plate_mean - actor_mean) * strength
        adjusted = channel_values + offset
    else:
        gain = numpy.clip(plate_deviation / actor_deviation, *GRADE_GAIN_LIMITS)
        target = (channel_values - actor_mean) * gain + plate_mean
        adjusted = channel_values + (target - channel_values) * strength
    result = channel_values.copy()
    result[coverage] = adjusted[coverage]
    return result


def apply_shadow(
    plate_region: numpy.ndarray,
    shadow: numpy.ndarray,
    strength: float = SHADOW_STRENGTH,
) -> numpy.ndarray:
    """Darken the plate where the contact-shadow pass has coverage.

    Multiplicative rather than additive: a shadow removes light, so it must scale what
    is already there and keep the plate's own texture visible through it.
    """
    if shadow.ndim == 3:
        shadow = shadow[..., 0]
    occlusion = (shadow.astype(numpy.float32) / 255.0) * strength
    darkened = plate_region.astype(numpy.float32) * (1.0 - occlusion[..., None])
    return numpy.clip(darkened, 0, 255).astype(numpy.uint8)


def alpha_over(
    background: numpy.ndarray, foreground_rgb: numpy.ndarray, alpha: numpy.ndarray,
) -> numpy.ndarray:
    weight = (alpha.astype(numpy.float32) / ALPHA_MAXIMUM)[..., None]
    blended = (
        foreground_rgb.astype(numpy.float32) * weight
        + background.astype(numpy.float32) * (1.0 - weight)
    )
    return numpy.clip(blended, 0, 255).astype(numpy.uint8)


def depth_test_alpha(
    actor_alpha: numpy.ndarray,
    actor_depth: numpy.ndarray | None,
    plate_depth: numpy.ndarray | None,
) -> numpy.ndarray:
    """Hide actor pixels that lie behind scene geometry (§5.4 step 2).

    Without a plate depth map every actor draws in front of everything, so a pedestrian
    walks over a railing instead of behind it. With one, the comparison is a mask.
    Absent depth is not an error — it is the documented reduced-fidelity path.
    """
    if actor_depth is None or plate_depth is None:
        return actor_alpha
    if actor_depth.shape != plate_depth.shape:
        raise CompositeError("Actor and plate depth layers must have identical shapes")
    occluded = actor_depth > plate_depth
    tested = actor_alpha.copy()
    tested[occluded] = 0
    return tested


def draw_contact_shadows(
    plate: numpy.ndarray, shadows: list[ShadowEllipse],
) -> numpy.ndarray:
    """Darken the ground under each actor. Multiplicative, so the plate shows through.

    A shadow removes light rather than adding darkness, so it scales what is already
    there — the paving texture stays visible through it, which is exactly what stops it
    reading as a grey sticker.
    """
    if not shadows:
        return plate
    height, width = plate.shape[:2]
    occlusion = numpy.zeros((height, width), dtype=numpy.float32)
    for shadow in shadows:
        if not shadow.is_visible:
            continue
        mask = numpy.zeros((height, width), dtype=numpy.float32)
        cv2.ellipse(
            mask,
            (int(round(shadow.center_x)), int(round(shadow.center_y))),
            (max(1, int(round(shadow.radius_x))), max(1, int(round(shadow.radius_y)))),
            0.0, 0.0, 360.0, float(shadow.strength), -1,
        )
        blur = max(1, int(shadow.blur_pixels) * 2 + 1)
        occlusion = numpy.maximum(occlusion, cv2.GaussianBlur(mask, (blur, blur), 0))
    darkened = plate.astype(numpy.float32) * (1.0 - occlusion[..., None])
    return numpy.clip(darkened, 0, 255).astype(numpy.uint8)


def plate_grain_sigma(plate_region: numpy.ndarray) -> float:
    """How much sensor noise the real footage carries, in 8-bit levels.

    Measured as the residual after a light blur, which is the high-frequency content a
    real sensor contributes and a clean render does not. A rendered actor dropped into
    grainy footage is conspicuously *too clean*, and matching this is the cheapest way
    to stop the eye separating the two layers.
    """
    grey = plate_region if plate_region.ndim == 2 else plate_region[..., 0]
    grey = grey.astype(numpy.float32)
    residual = grey - cv2.GaussianBlur(grey, (0, 0), GRAIN_MEASURE_BLUR_SIGMA)
    return float(numpy.std(residual))


def apply_grain(
    frame: numpy.ndarray, alpha: numpy.ndarray, sigma: float, seed: int,
) -> numpy.ndarray:
    """Add matched noise, but only where the actor actually is.

    Grain over the whole frame would re-noise footage that already has its own, so it is
    confined to the pixels the render contributed.
    """
    if sigma <= MINIMUM_GRAIN_SIGMA:
        return frame
    strength = min(sigma, MAXIMUM_GRAIN_SIGMA) * GRAIN_MATCH_RATIO
    generator = numpy.random.default_rng(seed)
    noise = generator.normal(0.0, strength, frame.shape[:2]).astype(numpy.float32)
    coverage = (alpha.astype(numpy.float32) / ALPHA_MAXIMUM)[..., None]
    grained = frame.astype(numpy.float32) + noise[..., None] * coverage
    return numpy.clip(grained, 0, 255).astype(numpy.uint8)


def soften_alpha(alpha: numpy.ndarray) -> numpy.ndarray:
    """Take the razor edge off the matte.

    A render's alpha is geometrically perfect and therefore aliased against footage that
    has lens softness and compression. One pixel of blur is the difference between an
    actor in the scene and a cutout pasted over it.
    """
    return cv2.GaussianBlur(alpha, (0, 0), EDGE_SOFTEN_SIGMA)


def composite_gap_frame(
    plate: numpy.ndarray,
    actor_layer: numpy.ndarray,
    region: RenderRegion,
    shadow_layer: numpy.ndarray | None = None,
    actor_depth: numpy.ndarray | None = None,
    plate_depth: numpy.ndarray | None = None,
    grade_strength: float = GRADE_STRENGTH,
    contact_shadows: list[ShadowEllipse] | None = None,
    grain_seed: int = 0,
) -> numpy.ndarray:
    """Build one reconstructed frame. The plate is never mutated.

    The actor layer is frame-sized and transparent outside the rendered region — see
    `apply_render_region` in the Blender service for why it is not cropped. The region
    still matters: it selects the patch of plate whose colour statistics the actor is
    graded against. Grading against the whole frame would average in sky and distant
    background that the actor is nowhere near.
    """
    _validate_layer(plate, RGB_CHANNELS, "Plate")
    _validate_layer(actor_layer, RGBA_CHANNELS, "Actor layer")
    if actor_layer.shape[:2] != plate.shape[:2]:
        raise CompositeError(
            f"Actor layer {actor_layer.shape[:2]} does not match the plate "
            f"{plate.shape[:2]}; the composite would be misaligned"
        )
    composed = plate.copy()
    # Shadows go down before the actor, because a figure stands on its own shadow.
    composed = draw_contact_shadows(composed, contact_shadows or [])
    if shadow_layer is not None:
        composed = apply_shadow(composed, shadow_layer)
    rows, columns = region_slice(plate, region)
    plate_region = composed[rows, columns]
    alpha = soften_alpha(depth_test_alpha(actor_layer[..., 3], actor_depth, plate_depth))
    graded = match_grade(actor_layer[..., :3], alpha, plate_region, grade_strength)
    graded = apply_grain(graded, alpha, plate_grain_sigma(plate_region), grain_seed)
    return alpha_over(composed, graded, alpha)
