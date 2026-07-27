"""Renders one gap as actors composited onto the recovered plate (§5).

This is the M2 replacement for the full-scene gap renderer. The difference is what
Blender is asked to do: instead of building a synthetic street and path-tracing the
whole frame, it draws only the entities YOLO detected, cropped to the rectangle they
occupy, on a transparent background. Everything else in the frame comes from real
footage via `clean_plate`.

Division of labour, deliberately:

  * this module decides *what* to render and *where* it lands
  * `render_region` computes the crop from the validated plan
  * the Blender service renders pixels and makes no decisions
  * `gap_compositor` assembles the frame on CPU while the GPU renders the next one

The plan is the only input. No hidden frames are read here — the plate is built from
visible ranges and the paths were validated upstream.
"""

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy

from application.gap_compositor import composite_gap_frame
from domain.actor_library import asset_object_name, catalog_digest, load_library
from domain.actor_proxies import is_articulated, proxy_for
from domain.cancellation import CancellationCheck, raise_if_cancelled
from domain.camera_projection import blender_camera_parameters
from domain.contact_shadow import gap_contact_shadows
from domain.gait import walk_pose
from domain.render_region import RenderRegion, gap_render_region, world_position_at_frame
from infrastructure.blender_protocol import frame_filename


LOGGER = logging.getLogger(__name__)

DEFAULT_RECONSTRUCTION_FPS = 12.0
MINIMUM_SPARSE_FRAMES = 2
DEFAULT_ACTOR_COLOUR = (0.55, 0.55, 0.58)
# Below this displacement between neighbouring waypoints the direction of travel is
# noise, and rotating an actor to face it would make a standing figure spin.
MINIMUM_TRAVEL_METERS = 0.05
GAP_SPECIFICATION_FILENAME = "gap_actors.json"
# Farneback parameters: pyramid scale, levels, window, iterations, polynomial
# neighbourhood, polynomial sigma, flags. Tuned for the large, smooth displacements a
# walking figure makes between sparse samples rather than for fine texture detail.
FLOW_PARAMETERS = (0.5, 3, 21, 3, 5, 1.2, 0)
# Forward-backward residual above which the flow field is not describing real motion.
# Measured on step tests: under 1.2 px while interpolation stays clean, over 7 px once
# the subject smears. Set between the two.
MAXIMUM_FLOW_RESIDUAL_PIXELS = 4.5
FLOW_TRUST_PERCENTILE = 95.0
# Above this 8-bit difference a pixel counts as having moved. Set above codec and grain
# noise so a still frame is recognised as still.
CHANGE_THRESHOLD = 6
# Padding around the changed area, wide enough for the flow window to see context on
# every side of the moving subject.
CHANGE_BOX_PADDING = 48
ACTOR_LAYER_DIRECTORY = "actors"


class ActorRenderError(RuntimeError):
    """A gap could not be rendered or composited."""


@dataclass(frozen=True)
class SparseFrame:
    """One rendered sample, and the source frames it will be expanded back onto."""

    render_index: int
    source_frame: int
    region: RenderRegion
    # Ground shadows are computed here rather than in the compositor so they use the
    # same positions and the same camera that placed the actors for this exact sample.
    shadows: tuple = ()


def reconstruction_fps(plan: dict) -> float:
    """Sparse render rate, never above the source rate."""
    render_contract = plan.get("render", {})
    target = float(render_contract.get("target_fps", DEFAULT_RECONSTRUCTION_FPS))
    return max(1.0, min(float(plan["fps"]), target))


def sparse_frame_count(plan: dict) -> int:
    duration = float(plan.get("duration_seconds", 0.0)) or (
        int(plan["frame_count"]) / max(1.0, float(plan["fps"]))
    )
    return max(MINIMUM_SPARSE_FRAMES, round(duration * reconstruction_fps(plan)))


def plan_sparse_frames(
    plan: dict, frame_width: int, frame_height: int,
) -> list[SparseFrame]:
    """Choose sparse samples and the crop each one needs.

    Render indexes are 1-based because that is what Blender's frame numbering and the
    `frame_NNNNNN.png` contract use; source frames come from the plan's hidden range.
    """
    entities = plan.get("entities", [])
    camera = plan["camera"]
    hidden_start = int(plan["hidden_range"]["start"])
    total_source_frames = int(plan["frame_count"])
    count = sparse_frame_count(plan)
    frames: list[SparseFrame] = []
    for render_index in range(1, count + 1):
        position = (render_index - 1) / max(1, count - 1)
        source_frame = hidden_start + round(position * (total_source_frames - 1))
        ground_positions = {
            str(entity.get("id")): world_position_at_frame(entity, source_frame)
            for entity in entities
        }
        frames.append(SparseFrame(
            render_index=render_index,
            source_frame=source_frame,
            region=gap_render_region(
                entities, source_frame, frame_width, frame_height, camera,
            ),
            shadows=tuple(gap_contact_shadows(
                entities,
                {key: value for key, value in ground_positions.items() if value is not None},
                frame_width, frame_height, camera,
            )),
        ))
    return frames


def _srgb_to_linear(value: float) -> float:
    """Blender shades in linear light; sampled pixel colours are sRGB.

    Handing an sRGB triple straight to `object.color` renders every actor noticeably
    darker and more saturated than the clothing it was sampled from, which is exactly
    the kind of small wrongness that makes a composite read as fake.
    """
    channel = max(0.0, min(1.0, float(value)))
    if channel <= 0.04045:
        return channel / 12.92
    return ((channel + 0.055) / 1.055) ** 2.4


def _entity_colour(entity: dict) -> list[float]:
    appearance = entity.get("appearance", {})
    for key in ("upper_color", "body_color", "vehicle_color", "lower_color"):
        colour = appearance.get(key)
        if isinstance(colour, (list, tuple)) and len(colour) >= 3:
            return [round(_srgb_to_linear(value), 6) for value in colour[:3]]
    return [round(_srgb_to_linear(value), 6) for value in DEFAULT_ACTOR_COLOUR]



def _sample_track(
    entity: dict, sparse_frames: list["SparseFrame"], fps: float,
) -> list[dict]:
    """Position, heading, distance travelled and speed at every render sample.

    Sampled at exactly the frames the regions were computed for, so the actor and its
    crop rectangle can never describe different moments.
    """
    positions = []
    previous = None
    for sparse_frame in sparse_frames:
        position = world_position_at_frame(entity, sparse_frame.source_frame)
        previous = position if position is not None else previous
        positions.append(previous or [0.0, 0.0, 0.0])
    track = []
    distance = 0.0
    for index, position in enumerate(positions):
        if index > 0:
            distance += math.dist(positions[index - 1][:2], position[:2])
        track.append({
            "position": position,
            "distance": distance,
            "speed": _local_speed(positions, index, sparse_frames, fps),
            "heading": _local_heading(positions, index),
            "elapsed": (sparse_frames[index].source_frame - sparse_frames[0].source_frame) / fps,
        })
    _smooth_headings(track)
    return track


def _local_speed(
    positions: list[list[float]], index: int, sparse_frames: list["SparseFrame"], fps: float,
) -> float:
    """Ground speed around this sample, from the samples either side of it."""
    first = max(0, index - 1)
    last = min(len(positions) - 1, index + 1)
    if last == first:
        return 0.0
    seconds = (sparse_frames[last].source_frame - sparse_frames[first].source_frame) / fps
    if seconds <= 0.0:
        return 0.0
    return math.dist(positions[first][:2], positions[last][:2]) / seconds


def _local_heading(positions: list[list[float]], index: int) -> float | None:
    first = max(0, index - 1)
    last = min(len(positions) - 1, index + 1)
    delta_x = positions[last][0] - positions[first][0]
    delta_y = positions[last][1] - positions[first][1]
    if math.hypot(delta_x, delta_y) < MINIMUM_TRAVEL_METERS:
        return None
    return math.degrees(math.atan2(delta_x, delta_y))


def _smooth_headings(track: list[dict]) -> None:
    """Fill gaps where the figure was too slow to infer a direction, and unwrap.

    Without unwrapping, a heading crossing 180 degrees makes the actor spin a full turn
    between two samples — visually far worse than the tiny direction change it encodes.
    """
    known = [item["heading"] for item in track]
    last_known = next((value for value in known if value is not None), 0.0)
    for index, value in enumerate(known):
        if value is None:
            known[index] = last_known
        else:
            last_known = value
    for index in range(1, len(known)):
        while known[index] - known[index - 1] > 180.0:
            known[index] -= 360.0
        while known[index] - known[index - 1] < -180.0:
            known[index] += 360.0
    for item, heading in zip(track, known):
        item["heading"] = heading



def build_gap_specification(plan: dict, sparse_frames: list["SparseFrame"]) -> dict:
    """Translate the validated plan into the shape the warm shell instantiates.

    A keyframe is emitted at every render sample rather than only at the plan's
    waypoints. Two reasons. The gait phase advances continuously with distance, so it
    needs evaluating everywhere, not interpolating between three poses. And sampling at
    exactly the frames the crop rectangles were computed for makes it impossible for the
    actor and its crop to describe different moments.
    """
    fps = float(plan["fps"])
    actors = []
    for entity in plan.get("entities", []):
        class_name = str(entity.get("kind", "person"))
        specification = proxy_for(class_name)
        track = _sample_track(entity, sparse_frames, fps)
        actors.append({
            "id": str(entity["id"]),
            "class_name": class_name,
            "asset_name": asset_object_name(class_name),
            "proxy": specification.proxy,
            "dimensions": [specification.length, specification.width, specification.height],
            "ground_offset": specification.ground_offset_meters,
            "body_height_ratio": specification.body_height_ratio,
            "cabin_length_ratio": specification.cabin_length_ratio,
            "cabin_width_ratio": specification.cabin_width_ratio,
            "color": _entity_colour(entity),
            "keyframes": _actor_keyframes(
                entity, class_name, track, sparse_frames,
            ),
        })
    return {
        "gap_index": int(plan["gap_index"]),
        "frame_count": len(sparse_frames),
        "actors": actors,
    }


def _actor_keyframes(
    entity: dict,
    class_name: str,
    track: list[dict],
    sparse_frames: list["SparseFrame"],
) -> list[dict]:
    declared_heading = entity.get("animation", {}).get("heading_degrees")
    articulated = is_articulated(class_name)
    keyframes = []
    for sample, sparse_frame in zip(track, sparse_frames):
        keyframe = {
            "frame": sparse_frame.render_index,
            "location": [round(float(value), 5) for value in sample["position"]],
            "heading_degrees": round(
                float(declared_heading) if declared_heading is not None
                else sample["heading"], 4,
            ),
        }
        if articulated:
            keyframe["pose"] = walk_pose(
                sample["distance"], sample["speed"], sample["elapsed"],
            ).as_dict()
            keyframe["speed"] = round(sample["speed"], 4)
        keyframes.append(keyframe)
    return keyframes


def actor_library_contract(plan: dict, library_directory: Path | None) -> dict:
    """Which prebuilt assets this job needs, when a current library provides them.

    Returns an empty contract when no library is built or it is stale, in which case
    Blender generates the geometry exactly as before and produces identical output.
    """
    if library_directory is None:
        return {}
    library = load_library(Path(library_directory))
    if library is None:
        return {}
    wanted = {str(entity.get("kind", "")) for entity in plan.get("entities", [])}
    covered = [kind for kind in wanted if library.object_name_for(kind) is not None]
    if not covered:
        return {}
    return {
        "path": str(library.blend_path),
        "assets": library.objects_to_append(covered),
        "digest": library.digest,
    }


def build_job_manifest(
    plan: dict,
    frame_width: int,
    frame_height: int,
    library_directory: Path | None = None,
) -> dict:
    """Camera and render settings the warm shell is built from, once per job.

    The camera is *derived* from the plan's projection contract rather than copied out
    of it. The contract describes the projection the region arithmetic uses; Blender
    needs the equivalent lens and rotation, and the two must be the same camera or
    actors render outside their own crop.
    """
    render_contract = plan.get("render", {})
    return {
        "resolution": [int(frame_width), int(frame_height)],
        "camera": blender_camera_parameters(frame_width, frame_height, plan["camera"]),
        # The geometry every actor is built from. Carried here so it reaches the layer
        # cache key: changing a proxy's dimensions or the skeleton changes the pixels,
        # and reusing layers across that change would mix two models in one video.
        "actor_geometry_digest": catalog_digest(),
        "actor_library": actor_library_contract(plan, library_directory),
        "render": {
            "engine": str(render_contract.get("engine", "BLENDER_EEVEE_NEXT")),
            "samples": int(render_contract.get("cycles_samples", 8)),
            "denoise": bool(render_contract.get("cycles_use_denoising", True)),
        },
    }


def load_actor_layer(path: Path) -> numpy.ndarray:
    """Read an RGBA actor layer.

    OpenCV yields BGRA, and the plate it will be composited onto is BGR from the same
    decoder, so the two already agree on channel order and nothing is swapped here.
    """
    layer = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if layer is None:
        raise ActorRenderError(f"Could not read rendered actor layer: {path}")
    if layer.ndim != 3:
        raise ActorRenderError(f"Actor layer is not an image with channels: {path}")
    if layer.shape[2] == 3:
        # No alpha means nothing was transparent; treat the whole tile as opaque.
        alpha = numpy.full(layer.shape[:2] + (1,), 255, dtype=numpy.uint8)
        return numpy.concatenate([layer, alpha], axis=2)
    return layer


def composite_sparse_frames(
    plate: numpy.ndarray,
    sparse_frames: list[SparseFrame],
    layer_directory: Path,
    cancellation_check: CancellationCheck | None = None,
) -> list[numpy.ndarray]:
    """Composite every rendered sample onto the plate."""
    composed: list[numpy.ndarray] = []
    for sparse_frame in sparse_frames:
        raise_if_cancelled(cancellation_check)
        layer_path = layer_directory / frame_filename(sparse_frame.render_index)
        if not layer_path.is_file():
            raise ActorRenderError(f"Rendered layer is missing: {layer_path}")
        composed.append(composite_gap_frame(
            plate,
            load_actor_layer(layer_path),
            sparse_frame.region,
            contact_shadows=list(sparse_frame.shadows),
            # Seeded per frame so grain differs frame to frame like real sensor noise,
            # but identically on every run so output stays reproducible.
            grain_seed=sparse_frame.render_index,
        ))
    return composed


def _interpolate_pair(
    earlier: numpy.ndarray, later: numpy.ndarray, position: float,
) -> numpy.ndarray:
    """One in-between frame, warped along the motion between two samples.

    Cross-fading alone would ghost — both figures visible at once. Warping each frame
    along the measured flow moves the actor to where it actually is at this instant, and
    blending the two warps fills whatever either one leaves behind.

    **Guarded, because flow fails badly rather than gracefully.** When a subject moves
    further between samples than the estimator can match, the two warps land in
    different places and the blend dissolves the subject into a smear — which is far
    worse than the judder it was meant to fix. So the flow is measured, and if the
    displacement is beyond the range it can be trusted at, this falls back to repeating
    the nearer sample. The worst case degrades to the old behaviour instead of to
    corruption.
    """
    if position <= 0.0:
        return earlier
    if position >= 1.0:
        return later
    # Only the actors move; the rest of the frame is the same plate in both samples.
    # Estimating flow over the whole frame spends most of its time on pixels that are
    # identical by construction, and invites the estimator to invent motion in static
    # background. Restricting it to what actually changed is both faster and cleaner.
    box = _changed_box(earlier, later)
    if box is None:
        return earlier
    left, top, right, bottom = box
    earlier_crop = earlier[top:bottom, left:right]
    later_crop = later[top:bottom, left:right]
    earlier_grey = cv2.cvtColor(earlier_crop, cv2.COLOR_BGR2GRAY)
    later_grey = cv2.cvtColor(later_crop, cv2.COLOR_BGR2GRAY)
    forward = cv2.calcOpticalFlowFarneback(earlier_grey, later_grey, None, *FLOW_PARAMETERS)
    backward = cv2.calcOpticalFlowFarneback(later_grey, earlier_grey, None, *FLOW_PARAMETERS)
    if not _flow_is_trustworthy(forward, backward):
        return earlier if position < 0.5 else later
    blended = cv2.addWeighted(
        _warp(earlier_crop, forward * position), 1.0 - position,
        _warp(later_crop, backward * (1.0 - position)), position, 0.0,
    )
    interpolated = earlier.copy()
    interpolated[top:bottom, left:right] = blended
    return interpolated


def _changed_box(
    earlier: numpy.ndarray, later: numpy.ndarray,
) -> tuple[int, int, int, int] | None:
    """Bounding box of what differs between two samples, padded for the flow window.

    Returns None when nothing moved, which is the correct answer for a gap whose actors
    are all standing still: there is nothing to interpolate and the sample is the frame.
    """
    difference = cv2.absdiff(earlier, later).max(axis=2)
    rows = numpy.flatnonzero(difference.max(axis=1) > CHANGE_THRESHOLD)
    columns = numpy.flatnonzero(difference.max(axis=0) > CHANGE_THRESHOLD)
    if rows.size == 0 or columns.size == 0:
        return None
    height, width = difference.shape
    return (
        max(0, int(columns[0]) - CHANGE_BOX_PADDING),
        max(0, int(rows[0]) - CHANGE_BOX_PADDING),
        min(width, int(columns[-1]) + 1 + CHANGE_BOX_PADDING),
        min(height, int(rows[-1]) + 1 + CHANGE_BOX_PADDING),
    )


def _flow_is_trustworthy(forward: numpy.ndarray, backward: numpy.ndarray) -> bool:
    """Whether the two flow fields agree with each other.

    Following the forward flow and then the backward flow should return a pixel to where
    it started. Where it does not, the estimator has not actually matched anything.

    This is the check rather than flow magnitude because Farneback does not fail loudly:
    when a subject moves further than it can match it returns a *small* flow, not a large
    one, so magnitude looks reassuring exactly when the estimate is worthless. Measured
    on step tests, this residual sits under 1.2 pixels while interpolation is clean and
    jumps past 7 once the subject starts smearing.
    """
    resampled = numpy.dstack([
        _warp_channel(backward[..., 0], forward), _warp_channel(backward[..., 1], forward),
    ])
    residual = numpy.hypot(
        forward[..., 0] + resampled[..., 0], forward[..., 1] + resampled[..., 1],
    )
    return float(numpy.percentile(residual, FLOW_TRUST_PERCENTILE)) <= MAXIMUM_FLOW_RESIDUAL_PIXELS


def _warp_channel(channel: numpy.ndarray, flow: numpy.ndarray) -> numpy.ndarray:
    height, width = channel.shape[:2]
    grid_x, grid_y = numpy.meshgrid(
        numpy.arange(width, dtype=numpy.float32),
        numpy.arange(height, dtype=numpy.float32),
    )
    return cv2.remap(
        channel.astype(numpy.float32),
        grid_x + flow[..., 0], grid_y + flow[..., 1],
        cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )


def _warp(frame: numpy.ndarray, flow: numpy.ndarray) -> numpy.ndarray:
    height, width = frame.shape[:2]
    grid_x, grid_y = numpy.meshgrid(
        numpy.arange(width, dtype=numpy.float32),
        numpy.arange(height, dtype=numpy.float32),
    )
    return cv2.remap(
        frame,
        grid_x + flow[..., 0],
        grid_y + flow[..., 1],
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def expand_to_source_frames(
    composed: list[numpy.ndarray],
    source_frame_count: int,
    interpolate: bool = True,
) -> list[numpy.ndarray]:
    """Restore the exact source frame count from the sparse renders (§7).

    Rendering at 12 fps and repeating each frame to reach 30 is what makes reconstructed
    gaps read as stuttering next to the real footage either side of them — the timing is
    exact but the motion arrives in visible steps.

    So in-between frames are warped along the optical flow between the two samples that
    bracket them. This interpolates *pixels that were rendered*; it never extrapolates
    past the last sample and never invents a position the plan did not contain. Set
    `interpolate=False` for the older nearest-sample behaviour.
    """
    if not composed:
        raise ActorRenderError("No composited frames were produced for this gap")
    if source_frame_count <= 0:
        raise ActorRenderError("Source frame count must be positive")
    last_index = len(composed) - 1
    if last_index == 0 or not interpolate:
        return _nearest_expansion(composed, source_frame_count)
    expanded = []
    for source_index in range(source_frame_count):
        exact = (source_index / max(1, source_frame_count - 1)) * last_index
        earlier_index = min(last_index - 1, int(exact))
        expanded.append(_interpolate_pair(
            composed[earlier_index], composed[earlier_index + 1], exact - earlier_index,
        ))
    return expanded


def _nearest_expansion(
    composed: list[numpy.ndarray], source_frame_count: int,
) -> list[numpy.ndarray]:
    last_index = len(composed) - 1
    return [
        composed[min(last_index, round(
            (source_index / max(1, source_frame_count - 1)) * last_index
        ))]
        for source_index in range(source_frame_count)
    ]


def write_gap_video(
    frames: list[numpy.ndarray], fps: float, output_path: Path,
) -> Path:
    """Encode the composited gap. Written to a temporary name and renamed on success."""
    if not frames:
        raise ActorRenderError("Cannot encode a gap with no frames")
    height, width = frames[0].shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".writing.mp4")
    writer = cv2.VideoWriter(
        str(temporary_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height),
    )
    if not writer.isOpened():
        raise ActorRenderError(f"Could not open a video writer for {output_path}")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()
    temporary_path.replace(output_path)
    return output_path
