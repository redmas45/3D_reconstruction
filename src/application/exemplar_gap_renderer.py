"""Draws a gap's actors from real photographs instead of rendering geometry.

Produces exactly what the Blender path produces — one frame-sized RGBA actor layer per
sparse sample, written as `frame_NNNNNN.png` — so the plate compositing, the optical-flow
expansion back to source frame rate, and the encoder downstream are untouched. The only
thing that changes is where the pixels come from.

Drawing order is by distance: an entity whose feet are lower in the frame is nearer the
camera, so it is drawn last and occludes the ones behind it. That single rule is what
stops a crowd looking like a collage, and it comes free from the projected foot position
without any depth buffer.
"""

import logging
from pathlib import Path

import cv2
import numpy

from application.exemplar_library import ExemplarBank
from domain.actor_placement import Placement, choose_observation, placement_for_frame
from domain.cancellation import CancellationCheck, raise_if_cancelled
# The layer filename is a contract shared with the compositor, which reads these back by
# name. Imported rather than restated so the two cannot drift into a "layer is missing".
from infrastructure.blender_protocol import frame_filename


LOGGER = logging.getLogger(__name__)

# Cut-outs are resampled to the size the projection asks for. Shrinking wants area
# averaging; enlarging wants something that does not turn edges to steps.
SHRINK_INTERPOLATION = cv2.INTER_AREA
ENLARGE_INTERPOLATION = cv2.INTER_CUBIC


def _resized(cutout: numpy.ndarray, pixel_height: float) -> numpy.ndarray | None:
    source_height, source_width = cutout.shape[:2]
    if source_height < 2 or source_width < 2:
        return None
    scale = float(pixel_height) / float(source_height)
    width = int(round(source_width * scale))
    height = int(round(pixel_height))
    if width < 2 or height < 2:
        return None
    interpolation = SHRINK_INTERPOLATION if scale < 1.0 else ENLARGE_INTERPOLATION
    return cv2.resize(cutout, (width, height), interpolation=interpolation)


def draw_cutout(
    layer: numpy.ndarray, cutout: numpy.ndarray, placement: Placement,
) -> bool:
    """Alpha-composite one cut-out onto the actor layer. True if anything landed."""
    resized = _resized(cutout, placement.pixel_height)
    if resized is None:
        return False
    height, width = resized.shape[:2]
    left = int(round(placement.centre_x - width / 2.0))
    top = int(round(placement.foot_y - height))
    layer_height, layer_width = layer.shape[:2]
    x0, y0 = max(0, left), max(0, top)
    x1, y1 = min(layer_width, left + width), min(layer_height, top + height)
    if x1 <= x0 or y1 <= y0:
        return False
    patch = resized[y0 - top:y1 - top, x0 - left:x1 - left]
    alpha = patch[:, :, 3:4].astype(numpy.float32) / 255.0
    target = layer[y0:y1, x0:x1]
    # Standard "over": the nearer actor wins where it is opaque, and the coverage of the
    # two accumulates where it is not, so a figure seen through a gap still shows.
    target[:, :, :3] = (
        patch[:, :, :3].astype(numpy.float32) * alpha
        + target[:, :, :3].astype(numpy.float32) * (1.0 - alpha)
    ).astype(numpy.uint8)
    target[:, :, 3:4] = numpy.maximum(
        target[:, :, 3:4],
        (alpha * 255.0).astype(numpy.uint8),
    )
    return True


def entities_with_banks(plan: dict, banks: dict[str, ExemplarBank]) -> list[dict]:
    return [
        entity for entity in plan.get("entities", [])
        if banks.get(str(entity.get("id")))
    ]


def coverage(plan: dict, banks: dict[str, ExemplarBank]) -> float:
    """Fraction of an entity's planned entities that have real footage to draw from."""
    entities = plan.get("entities", [])
    if not entities:
        return 1.0
    return len(entities_with_banks(plan, banks)) / len(entities)


def render_exemplar_layers(
    plan: dict,
    sparse_frames: list,
    banks: dict[str, ExemplarBank],
    frame_width: int,
    frame_height: int,
    layer_directory: Path,
    cancellation_check: CancellationCheck | None = None,
) -> dict:
    """Write one RGBA actor layer per sparse sample. Returns a summary for the report."""
    layer_directory.mkdir(parents=True, exist_ok=True)
    camera = plan["camera"]
    entities = entities_with_banks(plan, banks)
    previous_choice: dict[str, int] = {}
    drawn_total = 0
    empty_frames = 0
    for sparse in sparse_frames:
        raise_if_cancelled(cancellation_check)
        layer = numpy.zeros((frame_height, frame_width, 4), numpy.uint8)
        placed: list[tuple[float, dict, Placement]] = []
        for entity in entities:
            placement = placement_for_frame(
                entity, int(sparse.source_frame), frame_width, frame_height, camera,
            )
            if placement is None or not placement.is_drawable:
                continue
            placed.append((placement.foot_y, entity, placement))
        # Farthest first: a lower foot position means nearer the camera, so drawing in
        # increasing foot_y order leaves nearer figures on top of further ones.
        placed.sort(key=lambda item: item[0])
        drawn_here = 0
        for _, entity, placement in placed:
            entity_id = str(entity.get("id"))
            bank = banks[entity_id]
            index = choose_observation(
                list(bank.observations), list(bank.velocities), placement,
                previous_choice.get(entity_id),
            )
            if index is None:
                continue
            if draw_cutout(layer, bank.cutouts[index], placement):
                previous_choice[entity_id] = index
                drawn_here += 1
        drawn_total += drawn_here
        if drawn_here == 0:
            empty_frames += 1
        path = layer_directory / frame_filename(sparse.render_index)
        temporary = path.with_suffix(".writing.png")
        if not cv2.imwrite(str(temporary), layer):
            raise RuntimeError(f"Could not write actor layer {path}")
        temporary.replace(path)
    return {
        "mode": "observed_exemplar",
        "sparse_frames": len(sparse_frames),
        "entities_with_footage": len(entities),
        "entities_planned": len(plan.get("entities", [])),
        "actor_draws": drawn_total,
        "frames_with_no_actor": empty_frames,
    }
