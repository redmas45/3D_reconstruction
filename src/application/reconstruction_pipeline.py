"""Coordinates evidence analysis, reconstruction rendering, and final video assembly."""

import hashlib
import json
import logging
import random
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import cv2
import numpy

from detect import RELEVANT_COCO_CLASSES, detect_scene_objects
from evaluate import evaluate_reconstructions
from gap_selector import choose_hidden_gaps
from scene_intelligence import summarize_scene
from stitch import stitch_sequence
from visual_output import render_annotated_visible_chunk
from application.actor_render_job import (
    ActorJobError,
    ActorRenderJob,
    actor_path_is_supported,
    render_actor_gaps,
)
from application.blender_pipeline import prepare_blender_assets, render_blender_gap
from application.exemplar_library import build_exemplar_banks
from application.evidence_reasoning import reason_about_reconstruction
from application.plate_evidence import PlateEvidenceError, resolve_clean_plate
from domain.cancellation import CancellationCheck, raise_if_cancelled
from domain.configuration import load_validated_configuration
from domain.evidence_contract import validate_visible_evidence_only
from domain.reconstruction_cache import (
    cached_detections_are_valid as _cached_detections_are_valid,
    gap_cache_configuration as _gap_cache_configuration,
    selection_cache_is_compatible as _selection_cache_is_compatible,
    shot_contract as _shot_contract,
    source_video_contract as _source_video_contract,
)
from domain.presentation_manifest import (
    build_presentation_manifest,
    write_presentation_manifest,
)
from domain.render_runtime_budget import (
    RepresentativePreviewApprovalRequired,
    gap_render_costs,
    preview_is_approved,
    predicted_total_seconds,
    representative_gap_index,
)
from domain.render_resolution import scaled_render_dimensions
from domain.shot_timeline import (
    Shot,
    clip_ranges_to_shot,
    shot_containing,
    shot_spanning,
    single_shot_timeline,
)
from infrastructure.blender_kernel_cache import kernel_cache_environment
from infrastructure.blender_runner import DEFAULT_RENDER_STALL_TIMEOUT_SECONDS, find_blender_executable
from infrastructure.json_files import read_json_file, write_json_file
from infrastructure.shot_segmentation import detect_shots
from infrastructure.media_tools import (
    VideoContract,
    encode_with_source_audio,
    find_media_tool,
    validate_constant_frame_rate,
    validate_video_contract,
)
from infrastructure.source_video import (
    inspect_source_video as video_info,
    source_video_sha256 as _source_video_sha256,
    validate_source_resource_limits as _validate_source_resource_limits,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "config" / "reconstruction_config.json"
ProgressCallback = Callable[[str, float, str], None]
VALIDATION_PROGRESS = 0.01
# Shot detection reads the whole video once before anything is selected, because where
# the gaps may go depends on where the cuts are.
SHOT_SEGMENTATION_PROGRESS = 0.02
GAP_SELECTION_PROGRESS = 0.04
SEGMENT_PREPARATION_START = 0.06
SEGMENT_PREPARATION_SPAN = 0.07
DETECTION_START = 0.13
DETECTION_SPAN = 0.35
BASE_PLANNING_PROGRESS = 0.49
CLUE_EXTRACTION_PROGRESS = 0.51
REASONING_PROGRESS = 0.53
DECISION_VALIDATION_PROGRESS = 0.55
PLANNING_PROGRESS = 0.57
RENDERING_START = 0.58
RENDERING_SPAN = 0.27
EVALUATION_PROGRESS = 0.85
STITCHING_PROGRESS = 0.94
COMPLETED_PROGRESS = 1.0
DEFAULT_PARALLEL_GAP_RENDERERS = 3
BLENDER_RENDER_PROGRESS_SHARE = 0.85
ACTOR_COMPOSITE_MODE = "actor_composite"
# Where an actor's pixels come from. Observed footage of the actual subject beats
# modelled geometry at the size figures occupy in real frames, and brings the scene's own
# light, clothing, motion blur and grain with it for free.
ACTOR_SOURCE_EXEMPLAR = "observed_exemplar"
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PipelineOptions:
    config_data: dict
    output_dir: Path
    reuse_work: bool = False
    cancellation_check: CancellationCheck | None = None


@dataclass(frozen=True)
class PreparedReconstruction:
    video_info: dict
    gap_selection: dict
    segment_paths: dict[tuple[str, int], Path]
    scene_report: dict
    work_dir: Path
    blender_plan_paths: list[Path]
    # The scene structure every later stage is scoped to. A video with no cuts carries a
    # single shot spanning it, so nothing downstream needs a special case.
    shots: tuple[tuple[int, int], ...] = ()
    shot_report: dict | None = None


@dataclass(frozen=True)
class TimelineRenderContext:
    video_path: Path
    prepared: PreparedReconstruction
    configuration: dict
    reuse_work: bool
    blender_rendered_paths: dict[int, Path]
    cancellation_check: CancellationCheck | None


class ParallelGapProgress:
    def __init__(self, callback: ProgressCallback | None, gap_count: int, worker_count: int) -> None:
        self._callback = callback
        self._gap_count = max(1, gap_count)
        self._worker_count = worker_count
        self._fractions: dict[int, float] = {}
        self._lock = threading.Lock()

    def report(self, gap_index: int, current_frame: int, total_frames: int) -> None:
        fraction = current_frame / max(1, total_frames)
        with self._lock:
            self._fractions[gap_index] = max(self._fractions.get(gap_index, 0.0), fraction)
            overall_fraction = sum(self._fractions.values()) / self._gap_count
        progress = RENDERING_START + RENDERING_SPAN * BLENDER_RENDER_PROGRESS_SHARE * overall_fraction
        detail = (
            f"Rendering gap {gap_index + 1} of {self._gap_count}: "
            f"frame {current_frame} of {total_frames} with {self._worker_count} workers"
        )
        _report(self._callback, "rendering", progress, detail)


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict:
    return load_validated_configuration(path)


def _report(callback: ProgressCallback | None, stage: str, progress: float, detail: str) -> None:
    if callback is not None:
        callback(stage, max(0.0, min(1.0, progress)), detail)


def yolo_class_ids(config: dict) -> list[int]:
    classes = config.get("yolo", {}).get("classes", {})
    configured_ids = {int(class_id) for class_id in classes}
    relevant_ids = set(RELEVANT_COCO_CLASSES)
    selected_ids = configured_ids & relevant_ids if configured_ids else relevant_ids
    return sorted(selected_ids)


def normalize_confidence(value: float) -> float:
    confidence = float(value)
    if confidence > 1.0:
        confidence /= 100.0
    return max(0.0, min(1.0, confidence))


def write_json(path: Path, payload: object) -> None:
    write_json_file(path, payload)


def write_video_range(
    video_path: Path,
    start_frame: int,
    end_frame: int,
    output_path: Path,
    cancellation_check: CancellationCheck | None = None,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video_path.name}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise OSError(f"Cannot create video segment: {output_path.name}")
    written_frames = 0
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_frame))
        for _ in range(start_frame, end_frame + 1):
            raise_if_cancelled(cancellation_check)
            success, frame = capture.read()
            if not success:
                break
            writer.write(frame)
            written_frames += 1
    finally:
        capture.release()
        writer.release()
    expected_frames = end_frame - start_frame + 1
    if written_frames != expected_frames:
        raise ValueError(
            f"Video ended while extracting {output_path.name}: "
            f"wrote {written_frames} of {expected_frames} frames"
        )
    return output_path


def _segment_path(segment_dir: Path, segment: dict) -> Path:
    return segment_dir / f"{segment['kind']}_{segment['index']:02d}_{segment['start']}_{segment['end']}.mp4"


def reserve_timeline_segment_paths(
    timeline: list[dict],
    segment_dir: Path,
    progress_callback: ProgressCallback | None,
) -> dict[tuple[str, int], Path]:
    paths: dict[tuple[str, int], Path] = {}
    for segment in timeline:
        output_path = _segment_path(segment_dir, segment)
        paths[(segment["kind"], segment["index"])] = output_path
    segment_count = len(timeline)
    progress = SEGMENT_PREPARATION_START + SEGMENT_PREPARATION_SPAN
    _report(progress_callback, "preparing", progress, f"Indexed {segment_count} timeline segments")
    return paths


def _new_selection(
    info: dict, gap_config: dict, rng: random.Random, shots: list[tuple[int, int]] | None,
) -> dict:
    selection = choose_hidden_gaps(
        total_frames=info["frames"],
        fps=info["fps"],
        rng=rng,
        missing_fraction=gap_config.get("missing_fraction", 0.25),
        min_gap_seconds=gap_config.get("min_seconds", 5.0),
        max_gap_seconds=gap_config.get("max_seconds", 7.0),
        compact_min_gap_seconds=gap_config.get("compact_min_seconds", 1.0),
        compact_max_gap_seconds=gap_config.get("compact_max_seconds", 3.0),
        review_profile_min_video_seconds=gap_config.get(
            "review_profile_min_video_seconds", 60.0,
        ),
        context_seconds=gap_config.get("context_seconds", 2.0),
        shots=shots,
    )
    return {
        **selection,
        "source_video_contract": _source_video_contract(info),
        "gap_configuration": _gap_cache_configuration(gap_config),
        "shot_contract": _shot_contract(shots),
    }


def _load_selection(
    work_dir: Path,
    info: dict,
    config: dict,
    rng: random.Random,
    reuse_work: bool,
    shots: list[tuple[int, int]] | None,
) -> dict:
    selection_path = work_dir / "gap_selection.json"
    if reuse_work and selection_path.exists():
        selection = read_json_file(selection_path)
        if _selection_cache_is_compatible(selection, info, config.get("gap", {}), shots):
            return selection
    selection = _new_selection(info, config.get("gap", {}), rng, shots)
    write_json(selection_path, selection)
    return selection


def _load_shot_timeline(
    video_path: Path,
    work_dir: Path,
    info: dict,
    reuse_work: bool,
    cancellation_check: CancellationCheck | None,
) -> tuple[dict, list[tuple[int, int]]]:
    """Find the cuts, so every later stage can be scoped to a single scene.

    Cached like the other analysis outputs: detection re-reads the whole video, and the
    answer only changes when the video does.
    """
    report_path = work_dir / "shot_report.json"
    report = None
    if reuse_work and report_path.exists():
        cached = read_json_file(report_path)
        if isinstance(cached, dict) and cached.get("frame_count") == int(info["frames"]):
            report = cached
    if report is None:
        work_dir.mkdir(parents=True, exist_ok=True)
        report = detect_shots(video_path, int(info["frames"]), cancellation_check)
        write_json(report_path, report)
    shots = [(int(shot["start"]), int(shot["end"])) for shot in report["shots"]]
    return report, shots


def _load_detections(
    video_path: Path,
    work_dir: Path,
    selection: dict,
    config: dict,
    reuse_work: bool,
    progress_callback: ProgressCallback | None,
    cancellation_check: CancellationCheck | None,
) -> list[dict]:
    detections_path = work_dir / "detections.json"
    manifest_path = work_dir / "detections_manifest.json"
    yolo_config = config.get("yolo", {})
    tracker_config = yolo_config.get("tracker_config")
    if tracker_config and not Path(tracker_config).is_absolute():
        tracker_config = str(ROOT / tracker_config)
    cache_contract = _detection_cache_contract(selection, config, yolo_config, tracker_config)
    if reuse_work and detections_path.exists() and manifest_path.exists():
        cached_manifest = read_json_file(manifest_path)
        if cached_manifest == cache_contract:
            detections = read_json_file(detections_path)
            if _cached_detections_are_valid(detections, selection["visible_ranges"]):
                _report(
                    progress_callback,
                    "detecting",
                    DETECTION_START + DETECTION_SPAN,
                    "Reused compatible detection cache",
                )
                return detections

    def report_detection(completed: int, total: int) -> None:
        fraction = completed / max(1, total)
        progress = DETECTION_START + (DETECTION_SPAN * fraction)
        _report(progress_callback, "detecting", progress, f"Tracked evidence segment {completed} of {total}")

    detections = detect_scene_objects(
        video_path=str(video_path),
        visible_ranges=[tuple(item) for item in selection["visible_ranges"]],
        model_name=yolo_config.get("model", "yolo26m.pt"),
        class_ids=yolo_class_ids(config),
        frame_stride=yolo_config.get("frame_stride", 8),
        downscale_width=yolo_config.get("downscale_width", 960),
        conf=normalize_confidence(yolo_config.get("confidence", 0.3)),
        tracker_config=tracker_config,
        pose_model_name=(
            yolo_config.get("pose_model", "yolo26n-pose.pt")
            if yolo_config.get("pose_enabled", True) else None
        ),
        pose_confidence=normalize_confidence(yolo_config.get("pose_confidence", 0.3)),
        pose_boundary_samples=int(yolo_config.get("pose_boundary_samples", 2)),
        progress_callback=report_detection,
        cancellation_check=cancellation_check,
        inference_width=int(yolo_config.get("inference_width", 0)),
        iou=float(yolo_config.get("iou", 0.7)),
        max_detections=int(yolo_config.get("max_detections", 300)),
        augment=bool(yolo_config.get("augment", False)),
    )
    write_json(detections_path, detections)
    write_json(manifest_path, cache_contract)
    return detections


def _detection_cache_contract(
    selection: dict,
    config: dict,
    yolo_config: dict,
    tracker_config: str | None,
) -> dict:
    return {
        "source_video_contract": selection.get("source_video_contract"),
        "visible_ranges": selection["visible_ranges"],
        "model": str(yolo_config.get("model", "yolo26m.pt")),
        "class_ids": yolo_class_ids(config),
        "frame_stride": int(yolo_config.get("frame_stride", 8)),
        "downscale_width": int(yolo_config.get("downscale_width", 960)),
        "confidence": normalize_confidence(yolo_config.get("confidence", 0.3)),
        "tracker_config": tracker_config,
        "pose_enabled": bool(yolo_config.get("pose_enabled", True)),
        "pose_model": str(yolo_config.get("pose_model", "yolo26n-pose.pt")),
        "pose_confidence": normalize_confidence(
            yolo_config.get("pose_confidence", 0.3),
        ),
        "pose_boundary_samples": int(
            yolo_config.get("pose_boundary_samples", 2),
        ),
        "inference_width": int(yolo_config.get("inference_width", 0)),
        "iou": float(yolo_config.get("iou", 0.7)),
        "max_detections": int(yolo_config.get("max_detections", 300)),
        "augment": bool(yolo_config.get("augment", False)),
    }


def _render_visible_segment(
    video_path: Path,
    output_path: Path,
    segment: dict,
    scene_report: dict,
    info: dict,
    config: dict,
    visible_count: int,
    cancellation_check: CancellationCheck | None,
) -> None:
    scene_config = config.get("scene", {})
    yolo_config = config.get("yolo", {})
    render_annotated_visible_chunk(
        str(video_path),
        str(output_path),
        (segment["start"], segment["end"]),
        scene_report,
        f"EVIDENCE {segment['index'] + 1}/{visible_count}",
        info["fps"],
        max_gap=max(20, yolo_config.get("frame_stride", 8) * scene_config.get("track_interpolation_max_gap_multiplier", 4)),
        visual_config=config.get("visualization", {}),
        cancellation_check=cancellation_check,
    )


def _render_timeline(
    context: TimelineRenderContext,
    progress_callback: ProgressCallback | None,
) -> tuple[list[str], list[dict]]:
    sequence: list[str] = []
    evaluation_items: list[dict] = []
    timeline = context.prepared.gap_selection["timeline"]
    timeline_start = _timeline_render_start()
    timeline_span = (RENDERING_START + RENDERING_SPAN) - timeline_start
    for item_index, segment in enumerate(timeline):
        raise_if_cancelled(context.cancellation_check)
        output_path, evaluation_item = _render_timeline_segment(context, segment)
        sequence.append(str(output_path))
        if evaluation_item is not None:
            evaluation_items.append(evaluation_item)
        fraction = (item_index + 1) / max(1, len(timeline))
        progress = timeline_start + (timeline_span * fraction)
        _report(progress_callback, "rendering", progress, f"Rendered timeline segment {item_index + 1} of {len(timeline)}")
    return sequence, evaluation_items


def _render_timeline_segment(context: TimelineRenderContext, segment: dict) -> tuple[Path, dict | None]:
    prepared = context.prepared
    if segment["kind"] == "visible":
        output_path = prepared.work_dir / "visual_segments" / f"visible_{segment['index']:02d}.mp4"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        visible_count = len(prepared.gap_selection["visible_ranges"])
        _render_visible_segment(
            context.video_path, output_path, segment, prepared.scene_report, prepared.video_info,
            context.configuration, visible_count, context.cancellation_check,
        )
        return output_path, None
    return _render_hidden_segment(context, segment)


def _render_hidden_segment(context: TimelineRenderContext, segment: dict) -> tuple[Path, dict]:
    prepared = context.prepared
    gap_index = segment["index"]
    output_path = context.blender_rendered_paths.get(gap_index)
    if output_path is None:
        output_path = _render_blender_hidden_segment(context, gap_index)
    evaluation_item = {
        "gap_index": gap_index,
        "hidden_range": tuple(prepared.gap_selection["hidden_ranges"][gap_index]),
        "truth_path": str(prepared.segment_paths[("hidden", gap_index)]),
        "reconstruction_path": str(output_path),
    }
    return output_path, evaluation_item


def _render_blender_hidden_segment(
    context: TimelineRenderContext,
    gap_index: int,
    frame_progress_callback: Callable[[int, int], None] | None = None,
) -> Path:
    prepared = context.prepared
    gap_directory = prepared.work_dir / "gaps" / f"gap_{gap_index:02d}"
    stall_timeout_seconds = int(
        context.configuration.get("renderer", {}).get(
            "gap_render_stall_timeout_seconds", DEFAULT_RENDER_STALL_TIMEOUT_SECONDS,
        )
    )
    output_path = render_blender_gap(
        ROOT,
        prepared.blender_plan_paths[gap_index],
        gap_directory,
        context.reuse_work,
        cancellation_check=context.cancellation_check,
        progress_callback=frame_progress_callback,
        stall_timeout_seconds=stall_timeout_seconds,
    )
    hidden_range = prepared.gap_selection["hidden_ranges"][gap_index]
    plan = json.loads(
        prepared.blender_plan_paths[gap_index].read_text(encoding="utf-8"),
    )
    render_width, render_height = scaled_render_dimensions(
        int(plan["render"]["source_width"]),
        int(plan["render"]["source_height"]),
        int(plan["render"]["production_scale_percent"]),
    )
    expected_contract = VideoContract(
        render_width,
        render_height,
        prepared.video_info["fps"],
        int(hidden_range[1]) - int(hidden_range[0]) + 1,
    )
    validate_video_contract(output_path, expected_contract)
    return output_path


def _render_blender_gaps(
    context: TimelineRenderContext,
    progress_callback: ProgressCallback | None,
) -> dict[int, Path]:
    raise_if_cancelled(context.cancellation_check)
    gap_count = len(context.prepared.gap_selection["hidden_ranges"])
    if gap_count and _actor_render_requested(context.configuration):
        rendered = _render_actor_gaps(context, progress_callback)
        if rendered is not None:
            return rendered
    worker_count = _parallel_gap_renderer_count(context.configuration, gap_count)
    if _runtime_budget_enabled(context.configuration) and gap_count > 0:
        return _render_blender_gaps_with_budget(
            context, progress_callback, worker_count,
        )
    return _render_blender_gap_indexes(
        context, progress_callback, list(range(gap_count)), gap_count, worker_count,
    )


def _render_blender_gaps_with_budget(
    context: TimelineRenderContext,
    progress_callback: ProgressCallback | None,
    worker_count: int,
) -> dict[int, Path]:
    plans = _read_blender_plans(context.prepared.blender_plan_paths)
    costs = gap_render_costs(plans)
    representative_index = representative_gap_index(costs)
    _report(
        progress_callback,
        "rendering",
        RENDERING_START,
        f"Benchmarking representative gap {representative_index + 1} before the full render",
    )
    started_at = time.perf_counter()
    representative_path = _render_blender_hidden_segment(
        context,
        representative_index,
        lambda current, total: _report_representative_progress(
            progress_callback, representative_index, len(costs), current, total,
        ),
    )
    elapsed_seconds = _representative_elapsed_seconds(
        context.prepared.blender_plan_paths[representative_index],
        time.perf_counter() - started_at,
    )
    predicted_seconds = predicted_total_seconds(
        costs, representative_index, elapsed_seconds,
    )
    renderer = context.configuration["renderer"]
    maximum_seconds = int(renderer["maximum_predicted_render_seconds"])
    override_enabled = bool(renderer.get("allow_runtime_budget_override", False))
    advisory_exceeded = predicted_seconds > maximum_seconds and not override_enabled
    estimate = {
        "schema_version": 1,
        "status": "warning" if advisory_exceeded else "accepted",
        "representative_gap_index": representative_index,
        "representative_elapsed_seconds": round(elapsed_seconds, 3),
        "predicted_total_seconds": predicted_seconds,
        "maximum_predicted_seconds": maximum_seconds,
        "override_enabled": override_enabled,
        "gap_costs": [
            {
                "gap_index": item.gap_index,
                "target_frames": item.target_frames,
                "detailed_entities": item.detailed_entities,
                "weak_entities": item.weak_entities,
                "weight": round(item.weight, 3),
            }
            for item in costs
        ],
    }
    write_json_file(
        context.prepared.work_dir / "storyboard" / "runtime_estimate.json",
        estimate,
    )
    if len(costs) > 1:
        _require_representative_preview_approval(
            context, representative_index, representative_path,
        )
    _report(
        progress_callback,
        "rendering",
        RENDERING_START + 0.01,
        _runtime_projection_message(predicted_seconds, maximum_seconds, advisory_exceeded),
    )
    remaining_indexes = [
        item.gap_index for item in costs
        if item.gap_index != representative_index
    ]
    rendered_paths = {representative_index: representative_path}
    rendered_paths.update(_render_blender_gap_indexes(
        context,
        progress_callback,
        remaining_indexes,
        len(costs),
        min(worker_count, max(1, len(remaining_indexes))),
        completed_indexes={representative_index},
    ))
    return rendered_paths


def _actor_render_requested(configuration: dict) -> bool:
    return str(
        configuration.get("renderer", {}).get("render_mode", ACTOR_COMPOSITE_MODE)
    ) == ACTOR_COMPOSITE_MODE


def _actor_render_blocked(plans: list[dict]) -> str | None:
    """The reason the actor path cannot serve this video, or None when it can.

    Checked for every gap before any rendering starts: falling back halfway through
    would leave a video whose gaps were produced by two different renderers, which is
    worse for a forensic artifact than being uniformly slower.
    """
    for plan in plans:
        supported, reason = actor_path_is_supported(plan)
        if not supported:
            return f"gap {plan.get('gap_index')}: {reason}"
    return None


def _render_actor_gaps(
    context: TimelineRenderContext,
    progress_callback: ProgressCallback | None,
) -> dict[int, Path] | None:
    """Render every gap as composited actors, or return None to use the legacy path."""
    prepared = context.prepared
    plans = _read_blender_plans(prepared.blender_plan_paths)
    blocked_reason = _actor_render_blocked(plans)
    if blocked_reason is not None:
        LOGGER.warning(
            "Falling back to the full-scene renderer because %s", blocked_reason,
        )
        return None
    _report(
        progress_callback, "rendering", RENDERING_START,
        "Recovering the background plate from visible frames",
    )
    try:
        plate_for_gap = _resolve_pipeline_plates(context, plans, progress_callback)
    except PlateEvidenceError as error:
        LOGGER.warning("Falling back to the full-scene renderer: %s", error)
        return None
    gap_count = len(plans)
    tracker = ParallelGapProgress(progress_callback, gap_count, 1)
    job = _build_actor_job(
        context, plate_for_gap, _exemplar_bank_provider(context, plans, plate_for_gap),
    )
    gap_directories = [
        prepared.work_dir / "gaps" / f"gap_{int(plan['gap_index']):02d}"
        for plan in plans
    ]
    rendered = render_actor_gaps(
        plans, gap_directories, job, context.reuse_work,
        lambda position, completed, total: tracker.report(
            int(plans[position]["gap_index"]), completed, total,
        ),
    )
    _validate_actor_gap_videos(context, plans, rendered)
    return rendered


def _validate_actor_gap_videos(
    context: TimelineRenderContext, plans: list[dict], rendered: dict[int, Path],
) -> None:
    """Every gap must be exactly its hidden range, at source size and rate.

    The compositor guarantees this by construction, so a mismatch means the encoder
    truncated — which would silently shorten the final video against its source.
    """
    info = context.prepared.video_info
    for plan in plans:
        gap_index = int(plan["gap_index"])
        video_path = rendered.get(gap_index)
        if video_path is None:
            raise ActorJobError(f"Gap {gap_index} produced no rendered video")
        validate_video_contract(video_path, VideoContract(
            int(info["width"]), int(info["height"]),
            float(info["fps"]), int(plan["frame_count"]),
        ))


def _pipeline_shots(prepared: PreparedReconstruction) -> tuple[Shot, ...]:
    """The shot objects for this run; a single shot when nothing was segmented."""
    if not prepared.shots:
        return single_shot_timeline(int(prepared.video_info["frames"]))
    return tuple(
        Shot(index=index, start_frame=start, end_frame=end)
        for index, (start, end) in enumerate(prepared.shots)
    )


def _resolve_pipeline_plates(
    context: TimelineRenderContext,
    plans: list[dict],
    progress_callback: ProgressCallback | None = None,
) -> Callable[[int], "numpy.ndarray"]:
    """One background per shot, and a way to ask which one a gap belongs to.

    Building a single plate for the whole video only works when the whole video is one
    scene. Across a cut the per-pixel median averages every camera setup in the file, and
    the ghosted composite it returns becomes the background of every reconstructed frame
    — the largest visible defect in the output, and one no amount of actor quality fixes.
    So the median is taken within a shot, over that shot's own visible frames.
    """
    prepared = context.prepared
    selection = prepared.gap_selection
    detections = read_json_file(prepared.work_dir / "detections.json")
    if not isinstance(detections, list):
        raise PlateEvidenceError("Detections are unavailable for plate extraction")
    shots = _pipeline_shots(prepared)
    visible_ranges = [tuple(item) for item in selection["visible_ranges"]]
    hidden_ranges = [tuple(item) for item in selection["hidden_ranges"]]
    frame_size = (int(prepared.video_info["height"]), int(prepared.video_info["width"]))

    plates: dict[int, "numpy.ndarray"] = {}
    for shot_index in sorted({_shot_for_plan(shots, plan) for plan in plans}):
        shot = shots[shot_index]
        scoped = clip_ranges_to_shot(visible_ranges, shot) if len(shots) > 1 else visible_ranges
        if not scoped:
            raise PlateEvidenceError(
                f"Shot {shot_index} has no visible frames to recover a background from"
            )
        plate = resolve_clean_plate(
            video_path=context.video_path,
            plate_directory=_plate_directory(prepared.work_dir, shot_index, len(shots)),
            visible_ranges=scoped,
            hidden_ranges=hidden_ranges,
            detections=detections,
            detection_stride=int(
                context.configuration.get("yolo", {}).get("frame_stride", 8)
            ),
            video_sha256=str(prepared.video_info["sha256"]),
            reuse_work=context.reuse_work,
            cancellation_check=context.cancellation_check,
        )
        if plate.image.shape[:2] != frame_size:
            raise PlateEvidenceError(
                f"Clean plate {plate.image.shape[:2]} does not match the source frame size"
            )
        if not plate.is_stable:
            # Reported rather than fatal. A ghosted background is a visible quality
            # problem, not a correctness one, and the operator is better served by output
            # plus a stated caveat than by a refusal — but they must be told, not left to
            # notice. Within a single shot this now means real camera motion or unmasked
            # traffic, not simply that the video contained more than one scene.
            _report(
                progress_callback, "rendering", RENDERING_START,
                f"Warning: the background recovered for take {shot_index + 1} is unstable "
                f"(sample disagreement {plate.disagreement:.1f}) — the camera moves or "
                f"unmasked motion crosses the frame, so those gaps will show a blended "
                f"background",
            )
        plates[shot_index] = plate.image

    gap_shots = {int(plan["gap_index"]): _shot_for_plan(shots, plan) for plan in plans}
    return lambda gap_index: plates[gap_shots[gap_index]]


def _exemplar_bank_provider(
    context: TimelineRenderContext,
    plans: list[dict],
    plate_for_gap: Callable[[int], numpy.ndarray],
) -> Callable[[int], dict] | None:
    """Supplies each gap with cut-outs of its own entities from visible footage.

    Banks are built once per shot rather than once per gap: cutting entities out means
    walking the video, and every gap in a take shares both that take's background and
    most of its cast.
    """
    prepared = context.prepared
    if str(
        context.configuration.get("renderer", {}).get("actor_source", ACTOR_SOURCE_EXEMPLAR)
    ) != ACTOR_SOURCE_EXEMPLAR:
        return None
    tracks = {
        str(track["id"]): track for track in prepared.scene_report.get("tracks", [])
    }
    if not tracks:
        return None
    shots = _pipeline_shots(prepared)
    hidden_ranges = [tuple(item) for item in prepared.gap_selection["hidden_ranges"]]
    plans_by_index = {int(plan["gap_index"]): plan for plan in plans}
    shot_of = {index: _shot_for_plan(shots, plan) for index, plan in plans_by_index.items()}
    cache: dict[int, dict] = {}

    def banks_for(gap_index: int) -> dict:
        shot_index = shot_of.get(gap_index, 0)
        if shot_index not in cache:
            wanted: dict[str, list[dict]] = {}
            for other_index, other_plan in plans_by_index.items():
                if shot_of.get(other_index) != shot_index:
                    continue
                for entity in other_plan.get("entities", []):
                    track = tracks.get(str(entity.get("id")))
                    if track is not None:
                        wanted[str(entity["id"])] = track.get("detections", [])
            shot = shots[shot_index]
            try:
                cache[shot_index] = build_exemplar_banks(
                    context.video_path,
                    plate_for_gap(gap_index),
                    wanted,
                    hidden_ranges,
                    shot_bounds=(shot.start_frame, shot.end_frame),
                    cancellation_check=context.cancellation_check,
                )
            except (OSError, RuntimeError) as error:
                # Falling back to modelled geometry is a quality loss, not a failure.
                LOGGER.warning(
                    "Could not cut actors out of take %d: %s", shot_index, error,
                )
                cache[shot_index] = {}
        return cache[shot_index]

    return banks_for


def _plate_directory(work_dir: Path, shot_index: int, shot_count: int) -> Path:
    """Keep the familiar single-plate layout for single-take video."""
    if shot_count <= 1:
        return work_dir / "plate"
    return work_dir / "plate" / f"shot_{shot_index:02d}"


def _shot_for_plan(shots: tuple[Shot, ...], plan: dict) -> int:
    """Which shot a gap sits in.

    Gap selection places every gap wholly inside one shot, so this normally just looks
    the answer up. A selection restored from a cache written before segmentation existed
    can still straddle a cut; that gap is attributed to the shot holding its first frame
    so the run completes with the closest real background rather than failing.
    """
    hidden = plan.get("hidden_range", {})
    start_frame = int(hidden.get("start", 0))
    end_frame = int(hidden.get("end", start_frame))
    spanning = shot_spanning(shots, start_frame, end_frame)
    if spanning is not None:
        return spanning.index
    containing = shot_containing(shots, start_frame)
    if containing is not None:
        LOGGER.warning(
            "Gap %s spans frames %d-%d, which crosses a scene cut; compositing it onto "
            "the background of the take it starts in",
            plan.get("gap_index"), start_frame, end_frame,
        )
        return containing.index
    LOGGER.warning(
        "Gap %s at frames %d-%d begins inside a transition; using the first take's "
        "background", plan.get("gap_index"), start_frame, end_frame,
    )
    return 0


def _build_actor_job(
    context: TimelineRenderContext, plate_for_gap, exemplar_banks_for_gap=None,
) -> ActorRenderJob:
    prepared = context.prepared
    renderer = context.configuration.get("renderer", {})
    job_root = prepared.work_dir / "actor_job"
    return ActorRenderJob(
        blender_executable=find_blender_executable(),
        project_root=ROOT,
        job_root=job_root,
        plate_for_gap=plate_for_gap,
        frame_width=int(prepared.video_info["width"]),
        frame_height=int(prepared.video_info["height"]),
        source_fps=float(prepared.video_info["fps"]),
        stall_timeout_seconds=float(
            renderer.get(
                "gap_render_stall_timeout_seconds", DEFAULT_RENDER_STALL_TIMEOUT_SECONDS,
            )
        ),
        environment_overlay=kernel_cache_environment(job_root / "kernel_cache"),
        library_directory=ROOT / "assets" / "actors",
        cancellation_check=context.cancellation_check,
        exemplar_banks_for_gap=exemplar_banks_for_gap,
    )


def _runtime_projection_message(
    predicted_seconds: float,
    maximum_seconds: int,
    advisory_exceeded: bool,
) -> str:
    predicted_minutes = predicted_seconds / 60.0
    if not advisory_exceeded:
        return f"Projected Blender runtime: {predicted_minutes:.1f} minutes"
    return (
        f"Projected Blender runtime: {predicted_minutes:.1f} minutes, above the "
        f"{maximum_seconds / 60.0:.1f}-minute advisory; continuing after approval"
    )


def _require_representative_preview_approval(
    context: TimelineRenderContext,
    representative_index: int,
    representative_path: Path,
) -> None:
    renderer = context.configuration["renderer"]
    if not renderer.get("interactive_preview_approval", False):
        return
    plan_path = context.prepared.blender_plan_paths[representative_index]
    signature = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    approval_path = (
        context.prepared.work_dir
        / "storyboard"
        / "representative_preview_approved.json"
    )
    if not preview_is_approved(approval_path, signature):
        raise RepresentativePreviewApprovalRequired(
            representative_path, approval_path, signature,
        )


def _render_blender_gap_indexes(
    context: TimelineRenderContext,
    progress_callback: ProgressCallback | None,
    gap_indexes: list[int],
    gap_count: int,
    worker_count: int,
    completed_indexes: set[int] | None = None,
) -> dict[int, Path]:
    if not gap_indexes:
        return {}
    progress_tracker = ParallelGapProgress(progress_callback, gap_count, worker_count)
    for completed_index in completed_indexes or set():
        progress_tracker.report(completed_index, 1, 1)
    abort_event = threading.Event()
    worker_context = replace(
        context,
        cancellation_check=_combined_cancellation_check(context.cancellation_check, abort_event),
    )
    executor = ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="blender-gap")
    futures: dict[Future[Path], int] = {
        executor.submit(
            _render_blender_hidden_segment,
            worker_context,
            gap_index,
            lambda current, total, index=gap_index: progress_tracker.report(index, current, total),
        ): gap_index
        for gap_index in gap_indexes
    }
    rendered_paths: dict[int, Path] = {}
    try:
        already_completed = len(completed_indexes or set())
        for completed_count, future in enumerate(as_completed(futures), start=already_completed + 1):
            gap_index = futures[future]
            try:
                rendered_paths[gap_index] = future.result()
            except Exception:
                abort_event.set()
                for pending_future in futures:
                    pending_future.cancel()
                raise
            _report_parallel_render_progress(progress_callback, completed_count, gap_count, worker_count)
    finally:
        abort_event.set()
        executor.shutdown(wait=True, cancel_futures=True)
    return rendered_paths


def _read_blender_plans(plan_paths: list[Path]) -> list[dict]:
    plans = [read_json_file(path) for path in plan_paths]
    if not all(isinstance(plan, dict) for plan in plans):
        raise ValueError("A Blender reconstruction plan is missing or invalid")
    return [plan for plan in plans if isinstance(plan, dict)]


def _representative_elapsed_seconds(plan_path: Path, wall_seconds: float) -> float:
    report = read_json_file(plan_path.parent / "render_report.json")
    reported_seconds = (
        float(report.get("elapsed_seconds", 0.0))
        if isinstance(report, dict)
        else 0.0
    )
    return max(0.001, wall_seconds, reported_seconds)


def _runtime_budget_enabled(configuration: dict) -> bool:
    return bool(
        configuration.get("renderer", {}).get("runtime_budget_enabled", False)
    )


def _report_representative_progress(
    callback: ProgressCallback | None,
    gap_index: int,
    gap_count: int,
    current_frame: int,
    total_frames: int,
) -> None:
    fraction = current_frame / max(1, total_frames)
    progress = RENDERING_START + RENDERING_SPAN * BLENDER_RENDER_PROGRESS_SHARE * fraction / gap_count
    _report(
        callback,
        "rendering",
        progress,
        f"Benchmarking gap {gap_index + 1} of {gap_count}: frame {current_frame} of {total_frames}",
    )


def _combined_cancellation_check(
    external_check: CancellationCheck | None,
    abort_event: threading.Event,
) -> CancellationCheck:
    return lambda: abort_event.is_set() or (external_check is not None and external_check())


def _parallel_gap_renderer_count(configuration: dict, gap_count: int) -> int:
    renderer = configuration.get("renderer", {})
    configured_count = int(
        renderer.get("max_parallel_gap_renders", DEFAULT_PARALLEL_GAP_RENDERERS)
    )
    if renderer.get("engine") == "CYCLES":
        configured_count = min(
            configured_count, int(renderer.get("maximum_gpu_workers", 1)),
        )
    return max(1, min(configured_count, max(1, gap_count)))


def _scaled_render_dimension(source_dimension: int, configuration: dict) -> int:
    scale_percent = int(
        configuration.get("renderer", {}).get("production_scale_percent", 100),
    )
    scaled_dimension = max(2, round(source_dimension * scale_percent / 100.0))
    return scaled_dimension if scaled_dimension % 2 == 0 else scaled_dimension + 1


def _report_parallel_render_progress(
    callback: ProgressCallback | None,
    completed_count: int,
    gap_count: int,
    worker_count: int,
) -> None:
    fraction = completed_count / max(1, gap_count)
    progress = RENDERING_START + (RENDERING_SPAN * BLENDER_RENDER_PROGRESS_SHARE * fraction)
    detail = f"Rendered inferred gap {completed_count} of {gap_count} with {worker_count} parallel workers"
    _report(callback, "rendering", progress, detail)


def _timeline_render_start() -> float:
    return RENDERING_START + (RENDERING_SPAN * BLENDER_RENDER_PROGRESS_SHARE)


def _evaluate(
    video_path: Path,
    items: list[dict],
    config: dict,
    cancellation_check: CancellationCheck | None,
) -> dict:
    yolo_config = config.get("yolo", {})
    evaluation_config = config.get("evaluation", {})
    if not evaluation_config.get("enabled", True):
        return {"mode": "disabled"}
    return evaluate_reconstructions(
        items,
        str(video_path),
        yolo_config.get("model", "yolo26m.pt"),
        yolo_class_ids(config),
        normalize_confidence(yolo_config.get("confidence", 0.3)),
        evaluation_config.get("frame_stride", 12),
        cancellation_check,
    )


def _build_scene_report(
    detections: list[dict],
    info: dict,
    selection: dict,
    video_path: Path,
    shots: tuple[tuple[int, int], ...] = (),
) -> dict:
    scene_report = summarize_scene(
        detections, info["fps"], info["width"],
        [tuple(item) for item in selection["hidden_ranges"]],
        shots=shots,
    )
    scene_report["video"] = {"path": str(video_path), **info}
    scene_report["visible_ranges"] = [
        {"start": start, "end": end} for start, end in selection["visible_ranges"]
    ]
    return scene_report


def _prepare_reconstruction(
    video_path: Path,
    options: PipelineOptions,
    rng: random.Random,
    progress_callback: ProgressCallback | None,
) -> PreparedReconstruction:
    raise_if_cancelled(options.cancellation_check)
    config = options.config_data
    _report(progress_callback, "validating", VALIDATION_PROGRESS, "Checking runtime tools and video metadata")
    _validate_runtime_dependencies()
    info = video_info(video_path)
    validate_constant_frame_rate(video_path, info["fps"], options.cancellation_check)
    info = {**info, "sha256": _source_video_sha256(video_path, options.cancellation_check)}
    options.output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = options.output_dir / "_work" / f"{video_path.stem}_{info['sha256'][:12]}"
    _report(progress_callback, "segmenting_shots", SHOT_SEGMENTATION_PROGRESS, "Finding scene cuts")
    shot_report, shots = _load_shot_timeline(
        video_path, work_dir, info, options.reuse_work, options.cancellation_check,
    )
    _report(
        progress_callback,
        "selecting_gaps",
        GAP_SELECTION_PROGRESS,
        _gap_selection_detail(shot_report),
    )
    selection = _load_selection(work_dir, info, config, rng, options.reuse_work, shots)
    segment_paths = reserve_timeline_segment_paths(
        selection["timeline"], work_dir / "segments", progress_callback,
    )
    detections = _load_detections(
        video_path,
        work_dir,
        selection,
        config,
        options.reuse_work,
        progress_callback,
        options.cancellation_check,
    )
    _report(progress_callback, "planning", BASE_PLANNING_PROGRESS, "Building bounded reconstruction hypotheses")
    scene_report, blender_plan_paths = _prepare_scene_and_blender_plans(
        video_path, options, info, selection, work_dir, detections, tuple(shots),
    )
    if blender_plan_paths:
        _report(progress_callback, "extracting_clues", CLUE_EXTRACTION_PROGRESS, "Writing the visible-only evidence ledger")
        _report(progress_callback, "reasoning", REASONING_PROGRESS, "Selecting evidence-grounded motion hypotheses")
        reasoning_result = reason_about_reconstruction(
            scene_report,
            blender_plan_paths,
            work_dir,
            {**config["reasoning"], "renderer": config.get("renderer", {})},
            options.reuse_work,
            options.cancellation_check,
        )
        _report(
            progress_callback,
            "validating_decisions",
            DECISION_VALIDATION_PROGRESS,
            f"Validated the {reasoning_result.mode.replace('_', ' ')} decision trace",
        )
    _report(
        progress_callback,
        "validating_decisions",
        PLANNING_PROGRESS,
        "Finalized validated reconstruction plans",
    )
    return PreparedReconstruction(
        info, selection, segment_paths, scene_report, work_dir, blender_plan_paths,
        tuple(shots), shot_report,
    )


def _gap_selection_detail(shot_report: dict) -> str:
    shot_count = int(shot_report.get("shot_count", 1))
    if shot_count <= 1:
        return "Placing evidence gaps across a single continuous take"
    return f"Placing evidence gaps inside {shot_count} separate takes"


def _prepare_scene_and_blender_plans(
    video_path: Path,
    options: PipelineOptions,
    info: dict,
    selection: dict,
    work_dir: Path,
    detections: list[dict],
    shots: tuple[tuple[int, int], ...] = (),
) -> tuple[dict, list[Path]]:
    config = options.config_data
    scene_report = _build_scene_report(detections, info, selection, video_path, shots)
    validate_visible_evidence_only(scene_report)
    blender_assets = prepare_blender_assets(
        video_path,
        scene_report,
        selection["hidden_ranges"],
        work_dir,
        int(config.get("scene", {}).get("max_render_entities", 3)),
        config.get("renderer", {}),
        options.cancellation_check,
        shots,
    )
    scene_report = blender_assets.scene_report
    write_json(work_dir / "scene_report.json", scene_report)
    return scene_report, blender_assets.plan_paths


def _render_and_finalize(
    video_path: Path,
    options: PipelineOptions,
    prepared: PreparedReconstruction,
    progress_callback: ProgressCallback | None,
) -> Path:
    _report(progress_callback, "rendering", RENDERING_START, "Rendering evidence and inferred segments")
    render_context = _prepare_timeline_render_context(video_path, options, prepared, progress_callback)
    sequence, evaluation_items = _render_timeline(render_context, progress_callback)
    _materialize_hidden_truth(video_path, prepared, options.cancellation_check)
    _report(progress_callback, "evaluating", EVALUATION_PROGRESS, "Evaluating completed reconstructions")
    diagnostic_report = _evaluate(
        video_path, evaluation_items, options.config_data, options.cancellation_check,
    )
    write_json(prepared.work_dir / "diagnostic_report.json", diagnostic_report)
    final_output = _stitch_final_output(
        video_path, options, prepared, sequence, progress_callback,
    )
    presentation = build_presentation_manifest(
        prepared.video_info,
        prepared.gap_selection,
        prepared.scene_report,
        prepared.blender_plan_paths,
        prepared.work_dir,
        final_output,
    )
    write_presentation_manifest(
        presentation, prepared.work_dir / "presentation_manifest.json",
    )
    _report(
        progress_callback,
        "completed",
        COMPLETED_PROGRESS,
        "Reconstruction and judge presentation complete",
    )
    return final_output


def _prepare_timeline_render_context(
    video_path: Path,
    options: PipelineOptions,
    prepared: PreparedReconstruction,
    progress_callback: ProgressCallback | None,
) -> TimelineRenderContext:
    render_context = TimelineRenderContext(
        video_path,
        prepared,
        options.config_data,
        options.reuse_work,
        {},
        options.cancellation_check,
    )
    rendered_gap_paths = _render_blender_gaps(render_context, progress_callback)
    return TimelineRenderContext(
        video_path,
        prepared,
        options.config_data,
        options.reuse_work,
        rendered_gap_paths,
        options.cancellation_check,
    )


def _stitch_final_output(
    video_path: Path,
    options: PipelineOptions,
    prepared: PreparedReconstruction,
    sequence: list[str],
    progress_callback: ProgressCallback | None,
) -> Path:
    _report(progress_callback, "stitching", STITCHING_PROGRESS, "Stitching the final video")
    video_only_output = prepared.work_dir / "stitch" / "video_only.mp4"
    video_only_output.parent.mkdir(parents=True, exist_ok=True)
    stitch_sequence(
        sequence,
        str(video_only_output),
        fps=prepared.video_info["fps"],
        cancellation_check=options.cancellation_check,
    )
    final_output = options.output_dir / f"{video_path.stem}_reconstructed.mp4"
    encode_with_source_audio(
        video_only_output, video_path, final_output, options.cancellation_check,
    )
    validate_video_contract(final_output, VideoContract(
        prepared.video_info["width"],
        prepared.video_info["height"],
        prepared.video_info["fps"],
        prepared.video_info["frames"],
    ))
    _report(progress_callback, "completed", 0.99, "Preparing judge presentation")
    return final_output


def _materialize_hidden_truth(
    video_path: Path,
    prepared: PreparedReconstruction,
    cancellation_check: CancellationCheck | None,
) -> None:
    for gap_index, hidden_range in enumerate(prepared.gap_selection["hidden_ranges"]):
        raise_if_cancelled(cancellation_check)
        truth_path = prepared.segment_paths[("hidden", gap_index)]
        write_video_range(
            video_path,
            int(hidden_range[0]),
            int(hidden_range[1]),
            truth_path,
            cancellation_check,
        )


def _validate_runtime_dependencies() -> None:
    find_media_tool("ffmpeg")
    find_media_tool("ffprobe")
    find_blender_executable()


def process_video(
    video_path: Path,
    options: PipelineOptions,
    rng: random.Random,
    progress_callback: ProgressCallback | None = None,
) -> Path:
    prepared = _prepare_reconstruction(video_path, options, rng, progress_callback)
    return _render_and_finalize(video_path, options, prepared, progress_callback)
