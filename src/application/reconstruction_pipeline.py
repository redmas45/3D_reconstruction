"""Coordinates evidence analysis, reconstruction rendering, and final video assembly."""

import random
import hashlib
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import cv2

from detect import RELEVANT_COCO_CLASSES, detect_scene_objects
from evaluate import evaluate_reconstructions
from evidence_compositor import render_evidence_reconstruction
from gap_selector import choose_hidden_gaps
from reconstruction_plan import build_reconstruction_plan
from scene_intelligence import summarize_scene
from stitch import stitch_sequence
from visual_output import render_annotated_visible_chunk
from application.blender_pipeline import prepare_blender_assets, render_blender_gap
from application.evidence_reasoning import reason_about_reconstruction
from domain.cancellation import CancellationCheck, raise_if_cancelled
from domain.configuration import load_validated_configuration
from domain.evidence_contract import validate_visible_evidence_only
from domain.reconstruction_cache import (
    cached_detections_are_valid as _cached_detections_are_valid,
    gap_cache_configuration as _gap_cache_configuration,
    selection_cache_is_compatible as _selection_cache_is_compatible,
    source_video_contract as _source_video_contract,
)
from domain.render_runtime_budget import (
    RepresentativePreviewApprovalRequired,
    enforce_runtime_budget,
    gap_render_costs,
    preview_is_approved,
    predicted_total_seconds,
    representative_gap_index,
)
from infrastructure.blender_runner import DEFAULT_RENDER_STALL_TIMEOUT_SECONDS, find_blender_executable
from infrastructure.json_files import read_json_file, write_json_file
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


@dataclass(frozen=True)
class PipelineOptions:
    config_data: dict
    output_dir: Path
    reuse_work: bool = False
    renderer_mode: str = "blender"
    cancellation_check: CancellationCheck | None = None


@dataclass(frozen=True)
class PreparedReconstruction:
    video_info: dict
    gap_selection: dict
    segment_paths: dict[tuple[str, int], Path]
    reconstruction_plans: list[dict]
    scene_report: dict
    work_dir: Path
    blender_plan_paths: list[Path]


@dataclass(frozen=True)
class TimelineRenderContext:
    video_path: Path
    prepared: PreparedReconstruction
    configuration: dict
    reuse_work: bool
    renderer_mode: str
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


def _new_selection(info: dict, gap_config: dict, rng: random.Random) -> dict:
    selection = choose_hidden_gaps(
        total_frames=info["frames"],
        fps=info["fps"],
        rng=rng,
        missing_fraction=gap_config.get("missing_fraction", 0.25),
        min_gap_seconds=gap_config.get("min_seconds", 1.0),
        max_gap_seconds=gap_config.get("max_seconds", 3.0),
        context_seconds=gap_config.get("context_seconds", 2.0),
    )
    return {
        **selection,
        "source_video_contract": _source_video_contract(info),
        "gap_configuration": _gap_cache_configuration(gap_config),
    }


def _load_selection(work_dir: Path, info: dict, config: dict, rng: random.Random, reuse_work: bool) -> dict:
    selection_path = work_dir / "gap_selection.json"
    if reuse_work and selection_path.exists():
        selection = read_json_file(selection_path)
        if _selection_cache_is_compatible(selection, info, config.get("gap", {})):
            return selection
    selection = _new_selection(info, config.get("gap", {}), rng)
    write_json(selection_path, selection)
    return selection


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
        progress_callback=report_detection,
        cancellation_check=cancellation_check,
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
    }


def _build_plans(scene_report: dict, selection: dict, info: dict, work_dir: Path, scene_config: dict) -> list[dict]:
    plans: list[dict] = []
    for gap_index, hidden_range in enumerate(selection["hidden_ranges"]):
        plan = build_reconstruction_plan(
            scene_report,
            tuple(hidden_range),
            info["fps"],
            max_entities=scene_config.get("max_render_entities", 12),
            min_track_frames=scene_config.get("min_track_frames", 2),
        )
        plan["gap_index"] = gap_index
        plans.append(plan)
        write_json(work_dir / "gaps" / f"gap_{gap_index:02d}" / "reconstruction_plan.json", plan)
    manifest = [
        {
            "gap_index": item["gap_index"],
            "hidden_range": item["hidden_range"],
            "entities": len(item["entities"]),
            "confidence": item["overall_confidence"],
        }
        for item in plans
    ]
    write_json(work_dir / "reconstruction_plans.json", manifest)
    return plans


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
    timeline_start = _timeline_render_start(context.renderer_mode)
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
    if context.renderer_mode == "blender":
        output_path = context.blender_rendered_paths.get(gap_index)
        if output_path is None:
            output_path = _render_blender_hidden_segment(context, gap_index)
    else:
        output_path = _render_2d_hidden_segment(context, gap_index)
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
    expected_contract = VideoContract(
        _scaled_render_dimension(prepared.video_info["width"], context.configuration),
        _scaled_render_dimension(prepared.video_info["height"], context.configuration),
        prepared.video_info["fps"],
        int(hidden_range[1]) - int(hidden_range[0]) + 1,
    )
    validate_video_contract(output_path, expected_contract)
    return output_path


def _render_blender_gaps(
    context: TimelineRenderContext,
    progress_callback: ProgressCallback | None,
) -> dict[int, Path]:
    if context.renderer_mode != "blender":
        return {}
    raise_if_cancelled(context.cancellation_check)
    gap_count = len(context.prepared.gap_selection["hidden_ranges"])
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
    estimate = {
        "schema_version": 1,
        "status": "accepted" if predicted_seconds <= maximum_seconds or override_enabled else "rejected",
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
        enforce_runtime_budget(predicted_seconds, maximum_seconds, override_enabled)
        _require_representative_preview_approval(
            context, representative_index, representative_path,
        )
    _report(
        progress_callback,
        "rendering",
        RENDERING_START + 0.01,
        f"Projected Blender runtime: {predicted_seconds / 60.0:.1f} minutes; budget accepted",
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
    scale_percent = int(configuration.get("renderer", {}).get("production_scale_percent", 100))
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


def _timeline_render_start(renderer_mode: str) -> float:
    if renderer_mode != "blender":
        return RENDERING_START
    return RENDERING_START + (RENDERING_SPAN * BLENDER_RENDER_PROGRESS_SHARE)


def _render_2d_hidden_segment(context: TimelineRenderContext, gap_index: int) -> Path:
    prepared = context.prepared
    output_path = prepared.work_dir / "gaps" / f"gap_{gap_index:02d}" / "evidence_reconstruction.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    render_evidence_reconstruction(
        str(output_path), str(context.video_path), prepared.reconstruction_plans[gap_index], prepared.scene_report,
        prepared.video_info["width"], prepared.video_info["height"], prepared.video_info["fps"],
        context.configuration.get("visualization", {}),
        context.cancellation_check,
    )
    return output_path


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


def _build_scene_report(detections: list[dict], info: dict, selection: dict, video_path: Path) -> dict:
    scene_report = summarize_scene(
        detections, info["fps"], info["width"], [tuple(item) for item in selection["hidden_ranges"]],
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
    _validate_runtime_dependencies(options.renderer_mode)
    info = video_info(video_path)
    validate_constant_frame_rate(video_path, info["fps"], options.cancellation_check)
    info = {**info, "sha256": _source_video_sha256(video_path, options.cancellation_check)}
    options.output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = options.output_dir / "_work" / f"{video_path.stem}_{info['sha256'][:12]}"
    _report(progress_callback, "selecting_gaps", GAP_SELECTION_PROGRESS, "Selecting distributed 1–3 second gaps")
    selection = _load_selection(work_dir, info, config, rng, options.reuse_work)
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
    scene_report, blender_plan_paths, plans = _prepare_scene_and_plans(
        video_path, options, info, selection, work_dir, detections,
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
        info, selection, segment_paths, plans, scene_report, work_dir, blender_plan_paths,
    )


def _prepare_scene_and_plans(
    video_path: Path,
    options: PipelineOptions,
    info: dict,
    selection: dict,
    work_dir: Path,
    detections: list[dict],
) -> tuple[dict, list[Path], list[dict]]:
    config = options.config_data
    scene_report = _build_scene_report(detections, info, selection, video_path)
    validate_visible_evidence_only(scene_report)
    blender_plan_paths: list[Path] = []
    if options.renderer_mode == "blender":
        blender_assets = prepare_blender_assets(
            video_path,
            scene_report,
            selection["hidden_ranges"],
            work_dir,
            int(config.get("scene", {}).get("max_render_entities", 12)),
            config.get("renderer", {}),
            options.cancellation_check,
        )
        scene_report = blender_assets.scene_report
        blender_plan_paths = blender_assets.plan_paths
    write_json(work_dir / "scene_report.json", scene_report)
    plans = [] if options.renderer_mode == "blender" else _build_plans(
        scene_report, selection, info, work_dir, config.get("scene", {})
    )
    return scene_report, blender_plan_paths, plans


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
    return _stitch_final_output(video_path, options, prepared, sequence, progress_callback)


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
        options.renderer_mode,
        {},
        options.cancellation_check,
    )
    rendered_gap_paths = _render_blender_gaps(render_context, progress_callback)
    return TimelineRenderContext(
        video_path,
        prepared,
        options.config_data,
        options.reuse_work,
        options.renderer_mode,
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
    _report(progress_callback, "completed", COMPLETED_PROGRESS, "Reconstruction complete")
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


def _validate_runtime_dependencies(renderer_mode: str) -> None:
    if renderer_mode not in {"blender", "2d"}:
        raise ValueError("Renderer mode must be 'blender' or '2d'")
    find_media_tool("ffmpeg")
    find_media_tool("ffprobe")
    if renderer_mode == "blender":
        find_blender_executable()


def process_video(
    video_path: Path,
    options: PipelineOptions,
    rng: random.Random,
    progress_callback: ProgressCallback | None = None,
) -> Path:
    prepared = _prepare_reconstruction(video_path, options, rng, progress_callback)
    return _render_and_finalize(video_path, options, prepared, progress_callback)
