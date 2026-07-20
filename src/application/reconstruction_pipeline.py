"""Coordinates evidence analysis, reconstruction rendering, and final video assembly."""

import json
import random
from dataclasses import dataclass
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
from domain.configuration import load_validated_configuration
from infrastructure.media_tools import VideoContract, encode_with_source_audio, validate_video_contract


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "config" / "reconstruction_config.json"
ProgressCallback = Callable[[str, float, str], None]
VALIDATION_PROGRESS = 0.01
GAP_SELECTION_PROGRESS = 0.04
SEGMENT_PREPARATION_START = 0.06
SEGMENT_PREPARATION_SPAN = 0.07
DETECTION_START = 0.13
DETECTION_SPAN = 0.35
PLANNING_PROGRESS = 0.50
RENDERING_START = 0.55
RENDERING_SPAN = 0.27
EVALUATION_PROGRESS = 0.85
STITCHING_PROGRESS = 0.94
COMPLETED_PROGRESS = 1.0


@dataclass(frozen=True)
class PipelineOptions:
    config_data: dict
    output_dir: Path
    reuse_work: bool = False
    renderer_mode: str = "blender"


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


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict:
    return load_validated_configuration(path)


def _report(callback: ProgressCallback | None, stage: str, progress: float, detail: str) -> None:
    if callback is not None:
        callback(stage, max(0.0, min(1.0, progress)), detail)


def yolo_class_ids(config: dict) -> list[int]:
    classes = config.get("yolo", {}).get("classes", {})
    return sorted(int(class_id) for class_id in classes) if classes else sorted(RELEVANT_COCO_CLASSES)


def normalize_confidence(value: float) -> float:
    confidence = float(value)
    if confidence > 1.0:
        confidence /= 100.0
    return max(0.0, min(1.0, confidence))


def video_info(video_path: Path) -> dict:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video_path.name}")
    info = {
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS) or 30.0),
        "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    capture.release()
    if info["frames"] < 4 or info["width"] < 1 or info["height"] < 1:
        raise ValueError(f"Video is unreadable or too short: {video_path.name}")
    return info


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2)


def write_video_range(video_path: Path, start_frame: int, end_frame: int, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video_path.name}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 30.0)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_frame))
    for _ in range(start_frame, end_frame + 1):
        success, frame = capture.read()
        if not success:
            break
        writer.write(frame)
    capture.release()
    writer.release()
    return output_path


def _segment_path(segment_dir: Path, segment: dict) -> Path:
    return segment_dir / f"{segment['kind']}_{segment['index']:02d}_{segment['start']}_{segment['end']}.mp4"


def write_timeline_segments(
    video_path: Path,
    timeline: list[dict],
    segment_dir: Path,
    reuse_work: bool,
    progress_callback: ProgressCallback | None,
) -> dict[tuple[str, int], Path]:
    paths: dict[tuple[str, int], Path] = {}
    segment_total = max(1, len(timeline))
    for item_index, segment in enumerate(timeline):
        output_path = _segment_path(segment_dir, segment)
        if segment["kind"] == "hidden":
            paths[(segment["kind"], segment["index"])] = output_path
            continue
        if not reuse_work or not output_path.exists():
            write_video_range(video_path, segment["start"], segment["end"], output_path)
        paths[(segment["kind"], segment["index"])] = output_path
        progress = SEGMENT_PREPARATION_START + (SEGMENT_PREPARATION_SPAN * ((item_index + 1) / segment_total))
        _report(progress_callback, "preparing", progress, f"Prepared timeline segment {item_index + 1} of {segment_total}")
    return paths


def _new_selection(info: dict, gap_config: dict, rng: random.Random) -> dict:
    return choose_hidden_gaps(
        total_frames=info["frames"],
        fps=info["fps"],
        rng=rng,
        missing_fraction=gap_config.get("missing_fraction", 0.25),
        min_gap_seconds=gap_config.get("min_seconds", 1.0),
        max_gap_seconds=gap_config.get("max_seconds", 3.0),
        context_seconds=gap_config.get("context_seconds", 2.0),
    )


def _load_selection(work_dir: Path, info: dict, config: dict, rng: random.Random, reuse_work: bool) -> dict:
    selection_path = work_dir / "gap_selection.json"
    if reuse_work and selection_path.exists():
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if selection.get("policy") == "distributed_short_evidence_gaps":
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
) -> list[dict]:
    detections_path = work_dir / "detections.json"
    if reuse_work and detections_path.exists():
        _report(progress_callback, "detecting", DETECTION_START + DETECTION_SPAN, "Reused compatible detection cache")
        return json.loads(detections_path.read_text(encoding="utf-8"))
    yolo_config = config.get("yolo", {})
    tracker_config = yolo_config.get("tracker_config")
    if tracker_config and not Path(tracker_config).is_absolute():
        tracker_config = str(ROOT / tracker_config)

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
    )
    write_json(detections_path, detections)
    return detections


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
    reuse_work: bool,
) -> None:
    if reuse_work and output_path.exists() and output_path.stat().st_size >= 1_000:
        return
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
    )


def _render_timeline(
    context: TimelineRenderContext,
    progress_callback: ProgressCallback | None,
) -> tuple[list[str], list[dict]]:
    sequence: list[str] = []
    evaluation_items: list[dict] = []
    timeline = context.prepared.gap_selection["timeline"]
    for item_index, segment in enumerate(timeline):
        output_path, evaluation_item = _render_timeline_segment(context, segment)
        sequence.append(str(output_path))
        if evaluation_item is not None:
            evaluation_items.append(evaluation_item)
        fraction = (item_index + 1) / max(1, len(timeline))
        progress = RENDERING_START + (RENDERING_SPAN * fraction)
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
            context.configuration, visible_count, context.reuse_work,
        )
        return output_path, None
    return _render_hidden_segment(context, segment)


def _render_hidden_segment(context: TimelineRenderContext, segment: dict) -> tuple[Path, dict]:
    prepared = context.prepared
    gap_index = segment["index"]
    if context.renderer_mode == "blender":
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


def _render_blender_hidden_segment(context: TimelineRenderContext, gap_index: int) -> Path:
    prepared = context.prepared
    gap_directory = prepared.work_dir / "gaps" / f"gap_{gap_index:02d}"
    output_path = render_blender_gap(
        ROOT, prepared.blender_plan_paths[gap_index], gap_directory, context.reuse_work,
    )
    hidden_range = prepared.gap_selection["hidden_ranges"][gap_index]
    expected_contract = VideoContract(
        prepared.video_info["width"],
        prepared.video_info["height"],
        prepared.video_info["fps"],
        int(hidden_range[1]) - int(hidden_range[0]) + 1,
    )
    validate_video_contract(output_path, expected_contract)
    return output_path


def _render_2d_hidden_segment(context: TimelineRenderContext, gap_index: int) -> Path:
    prepared = context.prepared
    output_path = prepared.work_dir / "gaps" / f"gap_{gap_index:02d}" / "evidence_reconstruction.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    render_evidence_reconstruction(
        str(output_path), str(context.video_path), prepared.reconstruction_plans[gap_index], prepared.scene_report,
        prepared.video_info["width"], prepared.video_info["height"], prepared.video_info["fps"],
        context.configuration.get("visualization", {}),
    )
    return output_path


def _evaluate(video_path: Path, items: list[dict], config: dict) -> dict:
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
    config = options.config_data
    _report(progress_callback, "validating", VALIDATION_PROGRESS, "Validating video metadata")
    info = video_info(video_path)
    options.output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = options.output_dir / "_work" / video_path.stem
    _report(progress_callback, "selecting_gaps", GAP_SELECTION_PROGRESS, "Selecting distributed 1–3 second gaps")
    selection = _load_selection(work_dir, info, config, rng, options.reuse_work)
    segment_paths = write_timeline_segments(
        video_path, selection["timeline"], work_dir / "segments", options.reuse_work, progress_callback,
    )
    detections = _load_detections(
        video_path, work_dir, selection, config, options.reuse_work, progress_callback,
    )
    _report(progress_callback, "planning", PLANNING_PROGRESS, "Building scene intelligence and reconstruction plans")
    scene_report = _build_scene_report(detections, info, selection, video_path)
    blender_plan_paths: list[Path] = []
    if options.renderer_mode == "blender":
        blender_assets = prepare_blender_assets(
            video_path,
            scene_report,
            selection["hidden_ranges"],
            work_dir,
            int(config.get("scene", {}).get("max_render_entities", 12)),
            config.get("renderer", {}),
        )
        scene_report = blender_assets.scene_report
        blender_plan_paths = blender_assets.plan_paths
    write_json(work_dir / "scene_report.json", scene_report)
    plans = [] if options.renderer_mode == "blender" else _build_plans(
        scene_report, selection, info, work_dir, config.get("scene", {})
    )
    return PreparedReconstruction(
        info, selection, segment_paths, plans, scene_report, work_dir, blender_plan_paths,
    )


def _render_and_finalize(
    video_path: Path,
    options: PipelineOptions,
    prepared: PreparedReconstruction,
    progress_callback: ProgressCallback | None,
) -> Path:
    _report(progress_callback, "rendering", RENDERING_START, "Rendering evidence and inferred segments")
    render_context = TimelineRenderContext(
        video_path, prepared, options.config_data, options.reuse_work, options.renderer_mode,
    )
    sequence, evaluation_items = _render_timeline(render_context, progress_callback)
    _materialize_hidden_truth(video_path, prepared)
    _report(progress_callback, "evaluating", EVALUATION_PROGRESS, "Evaluating completed reconstructions")
    accuracy_report = _evaluate(video_path, evaluation_items, options.config_data)
    write_json(prepared.work_dir / "accuracy_report.json", accuracy_report)
    _report(progress_callback, "stitching", STITCHING_PROGRESS, "Stitching the final video")
    video_only_output = prepared.work_dir / "stitch" / "video_only.mp4"
    video_only_output.parent.mkdir(parents=True, exist_ok=True)
    stitch_sequence(sequence, str(video_only_output), fps=prepared.video_info["fps"])
    final_output = options.output_dir / f"{video_path.stem}_reconstructed.mp4"
    encode_with_source_audio(video_only_output, video_path, final_output)
    validate_video_contract(final_output, VideoContract(
        prepared.video_info["width"],
        prepared.video_info["height"],
        prepared.video_info["fps"],
        prepared.video_info["frames"],
    ))
    _report(progress_callback, "completed", COMPLETED_PROGRESS, "Reconstruction complete")
    return final_output


def _materialize_hidden_truth(video_path: Path, prepared: PreparedReconstruction) -> None:
    for gap_index, hidden_range in enumerate(prepared.gap_selection["hidden_ranges"]):
        truth_path = prepared.segment_paths[("hidden", gap_index)]
        write_video_range(video_path, int(hidden_range[0]), int(hidden_range[1]), truth_path)


def process_video(
    video_path: Path,
    options: PipelineOptions,
    rng: random.Random,
    progress_callback: ProgressCallback | None = None,
) -> Path:
    prepared = _prepare_reconstruction(video_path, options, rng, progress_callback)
    return _render_and_finalize(video_path, options, prepared, progress_callback)
