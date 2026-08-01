"""Prepare the browser scene plan from visible evidence only.

This module is deliberately renderer-neutral. It owns shot calibration, identity
registration, and per-gap scene manifests; the browser is responsible for drawing
the resulting Three.js scene. Server-side Blender planning was kept in the legacy
archive and is not imported by the active pipeline.
"""

import json
from dataclasses import dataclass
from pathlib import Path

from domain.cancellation import CancellationCheck, raise_if_cancelled
from domain.evidence_contract import validate_visible_evidence_only
from domain.identity_registry import build_identity_registry, write_identity_registry
from domain.reconstruction_plan_v2 import build_reconstruction_plan_v2, write_reconstruction_plan_v2
from domain.shot_scene import combine_shot_motion, scene_report_for_shot
from infrastructure.camera_motion_estimator import estimate_camera_motion
from infrastructure.json_files import write_json_file
from infrastructure.video_frames import export_forensic_context_frame


@dataclass(frozen=True)
class PreparedSceneAssets:
    """Calibrated report and browser scene-plan paths for one source video."""

    scene_report: dict
    plan_paths: list[Path]
    identity_registry_path: Path


def prepare_scene_assets(
    video_path: Path,
    scene_report: dict,
    hidden_ranges: list[list[int]],
    work_directory: Path,
    maximum_entities: int,
    render_configuration: dict,
    cancellation_check: CancellationCheck | None = None,
    shots: tuple[tuple[int, int], ...] = (),
) -> PreparedSceneAssets:
    """Calibrate each shot and write the validated plans consumed by Three.js."""
    raise_if_cancelled(cancellation_check)
    validate_visible_evidence_only(scene_report)
    shot_bounds = _shot_bounds(shots, int(scene_report["video"]["frames"]))
    scoped_reports, motion_by_shot = _calibrate_each_shot(
        video_path, scene_report, shot_bounds, cancellation_check,
    )
    calibrated_report = {
        **scene_report,
        "camera_motion_report": combine_shot_motion(motion_by_shot),
        "camera_motion_by_shot": motion_by_shot,
        "shots": [
            {"index": index, "start": start, "end": end}
            for index, (start, end) in enumerate(shot_bounds)
        ],
    }
    identity_registry = build_identity_registry(scene_report, video_path, cancellation_check)
    registry_path = work_directory / "entity_registry.json"
    write_identity_registry(identity_registry, registry_path)
    write_json_file(work_directory / "camera_motion_report.json", calibrated_report["camera_motion_report"])
    plan_paths = _write_gap_plans(
        video_path, calibrated_report, scoped_reports, shot_bounds, identity_registry,
        hidden_ranges, work_directory, maximum_entities, render_configuration,
        cancellation_check,
    )
    return PreparedSceneAssets(calibrated_report, plan_paths, registry_path)


def _shot_bounds(
    shots: tuple[tuple[int, int], ...], frame_count: int,
) -> list[tuple[int, int]]:
    return [(int(start), int(end)) for start, end in shots] or [(0, frame_count - 1)]


def _calibrate_each_shot(
    video_path: Path,
    scene_report: dict,
    shot_bounds: list[tuple[int, int]],
    cancellation_check: CancellationCheck | None,
) -> tuple[dict[int, dict], dict[int, dict]]:
    scoped_reports: dict[int, dict] = {}
    motion_by_shot: dict[int, dict] = {}
    for shot_index, bounds in enumerate(shot_bounds):
        raise_if_cancelled(cancellation_check)
        scoped = scene_report_for_shot(scene_report, shot_index, bounds)
        motion = estimate_camera_motion(video_path, scoped, cancellation_check)
        motion_by_shot[shot_index] = motion
        scoped_reports[shot_index] = {**scoped, "camera_motion_report": motion}
    return scoped_reports, motion_by_shot


def _shot_for_gap(
    shot_bounds: list[tuple[int, int]], hidden_range: tuple[int, int],
) -> int:
    for index, (start, end) in enumerate(shot_bounds):
        if start <= hidden_range[0] and hidden_range[1] <= end:
            return index
    for index, (start, end) in enumerate(shot_bounds):
        if start <= hidden_range[0] <= end:
            return index
    return 0


def _write_gap_plans(
    video_path: Path,
    scene_report: dict,
    scoped_reports: dict[int, dict],
    shot_bounds: list[tuple[int, int]],
    identity_registry: dict,
    hidden_ranges: list[list[int]],
    work_directory: Path,
    maximum_entities: int,
    render_configuration: dict,
    cancellation_check: CancellationCheck | None,
) -> list[Path]:
    plan_paths: list[Path] = []
    for gap_index, hidden_range_items in enumerate(hidden_ranges):
        raise_if_cancelled(cancellation_check)
        hidden_range = (int(hidden_range_items[0]), int(hidden_range_items[1]))
        shot_index = _shot_for_gap(shot_bounds, hidden_range)
        planning_report = scoped_reports.get(shot_index, scene_report)
        gap_directory = work_directory / "gaps" / f"gap_{gap_index:02d}" / "scene"
        context_path = export_forensic_context_frame(
            video_path, hidden_range[0] - 1, scene_report,
            gap_directory / "visible_boundary_context.jpg", cancellation_check,
        )
        post_context_path = export_forensic_context_frame(
            video_path,
            min(hidden_range[1] + 1, int(scene_report["video"]["frames"]) - 1),
            scene_report, gap_directory / "visible_boundary_context_after.jpg",
            cancellation_check,
        )
        plan = build_reconstruction_plan_v2(
            planning_report, identity_registry, hidden_range, gap_index,
            maximum_entities=maximum_entities, context_frame_path=context_path,
            post_context_frame_path=post_context_path,
            render_configuration=render_configuration,
        )
        plan["shot_index"] = shot_index
        plan_path = gap_directory / "scene_plan.json"
        write_reconstruction_plan_v2(plan, plan_path)
        plan_paths.append(plan_path)
    _write_plan_manifest(plan_paths, work_directory / "reconstruction_plans.json")
    return plan_paths


def _write_plan_manifest(plan_paths: list[Path], output_path: Path) -> None:
    manifest = []
    for plan_path in plan_paths:
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        manifest.append({
            "gap_index": plan["gap_index"],
            "hidden_range": plan["hidden_range"],
            "rendered_entities": plan["selection_report"]["rendered_count"],
            "overall_confidence": plan["overall_confidence"],
            "calibration_confidence": plan["camera"]["calibration_confidence"],
            "plan_path": str(plan_path),
        })
    write_json_file(output_path, manifest)
