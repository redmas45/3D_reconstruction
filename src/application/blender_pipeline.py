import json
from dataclasses import dataclass
from pathlib import Path

from domain.evidence_contract import validate_visible_evidence_only
from domain.identity_registry import build_identity_registry, write_identity_registry
from domain.reconstruction_plan_v2 import build_reconstruction_plan_v2, write_reconstruction_plan_v2
from infrastructure.blender_runner import BlenderRenderRequest, render_with_blender
from infrastructure.camera_motion_estimator import estimate_camera_motion
from infrastructure.video_frames import export_video_frame


@dataclass(frozen=True)
class PreparedBlenderAssets:
    scene_report: dict
    plan_paths: list[Path]
    identity_registry_path: Path


def prepare_blender_assets(
    video_path: Path,
    scene_report: dict,
    hidden_ranges: list[list[int]],
    work_directory: Path,
    maximum_entities: int,
    render_configuration: dict,
) -> PreparedBlenderAssets:
    validate_visible_evidence_only(scene_report)
    motion_report = estimate_camera_motion(video_path, scene_report)
    calibrated_report = {**scene_report, "camera_motion_report": motion_report}
    identity_registry = build_identity_registry(scene_report, video_path)
    registry_path = work_directory / "entity_registry.json"
    write_identity_registry(identity_registry, registry_path)
    _write_json(work_directory / "camera_motion_report.json", motion_report)
    plan_paths = _write_gap_plans(
        video_path,
        calibrated_report,
        identity_registry,
        hidden_ranges,
        work_directory,
        maximum_entities,
        render_configuration,
    )
    return PreparedBlenderAssets(calibrated_report, plan_paths, registry_path)


def render_blender_gap(
    project_root: Path,
    plan_path: Path,
    gap_directory: Path,
    reuse_render: bool,
) -> Path:
    blender_directory = gap_directory / "blender"
    output_path = blender_directory / "gap_blender.mp4"
    report_path = blender_directory / "render_report.json"
    blend_path = blender_directory / "scene.blend"
    log_path = blender_directory / "blender.log"
    if reuse_render and _render_cache_is_complete(output_path, report_path, blend_path):
        return output_path
    request = BlenderRenderRequest(plan_path, output_path, report_path, blend_path, log_path, "animation")
    render_with_blender(project_root, request)
    return output_path


def _write_gap_plans(
    video_path: Path,
    scene_report: dict,
    identity_registry: dict,
    hidden_ranges: list[list[int]],
    work_directory: Path,
    maximum_entities: int,
    render_configuration: dict,
) -> list[Path]:
    plan_paths: list[Path] = []
    for gap_index, hidden_range_items in enumerate(hidden_ranges):
        hidden_range = (int(hidden_range_items[0]), int(hidden_range_items[1]))
        gap_directory = work_directory / "gaps" / f"gap_{gap_index:02d}" / "blender"
        context_path = export_video_frame(
            video_path, hidden_range[0] - 1, gap_directory / "visible_boundary_context.jpg"
        )
        plan = build_reconstruction_plan_v2(
            scene_report,
            identity_registry,
            hidden_range,
            gap_index,
            maximum_entities=maximum_entities,
            context_frame_path=context_path,
            render_configuration=render_configuration,
        )
        plan_path = gap_directory / "plan_v2.json"
        write_reconstruction_plan_v2(plan, plan_path)
        plan_paths.append(plan_path)
    _write_plan_manifest(plan_paths, work_directory / "reconstruction_plans_v2.json")
    return plan_paths


def _write_plan_manifest(plan_paths: list[Path], output_path: Path) -> None:
    manifest: list[dict] = []
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
    _write_json(output_path, manifest)


def _write_json(output_path: Path, payload: object) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2)


def _render_cache_is_complete(output_path: Path, report_path: Path, blend_path: Path) -> bool:
    return all(path.is_file() and path.stat().st_size > 0 for path in (output_path, report_path, blend_path))
