import json
from dataclasses import dataclass
from pathlib import Path

from domain.evidence_contract import validate_visible_evidence_only
from domain.identity_registry import build_identity_registry, write_identity_registry
from domain.reconstruction_plan_v2 import build_reconstruction_plan_v2, write_reconstruction_plan_v2
from infrastructure.blender_runner import BlenderRenderRequest, BlenderRenderResult, render_with_blender
from infrastructure.camera_motion_estimator import estimate_camera_motion
from infrastructure.video_frames import export_video_frame


@dataclass(frozen=True)
class BlenderArtifactPaths:
    output_directory: Path
    identity_registry: Path
    reconstruction_plan: Path
    rendered_media: Path
    render_report: Path
    blend_scene: Path
    blender_log: Path


def prepare_and_render_gap(
    project_root: Path,
    video_path: Path,
    scene_report_path: Path,
    gap_index: int,
    output_directory: Path,
    mode: str = "preview",
) -> BlenderRenderResult:
    scene_report = _load_json(scene_report_path)
    validate_visible_evidence_only(scene_report)
    hidden_range = _selected_hidden_range(scene_report, gap_index)
    artifact_paths = _artifact_paths(output_directory, gap_index, mode)
    context_frame_path = export_video_frame(
        video_path,
        hidden_range[0] - 1,
        artifact_paths.output_directory / "visible_boundary_context.jpg",
    )
    camera_motion_report = estimate_camera_motion(video_path, scene_report)
    calibrated_scene_report = {**scene_report, "camera_motion_report": camera_motion_report}
    registry = build_identity_registry(scene_report, video_path)
    plan = build_reconstruction_plan_v2(
        calibrated_scene_report,
        registry,
        hidden_range,
        gap_index,
        context_frame_path=context_frame_path,
    )
    write_identity_registry(registry, artifact_paths.identity_registry)
    write_reconstruction_plan_v2(plan, artifact_paths.reconstruction_plan)
    request = BlenderRenderRequest(
        plan_path=artifact_paths.reconstruction_plan,
        output_path=artifact_paths.rendered_media,
        report_path=artifact_paths.render_report,
        blend_path=artifact_paths.blend_scene,
        log_path=artifact_paths.blender_log,
        mode=mode,
    )
    return render_with_blender(project_root, request)


def _load_json(file_path: Path) -> dict:
    if not file_path.is_file():
        raise FileNotFoundError(f"Required evidence file is missing: {file_path}")
    with file_path.open("r", encoding="utf-8") as input_file:
        payload = json.load(input_file)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {file_path.name}")
    return payload


def _selected_hidden_range(scene_report: dict, gap_index: int) -> tuple[int, int]:
    hidden_ranges = scene_report.get("hidden_ranges", [])
    if gap_index < 0 or gap_index >= len(hidden_ranges):
        raise ValueError(f"Gap index {gap_index} is outside the available range")
    selected_range = hidden_ranges[gap_index]
    return int(selected_range["start"]), int(selected_range["end"])


def _artifact_paths(output_directory: Path, gap_index: int, mode: str) -> BlenderArtifactPaths:
    gap_directory = output_directory / f"gap_{gap_index:02d}"
    extension = ".png" if mode == "preview" else ".mp4"
    media_name = "midpoint_preview" if mode == "preview" else "gap_animation"
    return BlenderArtifactPaths(
        output_directory=gap_directory,
        identity_registry=output_directory / "entity_registry.json",
        reconstruction_plan=gap_directory / "reconstruction_plan_v2.json",
        rendered_media=gap_directory / f"{media_name}{extension}",
        render_report=gap_directory / f"{mode}_render_report.json",
        blend_scene=gap_directory / "forensic_scene.blend",
        blender_log=gap_directory / f"blender_{mode}.log",
    )
