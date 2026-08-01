"""Assembles and writes the browser scene manifest from on-disk reconstruction artifacts.

This is the terminal backend step of the browser-first path. It reads the validated
reconstruction plans and the public reasoning/hypothesis artifacts that earlier stages
produced from visible evidence only, hands them to the pure manifest builder, and
writes the single JSON file the Three.js renderer consumes. It never reads the source
video, so no hidden frame can reach the browser through this path.
"""

from pathlib import Path

from domain.three_scene_manifest import build_three_scene_manifest
from infrastructure.json_files import read_json_file, write_json_file


SCENE_MANIFEST_FILENAME = "scene_manifest.json"
PUBLIC_REASONING_FILENAME = "reasoning_public.json"
HYPOTHESES_RELATIVE_PATH = Path("reasoning") / "gap_hypotheses_v2.json"


def build_and_write_scene_manifest(
    video_info: dict,
    gap_selection: dict,
    scene_report: dict,
    plan_paths: list[Path],
    work_directory: Path,
) -> Path:
    """Build the scene manifest and write it beside the other work artifacts.

    Returns the manifest path so the caller can surface it as the job's deliverable.
    """
    plans = _read_plans(plan_paths)
    reasoning = _read_dict(work_directory / PUBLIC_REASONING_FILENAME)
    hypotheses = _read_dict(work_directory / HYPOTHESES_RELATIVE_PATH)
    manifest = build_three_scene_manifest(
        video_info, gap_selection, scene_report, plans, reasoning, hypotheses,
    )
    manifest_path = work_directory / SCENE_MANIFEST_FILENAME
    write_json_file(manifest_path, manifest)
    return manifest_path


def _read_plans(plan_paths: list[Path]) -> list[dict]:
    plans = []
    for plan_path in plan_paths:
        payload = read_json_file(plan_path)
        if isinstance(payload, dict) and isinstance(payload.get("gap_index"), int):
            plans.append(payload)
    return sorted(plans, key=lambda plan: int(plan["gap_index"]))


def _read_dict(path: Path) -> dict:
    if not path.is_file():
        return {}
    payload = read_json_file(path)
    return payload if isinstance(payload, dict) else {}
