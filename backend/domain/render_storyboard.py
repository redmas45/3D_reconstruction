"""Compiles validated decisions into deterministic renderer-only tokens."""

import hashlib
import json


STORYBOARD_SCHEMA_VERSION = 1
SCENE_SHELL_SCHEMA_VERSION = 1
RENDER_BUDGET_SCHEMA_VERSION = 1


def compile_render_storyboard(
    scene_report: dict,
    plans: list[dict],
    hypothesis_library: dict,
    gap_decisions: dict,
    renderer_configuration: dict,
) -> tuple[dict, dict, dict]:
    source = _source_contract(scene_report)
    profile = _render_profile(renderer_configuration)
    hypotheses = _hypothesis_index(hypothesis_library)
    decisions = {int(item["gap_index"]): item for item in gap_decisions["decisions"]}
    gaps = [
        _storyboard_gap(plan, decisions[int(plan["gap_index"])], hypotheses)
        for plan in sorted(plans, key=lambda item: int(item["gap_index"]))
    ]
    storyboard = {
        "schema_version": STORYBOARD_SCHEMA_VERSION,
        "source": source,
        "render_profile": profile,
        "gaps": gaps,
    }
    storyboard["storyboard_digest"] = _digest(storyboard)
    return storyboard, _scene_shell_manifest(source, plans), _render_budget(source, gaps, profile)


def _storyboard_gap(plan: dict, decision: dict, hypotheses: dict) -> dict:
    gap_index = int(plan["gap_index"])
    selected = {item["entity_id"]: item for item in decision["entities"]}
    return {
        "gap_index": gap_index,
        "hidden_range": plan["hidden_range"],
        "duration_seconds": plan["duration_seconds"],
        "confidence": decision["confidence"],
        "event_beats": decision["event_beats"],
        "entities": [
            _storyboard_entity(entity, selected[str(entity["id"])], hypotheses[gap_index])
            for entity in plan.get("entities", [])
        ],
    }


def _storyboard_entity(entity: dict, decision: dict, gap_hypotheses: dict) -> dict:
    hypothesis = gap_hypotheses[str(entity["id"])][decision["selected_hypothesis_id"]]
    return {
        "entity_id": str(entity["id"]),
        "kind": str(entity["kind"]),
        "identity_registry_id": str(entity.get("identity_registry_id", entity["id"])),
        "hypothesis_id": hypothesis["id"],
        "action": hypothesis["action"],
        "visibility": hypothesis["visibility"],
        "path": hypothesis["path"],
        "speed_meters_per_second": hypothesis["speed_meters_per_second"],
        "fidelity_tier": _fidelity_tier(entity, hypothesis),
        "confidence": decision["confidence"],
    }


def _fidelity_tier(entity: dict, hypothesis: dict) -> str:
    if hypothesis["action"] == "proxy" or float(entity["confidence"]) < 0.5:
        return "proxy"
    if float(entity["confidence"]) < 0.75:
        return "transparent_supported"
    return "solid_supported"


def _source_contract(scene_report: dict) -> dict:
    video = scene_report["video"]
    return {
        "sha256": video.get("sha256"),
        "width": int(video["width"]),
        "height": int(video["height"]),
        "fps": float(video["fps"]),
        "frame_count": int(video["frames"]),
    }


def _render_profile(configuration: dict) -> dict:
    target_fps = max(1, int(configuration.get("target_fps", 10)))
    return {
        "name": str(configuration.get("default_profile", "standard_forensic")),
        "target_fps": target_fps,
        "scale_percent": int(configuration.get("scale_percent", 50)),
        "cycles_samples": int(configuration.get("cycles_samples", 4)),
        "maximum_detailed_entities": int(configuration.get("maximum_detailed_entities", 8)),
        "maximum_gpu_workers": int(configuration.get("maximum_gpu_workers", 1)),
        "maximum_cpu_workers": int(configuration.get("maximum_cpu_workers", 2)),
        "checkpoint_frame_batch": int(configuration.get("checkpoint_frame_batch", 24)),
        "diagnostic_pose_count": int(configuration.get("diagnostic_pose_count", 5)),
    }


def _scene_shell_manifest(source: dict, plans: list[dict]) -> dict:
    cameras = [
        {
            "gap_index": int(plan["gap_index"]),
            "calibration_confidence": float(plan["camera"]["calibration_confidence"]),
            "environment": plan["environment"],
        }
        for plan in plans
    ]
    return {
        "schema_version": SCENE_SHELL_SCHEMA_VERSION,
        "source_sha256": source["sha256"],
        "scene_file": "storyboard/scene_shell.blend",
        "reusable_components": [
            "calibrated_camera", "ground_plane", "proxy_occluders",
            "lighting", "world_materials",
        ],
        "gap_cameras": cameras,
    }


def _render_budget(source: dict, gaps: list[dict], profile: dict) -> dict:
    gap_budgets = []
    for gap in gaps:
        target_frames = max(2, round(float(gap["duration_seconds"]) * profile["target_fps"]))
        detailed_count = sum(item["fidelity_tier"] != "proxy" for item in gap["entities"])
        gap_budgets.append({
            "gap_index": gap["gap_index"],
            "target_frames": target_frames,
            "diagnostic_pose_frames": _pose_frames(target_frames, profile["diagnostic_pose_count"]),
            "detailed_entities": min(detailed_count, profile["maximum_detailed_entities"]),
            "proxy_entities": max(0, len(gap["entities"]) - profile["maximum_detailed_entities"]),
            "estimated_pixel_frames": _pixel_frame_budget(source, target_frames, profile),
        })
    return {
        "schema_version": RENDER_BUDGET_SCHEMA_VERSION,
        "profile": profile["name"],
        "requires_preview_approval": True,
        "gaps": gap_budgets,
    }


def _pose_frames(frame_count: int, pose_count: int) -> list[int]:
    bounded_count = min(max(1, pose_count), frame_count)
    if bounded_count == 1:
        return [0]
    return sorted({
        round(position * (frame_count - 1) / (bounded_count - 1))
        for position in range(bounded_count)
    })


def _pixel_frame_budget(source: dict, frame_count: int, profile: dict) -> int:
    scale = profile["scale_percent"] / 100.0
    return round(source["width"] * scale * source["height"] * scale * frame_count)


def _hypothesis_index(library: dict) -> dict:
    return {
        int(gap["gap_index"]): {
            entity["entity_id"]: {item["id"]: item for item in entity["hypotheses"]}
            for entity in gap["entities"]
        }
        for gap in library["gaps"]
    }


def _digest(value: dict) -> str:
    serialized = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()
