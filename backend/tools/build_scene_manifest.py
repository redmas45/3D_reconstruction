"""Build a Three.js scene manifest, from a completed work directory or a demo fixture.

Two modes:

  python backend/tools/build_scene_manifest.py --work <work_dir> --out scene_manifest.json
      Reassemble the manifest for a finished run from its on-disk artifacts.

  python backend/tools/build_scene_manifest.py --demo --out frontend/assets/fixtures/sample-scene.json
      Emit a representative multi-entity manifest for exercising the browser renderer.
      The demo geometry is synthetic and self-contained; it is NOT taken from any real
      test video, so it cannot leak or overfit to one.
"""

import argparse
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from domain.three_scene_manifest import build_three_scene_manifest  # noqa: E402
from infrastructure.json_files import read_json_file, write_json_file  # noqa: E402


def _curved_waypoints(start_x, start_y, curl, frames):
    points = []
    count = 4
    for index in range(count):
        fraction = index / (count - 1)
        forward = start_y + fraction * 9.0
        lateral = start_x + math.sin(fraction * math.pi) * curl
        role = "start" if index == 0 else "predicted_end" if index == count - 1 else f"inferred_{index:02d}"
        points.append({
            "role": role,
            "frame": int(frames[0] + fraction * (frames[1] - frames[0])),
            "world": [round(lateral, 3), round(forward, 3), 0.0],
        })
    return points


def _person(track_id, start_x, curl, confidence, tier, upper, lower, speed):
    frames = (299, 451)
    return {
        "id": track_id, "kind": "person", "confidence": confidence, "fidelity_tier": tier,
        "lifecycle": "continuous",
        "appearance": {"upper_color": upper, "lower_color": lower, "vehicle_color": upper, "source": "visible_evidence"},
        "body_proportions": {
            "height_scale": round(0.94 + (hash(track_id) % 12) / 100, 4),
            "shoulder_scale": round(0.9 + (hash(track_id) % 8) / 40, 4),
            "limb_scale": 1.0,
        },
        "animation": {"state": "walk", "speed_meters_per_second": speed, "phase_offset": (hash(track_id) % 100) / 100},
        "motion_profile": {
            "schema_version": 1, "source": "yolo_pose_visible_boundaries", "clip": "walk",
            "phase_offset": (hash(track_id) % 100) / 100, "cadence_scale": 1.05,
            "blend_seconds": 0.18, "pose_confidence": 0.6,
        },
        "kinematics": {
            "model": "ground_plane_kinematic", "duration_seconds": 5.0,
            "maximum_speed_meters_per_second": 3.0, "maximum_acceleration_meters_per_second_squared": 3.0,
            "maximum_turn_rate_degrees_per_second": 120.0, "ground_contact_required": True,
        },
        "uncertainty": {"position_radius_meters": round(0.3 + (1 - confidence) * 1.4, 3), "alternative_paths": 0 if confidence >= 0.75 else 2},
        "boundary_evidence": {"heading_disagreement_degrees": 8.0 if confidence > 0.7 else 74.0},
        "path_prediction": {"method": "centripetal_catmull_rom", "waypoints": _curved_waypoints(start_x, 3.0, curl, frames)},
    }


def _vehicle(track_id, start_x, curl, confidence):
    frames = (299, 451)
    entity = _person(track_id, start_x, curl, confidence, "supported", [0.5, 0.1, 0.12], [0.1, 0.1, 0.1], 8.0)
    entity["kind"] = "car"
    entity.pop("motion_profile")
    entity["animation"] = {"state": "walk", "speed_meters_per_second": 8.0, "phase_offset": 0.0}
    entity["kinematics"]["maximum_speed_meters_per_second"] = 35.0
    entity["path_prediction"]["waypoints"] = _curved_waypoints(start_x, 2.5, curl, frames)
    entity["appearance"]["vehicle_color"] = [0.6, 0.15, 0.15]
    return entity


def _demo_manifest():
    entities = [
        _person("7", -2.5, 1.2, 0.86, "supported", [0.20, 0.52, 0.62], [0.12, 0.14, 0.20], 1.4),
        _person("11", 0.4, -0.8, 0.63, "plausible", [0.70, 0.42, 0.20], [0.15, 0.16, 0.18], 1.1),
        _person("14", 2.8, 2.0, 0.38, "weak", [0.45, 0.45, 0.5], [0.2, 0.2, 0.22], 0.9),
        _vehicle("22", -1.0, 0.6, 0.79),
    ]
    camera = {
        "projection_model": "pinhole_ground_plane_v2", "position": [0.0, -3.0, 1.6],
        "look_at": [0.0, 8.0, 1.0], "field_of_view_degrees": 58.0, "horizon_normalized_y": 0.46,
        "focal_length_mm": 33.0, "motion_model": "static_camera",
        "presentation_mode": "source_camera_aligned", "calibration_confidence": 0.8,
    }
    plan = {
        "gap_index": 0, "overall_confidence": 0.72, "camera": camera,
        "environment": {
            "style": "forensic_3d", "ground_color": [0.05, 0.06, 0.09], "grid_color": [0.05, 0.62, 0.68],
            "backplate_frame": 298, "hybrid_backplate_enabled": True,
            "hybrid_backplate_reason": "static_camera_visible_evidence",
        },
        "entities": entities,
    }
    reasoning = {
        "headline": "Four tracked subjects continue across the crossing",
        "whole_video_summary": "Three pedestrians and one car were tracked on both sides of the interval and their motion was continued from the last observed heading.",
        "confidence": 0.72, "causal_link_supported": False, "mode": "azure",
        "unknowns": ["Exact pace inside the interval", "Whether the car yielded"],
        "story_points": [{"statement": "Every subject was seen entering and leaving the gap."}],
        "clues": [
            {"id": "scene_tracks", "category": "entity_inventory", "statement": "Four subjects tracked continuously", "confidence": 0.9, "scope": "scene"},
            {"id": "scene_camera", "category": "camera", "statement": "Camera is static and well calibrated", "confidence": 0.82, "scope": "scene"},
        ],
        "gap_summaries": [{
            "gap_index": 0, "before_observed": "Subjects moving north across the crossing.",
            "inside_inferred": "Motion continued along measured headings.",
            "after_observed": "Subjects reappear consistent with the inferred paths.",
            "confidence": 0.72, "unknowns": ["Exact pace"],
        }],
        "decisions": [{
            "gap_index": 0, "gap_summary": "Continue measured motion for all subjects.", "confidence": 0.72,
            "evidence_references": ["track:7:pre_boundary", "scene:camera_motion_report"],
            "clue_ids": ["scene_tracks", "scene_camera"], "unknowns": ["Exact pace"],
            "entities": [
                {"entity_id": entity["id"], "selected_hypothesis_id": f"gap_00_{entity['id']}_continue_measured_motion",
                 "decision_summary": "Measured motion is the best-supported continuation.", "confidence": entity["confidence"],
                 "rejected_hypotheses": [{"id": f"gap_00_{entity['id']}_hold_position", "reason": "Subject was clearly moving before the gap."}]}
                for entity in entities
            ],
            "event_beats": [{"time_fraction": 0.5, "action": "continue", "entity_ids": [e["id"] for e in entities]}],
        }],
    }
    hypotheses = {"schema_version": 2, "gaps": [{"gap_index": 0, "entities": [
        {"entity_id": entity["id"], "kind": entity["kind"], "hypotheses": [{
            "id": f"gap_00_{entity['id']}_continue_measured_motion", "type": "continue_measured_motion",
            "action": "drive" if entity["kind"] != "person" else "walk",
            "visibility": "visible_throughout", "path": entity["path_prediction"]["waypoints"],
            "speed_meters_per_second": entity["animation"]["speed_meters_per_second"],
        }]}
        for entity in entities
    ]}]}
    video_info = {"width": 1920, "height": 1080, "fps": 30.0, "frames": 1800, "name": "demo_crossing.mp4"}
    gap_selection = {"missing_fraction_actual": 0.25, "hidden_ranges": [[300, 450]]}
    scene_report = {"video": {"name": "demo_crossing.mp4"}}
    return build_three_scene_manifest(video_info, gap_selection, scene_report, [plan], reasoning, hypotheses)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a Three.js scene manifest")
    parser.add_argument("--demo", action="store_true", help="Emit a synthetic demo manifest")
    parser.add_argument("--work", type=Path, help="A completed run work directory")
    parser.add_argument("--out", type=Path, required=True, help="Output manifest path")
    arguments = parser.parse_args()
    if arguments.demo:
        manifest = _demo_manifest()
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        write_json_file(arguments.out, manifest)
        print(f"Wrote demo scene manifest: {arguments.out}")
        return 0
    if arguments.work is None:
        parser.error("Either --demo or --work is required")
    # A finished run already writes scene_manifest.json into its work directory; this
    # mode just surfaces that file, and fails loudly if the run did not produce one.
    produced = arguments.work / "scene_manifest.json"
    if not produced.is_file():
        print(f"No scene_manifest.json found in {arguments.work}; run the pipeline first.")
        return 1
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    write_json_file(arguments.out, _read_json(produced))
    print(f"Copied scene manifest: {arguments.out}")
    return 0


def _read_json(path: Path) -> dict:
    payload = read_json_file(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Scene manifest is not an object: {path}")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
