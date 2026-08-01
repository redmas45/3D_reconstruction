import copy
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from domain.three_scene_manifest import build_three_scene_manifest
from domain.three_scene_validation import (
    SceneManifestValidationError,
    validate_three_scene_manifest,
)


def _camera(motion_model: str = "static_camera", confidence: float = 0.8) -> dict:
    return {
        "projection_model": "pinhole_ground_plane_v2",
        "position": [0.0, 0.0, 1.6],
        "look_at": [0.0, 10.0, 1.6],
        "field_of_view_degrees": 58.0,
        "horizon_normalized_y": 0.5,
        "focal_length_mm": 35.0,
        "motion_model": motion_model,
        "presentation_mode": (
            "source_camera_aligned" if motion_model == "static_camera" else "stabilized_forensic_view"
        ),
        "calibration_confidence": confidence,
    }


def _person_entity(track_id: str = "7", confidence: float = 0.82, fidelity: str = "supported") -> dict:
    return {
        "id": track_id,
        "kind": "person",
        "confidence": confidence,
        "fidelity_tier": fidelity,
        "lifecycle": "continuous",
        "appearance": {
            "upper_color": [0.2, 0.5, 0.6],
            "lower_color": [0.1, 0.1, 0.2],
            "vehicle_color": [0.2, 0.5, 0.6],
            "source": "visible_evidence",
        },
        "body_proportions": {"height_scale": 1.02, "shoulder_scale": 0.98, "limb_scale": 1.0},
        "animation": {"state": "walk", "speed_meters_per_second": 1.3, "phase_offset": 0.4},
        "motion_profile": {
            "schema_version": 1,
            "source": "yolo_pose_visible_boundaries",
            "clip": "walk",
            "phase_offset": 0.4,
            "cadence_scale": 1.08,
            "blend_seconds": 0.18,
            "pose_confidence": 0.6,
        },
        "kinematics": {
            "model": "ground_plane_kinematic",
            "duration_seconds": 5.0,
            "maximum_speed_meters_per_second": 3.0,
            "maximum_acceleration_meters_per_second_squared": 3.0,
            "maximum_turn_rate_degrees_per_second": 120.0,
            "ground_contact_required": True,
        },
        "uncertainty": {"position_radius_meters": 0.4, "alternative_paths": 0},
        "boundary_evidence": {"heading_disagreement_degrees": 12.0},
        "path_prediction": {
            "method": "centripetal_catmull_rom",
            "waypoints": [
                {"role": "start", "frame": 299, "world": [1.0, 4.0, 0.0]},
                {"role": "inferred_01", "frame": 375, "world": [1.2, 6.0, 0.0]},
                {"role": "predicted_end", "frame": 451, "world": [1.4, 8.0, 0.0]},
            ],
        },
    }


def _vehicle_entity(track_id: str = "12") -> dict:
    entity = _person_entity(track_id, confidence=0.7, fidelity="plausible")
    entity["kind"] = "car"
    entity.pop("motion_profile")
    entity["animation"] = {"state": "walk", "speed_meters_per_second": 9.0, "phase_offset": 0.0}
    return entity


def _plan(entities: list[dict], camera: dict | None = None) -> dict:
    return {
        "gap_index": 0,
        "overall_confidence": 0.7,
        "camera": camera or _camera(),
        "environment": {
            "style": "forensic_3d",
            "ground_color": [0.03, 0.04, 0.06],
            "grid_color": [0.04, 0.6, 0.68],
            "backplate_frame": 299,
            "hybrid_backplate_enabled": True,
            "hybrid_backplate_reason": "static_camera_visible_evidence",
        },
        "entities": entities,
    }


def _reasoning() -> dict:
    return {
        "headline": "A pedestrian continues north",
        "whole_video_summary": "One tracked pedestrian.",
        "confidence": 0.7,
        "causal_link_supported": False,
        "mode": "azure",
        "unknowns": ["exact pace"],
        "story_points": [{"statement": "Pedestrian tracked before and after."}],
        "clues": [
            {"id": "scene_tracks", "category": "entity_inventory", "statement": "One person tracked", "confidence": 0.9, "scope": "scene"},
        ],
        "gap_summaries": [
            {"gap_index": 0, "before_observed": "Walking north", "inside_inferred": "Continues north",
             "after_observed": "Reappears north", "confidence": 0.72, "unknowns": ["pace"]},
        ],
        "decisions": [
            {"gap_index": 0, "gap_summary": "Continues north", "confidence": 0.72,
             "evidence_references": ["track:7:pre_boundary"], "clue_ids": ["scene_tracks"], "unknowns": ["pace"],
             "entities": [
                 {"entity_id": "7", "selected_hypothesis_id": "gap_00_7_continue_measured_motion",
                  "decision_summary": "Measured motion continues", "confidence": 0.72,
                  "rejected_hypotheses": [{"id": "gap_00_7_hold_position", "reason": "Was clearly moving"}]},
             ],
             "event_beats": [{"time_fraction": 0.5, "action": "continue", "entity_ids": ["7"]}]},
        ],
    }


def _hypotheses() -> dict:
    return {
        "schema_version": 2,
        "gaps": [
            {"gap_index": 0, "entities": [
                {"entity_id": "7", "kind": "person", "hypotheses": [
                    {"id": "gap_00_7_continue_measured_motion", "type": "continue_measured_motion",
                     "action": "walk", "visibility": "visible_throughout",
                     "path": [
                         {"role": "start", "frame": 299, "world": [1.0, 4.0, 0.0]},
                         {"role": "inferred_01", "frame": 375, "world": [1.2, 6.0, 0.0]},
                         {"role": "predicted_end", "frame": 451, "world": [1.4, 8.0, 0.0]},
                     ], "speed_meters_per_second": 1.3},
                ]},
            ]},
        ],
    }


def _build(plan: dict | None = None, reasoning: dict | None = None, hypotheses: dict | None = None,
           video_info: dict | None = None, gap_selection: dict | None = None) -> dict:
    return build_three_scene_manifest(
        video_info or {"width": 1920, "height": 1080, "fps": 30.0, "frames": 900, "name": "clip.mp4"},
        gap_selection or {"missing_fraction_actual": 0.25, "hidden_ranges": [[300, 450]]},
        {"video": {"name": "clip.mp4"}},
        [plan or _plan([_person_entity()])],
        reasoning or _reasoning(),
        hypotheses or _hypotheses(),
    )


class SceneManifestBuildTests(unittest.TestCase):
    def test_manifest_is_threejs_and_self_validates(self) -> None:
        manifest = _build()
        self.assertEqual(manifest["renderer"], "threejs")
        self.assertEqual(manifest["schema_version"], 1)
        validate_three_scene_manifest(manifest)

    def test_source_metrics_are_derived_not_hardcoded(self) -> None:
        manifest = _build(video_info={"width": 1280, "height": 720, "fps": 25.0, "frames": 500, "name": "other.mp4"})
        self.assertEqual(manifest["source"]["width"], 1280)
        self.assertEqual(manifest["source"]["height"], 720)
        self.assertEqual(manifest["source"]["duration_seconds"], 20.0)
        self.assertEqual(manifest["source"]["observed_fraction"], 0.75)

    def test_entity_carries_waypoints_appearance_and_decision(self) -> None:
        entity = _build()["gaps"][0]["entities"][0]
        self.assertGreaterEqual(len(entity["waypoints"]), 3)
        self.assertEqual(entity["appearance"]["source"], "visible_evidence")
        self.assertEqual(entity["selected_hypothesis"]["type"], "continue_measured_motion")
        self.assertEqual(entity["rejected_hypotheses"][0]["id"], "gap_00_7_hold_position")

    def test_visible_anchor_is_carried_for_projected_scale_matching(self) -> None:
        plan = _plan([_person_entity()])
        plan["entities"] = [{**_person_entity(), "visual_anchor": {
            "center_x_fraction": 0.42,
            "ground_y_fraction": 0.81,
            "width_fraction": 0.08,
            "height_fraction": 0.31,
            "source_frame": 299,
        }}]
        entity = _build(plan=plan)["gaps"][0]["entities"][0]
        self.assertEqual(entity["visual_anchor"]["height_fraction"], 0.31)
        self.assertEqual(entity["visual_anchor"]["source_frame"], 299)

    def test_waypoint_headings_face_travel_direction(self) -> None:
        entity = _build()["gaps"][0]["entities"][0]
        # Path advances mostly along +Y (forward) with slight +X, so heading is a small
        # positive angle east of north, identical along a straight segment.
        headings = [waypoint["heading_degrees"] for waypoint in entity["waypoints"]]
        self.assertTrue(all(0.0 < heading < 45.0 for heading in headings))

    def test_time_fractions_span_zero_to_one(self) -> None:
        waypoints = _build()["gaps"][0]["entities"][0]["waypoints"]
        self.assertEqual(waypoints[0]["time_fraction"], 0.0)
        self.assertEqual(waypoints[-1]["time_fraction"], 1.0)


class DeterministicSeedingTests(unittest.TestCase):
    def test_same_track_id_seeds_identically_across_gaps(self) -> None:
        gap_selection = {"missing_fraction_actual": 0.25, "hidden_ranges": [[300, 450], [600, 750]]}
        second_plan = _plan([_person_entity("7")])
        second_plan["gap_index"] = 1
        manifest = build_three_scene_manifest(
            {"width": 1920, "height": 1080, "fps": 30.0, "frames": 900, "name": "clip.mp4"},
            gap_selection, {"video": {"name": "clip.mp4"}},
            [_plan([_person_entity("7")]), second_plan],
            _reasoning(), _hypotheses(),
        )
        first = manifest["gaps"][0]["entities"][0]
        second = manifest["gaps"][1]["entities"][0]
        self.assertEqual(first["appearance_seed"], second["appearance_seed"])
        self.assertEqual(first["appearance"], second["appearance"])
        self.assertEqual(first["body_proportions"], second["body_proportions"])

    def test_different_track_ids_seed_differently(self) -> None:
        manifest = _build(plan=_plan([_person_entity("7"), _person_entity("8")]))
        seeds = {entity["appearance_seed"] for entity in manifest["gaps"][0]["entities"]}
        self.assertEqual(len(seeds), 2)


class AntiOverfittingScenarioTests(unittest.TestCase):
    def test_multiple_people(self) -> None:
        manifest = _build(plan=_plan([_person_entity("1"), _person_entity("2"), _person_entity("3")]))
        self.assertEqual(len(manifest["gaps"][0]["entities"]), 3)
        validate_three_scene_manifest(manifest)

    def test_vehicle_entity_is_categorized_and_grounded(self) -> None:
        manifest = _build(plan=_plan([_vehicle_entity("12")]))
        entity = manifest["gaps"][0]["entities"][0]
        self.assertEqual(entity["category"], "vehicle")
        self.assertEqual(entity["proxy"]["type"], "vehicle")
        self.assertEqual(len(entity["proxy"]["dimensions_metres"]), 3)

    def test_gap_with_no_entities_still_valid(self) -> None:
        manifest = _build(plan=_plan([]))
        self.assertEqual(manifest["gaps"][0]["entities"], [])
        validate_three_scene_manifest(manifest)

    def test_moving_camera_raises_warning_and_lowers_presentation_mode(self) -> None:
        plan = _plan([_person_entity()], camera=_camera(motion_model="dynamic_or_unclassified", confidence=0.42))
        manifest = _build(plan=plan)
        camera = manifest["gaps"][0]["camera"]
        self.assertNotEqual(camera["motion_warning"], "")
        self.assertEqual(camera["presentation_mode"], "stabilized_forensic_view")

    def test_short_and_long_videos_scale_frame_counts(self) -> None:
        short = _build(video_info={"width": 640, "height": 480, "fps": 30.0, "frames": 120, "name": "s.mp4"},
                       gap_selection={"missing_fraction_actual": 0.25, "hidden_ranges": [[40, 70]]})
        self.assertEqual(short["source"]["frame_count"], 120)
        validate_three_scene_manifest(short)


class ConfidenceToFidelityTests(unittest.TestCase):
    def test_fidelity_tier_is_carried_through(self) -> None:
        manifest = _build(plan=_plan([_person_entity("7", confidence=0.3, fidelity="weak")]))
        self.assertEqual(manifest["gaps"][0]["entities"][0]["visual_fidelity_tier"], "weak")

    def test_unknown_fidelity_tier_collapses_to_weak(self) -> None:
        entity = _person_entity("7")
        entity["fidelity_tier"] = "nonsense"
        manifest = _build(plan=_plan([entity]))
        self.assertEqual(manifest["gaps"][0]["entities"][0]["visual_fidelity_tier"], "weak")

    def test_heading_disagreement_is_preserved_for_the_trace(self) -> None:
        entity = _person_entity("7")
        entity["boundary_evidence"]["heading_disagreement_degrees"] = 118.0
        manifest = _build(plan=_plan([entity]))
        self.assertEqual(
            manifest["gaps"][0]["entities"][0]["uncertainty"]["heading_disagreement_degrees"], 118.0,
        )


class HypothesisSelectionTests(unittest.TestCase):
    def test_held_position_hypothesis_supplies_collapsed_path(self) -> None:
        hypotheses = _hypotheses()
        held_path = [
            {"role": "start", "frame": 299, "world": [1.0, 4.0, 0.0]},
            {"role": "inferred_01", "frame": 375, "world": [1.0, 4.0, 0.0]},
            {"role": "predicted_end", "frame": 451, "world": [1.0, 4.0, 0.0]},
        ]
        hypotheses["gaps"][0]["entities"][0]["hypotheses"].append({
            "id": "gap_00_7_hold_position", "type": "hold_position", "action": "idle",
            "visibility": "visible_throughout", "path": held_path, "speed_meters_per_second": 0.0,
        })
        reasoning = _reasoning()
        reasoning["decisions"][0]["entities"][0]["selected_hypothesis_id"] = "gap_00_7_hold_position"
        manifest = _build(reasoning=reasoning, hypotheses=hypotheses)
        entity = manifest["gaps"][0]["entities"][0]
        self.assertEqual(entity["selected_hypothesis"]["type"], "hold_position")
        worlds = {tuple(waypoint["world"]) for waypoint in entity["waypoints"]}
        self.assertEqual(len(worlds), 1)

    def test_missing_hypothesis_falls_back_to_prediction_waypoints(self) -> None:
        reasoning = _reasoning()
        reasoning["decisions"][0]["entities"][0]["selected_hypothesis_id"] = "unknown_id"
        manifest = _build(reasoning=reasoning, hypotheses={"schema_version": 2, "gaps": []})
        self.assertGreaterEqual(len(manifest["gaps"][0]["entities"][0]["waypoints"]), 3)


class HiddenTruthProtectionTests(unittest.TestCase):
    def test_backplate_inside_gap_is_rejected(self) -> None:
        manifest = _build()
        manifest["gaps"][0]["environment"]["backplate_frame"] = 400
        with self.assertRaises(SceneManifestValidationError):
            validate_three_scene_manifest(manifest)

    def test_forbidden_hidden_truth_key_is_rejected(self) -> None:
        manifest = _build()
        manifest["gaps"][0]["entities"][0]["ground_truth"] = [1.0, 2.0, 0.0]
        with self.assertRaises(SceneManifestValidationError):
            validate_three_scene_manifest(manifest)

    def test_fewer_than_three_waypoints_is_rejected(self) -> None:
        manifest = _build()
        manifest["gaps"][0]["entities"][0]["waypoints"] = manifest["gaps"][0]["entities"][0]["waypoints"][:2]
        with self.assertRaises(SceneManifestValidationError):
            validate_three_scene_manifest(manifest)

    def test_out_of_range_confidence_is_rejected(self) -> None:
        manifest = _build()
        manifest["gaps"][0]["entities"][0]["confidence"] = 1.4
        with self.assertRaises(SceneManifestValidationError):
            validate_three_scene_manifest(manifest)

    def test_deep_copy_round_trip_still_validates(self) -> None:
        validate_three_scene_manifest(copy.deepcopy(_build()))


if __name__ == "__main__":
    unittest.main()
