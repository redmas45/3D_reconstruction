import copy
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from domain.reconstruction_plan_v2 import (
    PlanValidationError,
    _relevance_score,
    build_reconstruction_plan_v2,
    validate_reconstruction_plan_v2,
)


HIDDEN_RANGE = (10, 20)
FRAME_RATE = 10.0


class ReconstructionPlanV2Tests(unittest.TestCase):
    def test_render_contract_preserves_landscape_and_portrait_dimensions(self) -> None:
        for width, height in ((1920, 1080), (720, 1280)):
            with self.subTest(width=width, height=height):
                plan = _build_plan(width, height, [])

                self.assertEqual(width, plan["render"]["source_width"])
                self.assertEqual(height, plan["render"]["source_height"])

    def test_render_contract_preserves_cycles_gpu_settings(self) -> None:
        cycles_configuration = {
            "engine": "CYCLES",
            "cycles_compute_device": "OPTIX",
            "cycles_samples": 16,
            "cycles_use_denoising": True,
        }

        plan = _build_plan(1280, 720, [], render_configuration=cycles_configuration)

        validate_reconstruction_plan_v2(plan)
        self.assertEqual(cycles_configuration["cycles_compute_device"], plan["render"]["cycles_compute_device"])

    def test_render_contract_rejects_nonpositive_source_dimensions(self) -> None:
        plan = _build_plan(1280, 720, [])
        for field_name, invalid_value in (("source_width", 0), ("source_height", -1)):
            with self.subTest(field_name=field_name):
                invalid_plan = copy.deepcopy(plan)
                invalid_plan["render"][field_name] = invalid_value

                with self.assertRaisesRegex(PlanValidationError, field_name):
                    validate_reconstruction_plan_v2(invalid_plan)

    def test_entering_and_exiting_tracks_survive_likely_entity_hint(self) -> None:
        tracks = [
            _track("person_continuous", [_detection(8), _detection(9), _detection(21), _detection(22)]),
            _track("person_enters", [_detection(21), _detection(22)]),
            _track("person_exits", [_detection(8), _detection(9)]),
        ]
        plan = _build_plan(
            100,
            100,
            tracks,
            likely_gap_entities={"0": ["person_continuous"]},
        )

        lifecycles = {entity["id"]: entity["lifecycle"] for entity in plan["entities"]}
        self.assertEqual("enters", lifecycles["person_enters"])
        self.assertEqual("exits", lifecycles["person_exits"])
        self.assertEqual(3, plan["selection_report"]["candidate_count"])

    def test_relevance_uses_detection_nearest_current_gap(self) -> None:
        near_gap = {"frame": 9, "bbox": [10, 10, 20, 20], "confidence": 0.95}
        distant_final = {"frame": 100, "bbox": [0, 0, 100, 100], "confidence": 0.95}
        video = {"width": 100, "height": 100}

        score = _relevance_score(
            {"detections": [near_gap, distant_final]}, "continuous", 1.0, video, HIDDEN_RANGE,
        )
        boundary_only_score = _relevance_score(
            {"detections": [near_gap]}, "continuous", 1.0, video, HIDDEN_RANGE,
        )

        self.assertEqual(boundary_only_score, score)
        self.assertEqual(0.24, score)

    def test_environment_never_invents_street_buildings_for_vehicles(self) -> None:
        person_plan = _build_plan(100, 100, [_track("person_1", [_detection(8), _detection(9)])])
        vehicle_plan = _build_plan(
            100,
            100,
            [_track("car_1", [_detection(8), _detection(9)], class_name="car")],
        )

        self.assertEqual("neutral", person_plan["environment"]["proxy_profile"])
        self.assertEqual("neutral", vehicle_plan["environment"]["proxy_profile"])

    def test_dynamic_camera_uses_stabilized_visible_backplate(self) -> None:
        plan = _build_plan(
            100,
            100,
            [_track("person_1", [_detection(8), _detection(9)])],
            context_frame_path=Path(__file__),
        )

        self.assertTrue(plan["environment"]["hybrid_backplate_enabled"])
        self.assertFalse(plan["environment"]["show_debug_grid"])
        self.assertEqual(
            "stabilized_visible_boundary_for_dynamic_camera",
            plan["environment"]["hybrid_backplate_reason"],
        )

    def test_overlapping_duplicate_tracks_are_suppressed(self) -> None:
        tracks = [
            _track("person_primary", [_detection(8), _detection(9)]),
            _track("person_duplicate", [_detection(8), _detection(9)]),
        ]

        plan = _build_plan(100, 100, tracks)

        self.assertEqual(1, len(plan["entities"]))
        self.assertEqual(2, plan["selection_report"]["candidate_count"])

    def test_person_plan_carries_visible_pose_grounded_motion_profile(self) -> None:
        before = _detection(9)
        after = _detection(21)
        before["pose_evidence"] = _pose_evidence()
        after["pose_evidence"] = _pose_evidence()

        plan = _build_plan(100, 100, [_track("person_1", [before, after])])

        profile = plan["entities"][0]["motion_profile"]
        self.assertEqual("yolo_pose_visible_boundaries", profile["source"])
        self.assertEqual([9, 21], [item["frame"] for item in profile["evidence"]])


def _build_plan(
    width: int,
    height: int,
    tracks: list[dict],
    likely_gap_entities: dict[str, list[str]] | None = None,
    render_configuration: dict | None = None,
    context_frame_path: Path | None = None,
) -> dict:
    scene_report = {
        "video": {"width": width, "height": height, "fps": FRAME_RATE, "frames": 120},
        "tracks": tracks,
        "likely_gap_entities": likely_gap_entities or {},
        "camera_motion_report": {
            "classification": "static",
            "static_feature_inlier_score": 0.9,
            "camera_motion_fit_score": 0.9,
        },
    }
    return build_reconstruction_plan_v2(
        scene_report,
        _identity_registry(tracks),
        HIDDEN_RANGE,
        gap_index=0,
        context_frame_path=context_frame_path,
        render_configuration=render_configuration,
    )


def _track(track_id: str, detections: list[dict], class_name: str = "person") -> dict:
    return {
        "id": track_id,
        "class_name": class_name,
        "continuity_confidence": 0.95,
        "avg_confidence": 0.95,
        "detections": detections,
        "last_bbox": detections[-1]["bbox"],
    }


def _detection(frame_index: int) -> dict:
    return {
        "frame": frame_index,
        "bbox": [20, 20, 60, 90],
        "confidence": 0.95,
    }


def _pose_evidence() -> dict:
    return {
        "schema_version": 1,
        "format": "coco17_bbox_normalized",
        "keypoints": [[0.5, 0.5, 0.9] for _ in range(17)],
    }


def _identity_registry(tracks: list[dict]) -> dict:
    identities = {
        track["id"]: {
            "appearance": {
                "upper_color": [0.2, 0.4, 0.6],
                "lower_color": [0.1, 0.2, 0.3],
                "vehicle_color": [0.2, 0.4, 0.6],
            },
            "body_proportions": {"height_scale": 1.0, "shoulder_scale": 1.0, "limb_scale": 1.0},
            "associated_objects": [],
            "animation_phase": 0.0,
        }
        for track in tracks
    }
    return {
        "schema_version": 1,
        "generator_version": "test",
        "identities": identities,
    }


if __name__ == "__main__":
    unittest.main()
