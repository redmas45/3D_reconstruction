import copy
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from domain.clue_catalog import build_clue_catalog
from domain.gap_decisions import (
    GapDecisionValidationError,
    apply_gap_decisions,
    build_deterministic_gap_decisions,
    validate_gap_decisions,
)
from domain.gap_hypotheses import build_gap_hypotheses
from domain.reconstruction_narrative import (
    MAXIMUM_TEXT_LENGTH,
    build_deterministic_narrative,
    validate_narrative,
)
from domain.render_storyboard import compile_render_storyboard


class StoryReasoningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.scene = _scene_fixture()
        self.plan = _plan_fixture()
        self.clues = build_clue_catalog(self.scene, [self.plan])
        self.hypotheses = build_gap_hypotheses([self.plan], self.clues)
        self.decisions = build_deterministic_gap_decisions(
            "evidence-digest", self.clues, self.hypotheses, "test fallback",
        )

    def test_per_entity_hypotheses_include_proxy_for_low_confidence_entity(self) -> None:
        hypothesis_types = {
            item["type"]
            for item in self.hypotheses["gaps"][0]["entities"][0]["hypotheses"]
        }
        self.assertIn("identity_unresolved_proxy", hypothesis_types)
        self.assertIn("continue_measured_motion", hypothesis_types)

    def test_decision_rejects_clue_identifier_that_was_not_supplied(self) -> None:
        malformed = copy.deepcopy(self.decisions)
        malformed["decisions"][0]["clue_ids"].append("hidden_frame_clue")
        with self.assertRaisesRegex(GapDecisionValidationError, "unknown clue"):
            validate_gap_decisions(
                malformed, "evidence-digest", self.clues, self.hypotheses,
            )

    def test_decision_rebuilds_references_from_valid_clue_identifiers(self) -> None:
        malformed = copy.deepcopy(self.decisions)
        malformed["decisions"][0]["evidence_references"] = ["hidden:frame:15"]

        validated = validate_gap_decisions(
            malformed, "evidence-digest", self.clues, self.hypotheses,
        )

        self.assertEqual(
            self.decisions["decisions"][0]["evidence_references"],
            validated["decisions"][0]["evidence_references"],
        )
        self.assertNotIn("hidden:frame:15", validated["decisions"][0]["evidence_references"])

    def test_fallback_bounds_scene_wide_evidence_references(self) -> None:
        large_catalog = copy.deepcopy(self.clues)
        scene_tracks = next(
            item for item in large_catalog["clues"] if item["id"] == "scene_tracks"
        )
        scene_tracks["evidence_references"] = [
            f"track:person_{index}:visible_observations" for index in range(100)
        ]
        decisions = build_deterministic_gap_decisions(
            "evidence-digest", large_catalog, self.hypotheses, "test fallback",
        )

        validated = validate_gap_decisions(
            decisions, "evidence-digest", large_catalog, self.hypotheses,
        )

        references = validated["decisions"][0]["evidence_references"]
        self.assertEqual(32, len(references))
        self.assertEqual("gap:0:camera_calibration", references[0])

    def test_validated_decision_compiles_plan_storyboard_and_narrative(self) -> None:
        validated = validate_gap_decisions(
            self.decisions, "evidence-digest", self.clues, self.hypotheses,
        )
        updated_plan = apply_gap_decisions([self.plan], self.hypotheses, validated)[0]
        narrative = build_deterministic_narrative(
            self.clues, validated, "deterministic_fallback", "test fallback",
        )
        validated_narrative = validate_narrative(
            narrative, self.clues, validated, "deterministic_fallback",
        )
        storyboard, shell, budget = compile_render_storyboard(
            self.scene, [updated_plan], self.hypotheses, validated,
            {"target_fps": 10, "scale_percent": 50, "cycles_samples": 4},
        )
        self.assertIn("reasoning_decision_v2", updated_plan)
        self.assertFalse(validated_narrative["causal_link_supported"])
        self.assertEqual(5, len(budget["gaps"][0]["diagnostic_pose_frames"]))
        self.assertEqual("storyboard/scene_shell.blend", shell["scene_file"])
        self.assertEqual("proxy", storyboard["gaps"][0]["entities"][0]["fidelity_tier"])

    def test_presentation_prose_cannot_change_renderer_storyboard(self) -> None:
        validated = validate_gap_decisions(
            self.decisions, "evidence-digest", self.clues, self.hypotheses,
        )
        plan = apply_gap_decisions([self.plan], self.hypotheses, validated)[0]
        first, _, _ = compile_render_storyboard(
            self.scene, [plan], self.hypotheses, validated, {"target_fps": 10},
        )
        narrative = build_deterministic_narrative(
            self.clues, validated, "deterministic_fallback", None,
        )
        narrative["whole_video_summary"] = "Presentation wording changed."
        validate_narrative(narrative, self.clues, validated, "deterministic_fallback")
        second, _, _ = compile_render_storyboard(
            self.scene, [plan], self.hypotheses, validated, {"target_fps": 10},
        )
        self.assertEqual(first["storyboard_digest"], second["storyboard_digest"])

    def test_multi_gap_fallback_narrative_stays_within_schema_limit(self) -> None:
        decisions = copy.deepcopy(self.decisions)
        template = decisions["decisions"][0]
        decisions["decisions"] = [
            {
                **copy.deepcopy(template),
                "gap_index": gap_index,
                "gap_summary": f"Gap {gap_index}: " + ("bounded visible evidence. " * 15),
            }
            for gap_index in range(7)
        ]

        narrative = build_deterministic_narrative(
            self.clues, decisions, "deterministic_fallback", "Azure response was invalid.",
        )
        validated = validate_narrative(
            narrative, self.clues, decisions, "deterministic_fallback",
        )

        self.assertLessEqual(len(validated["whole_video_summary"]), MAXIMUM_TEXT_LENGTH)
        self.assertIn("Across 7 missing intervals", validated["whole_video_summary"])


def _scene_fixture() -> dict:
    return {
        "video": {
            "width": 100, "height": 100, "fps": 10.0, "frames": 120,
            "sha256": "source-digest",
        },
        "visible_ranges": [{"start": 0, "end": 9}, {"start": 21, "end": 119}],
        "hidden_ranges": [{"start": 10, "end": 20}],
        "tracks": [{
            "id": "person_1",
            "class_name": "person",
            "avg_confidence": 0.45,
            "direction": "right",
            "detections": [
                {"frame": 8, "bbox": [20, 20, 60, 90], "confidence": 0.45},
                {"frame": 9, "bbox": [22, 20, 62, 90], "confidence": 0.45},
                {"frame": 21, "bbox": [40, 20, 80, 90], "confidence": 0.45},
                {"frame": 22, "bbox": [42, 20, 82, 90], "confidence": 0.45},
            ],
        }],
        "camera_motion_report": {
            "classification": "static",
            "static_feature_inlier_score": 0.9,
            "camera_motion_fit_score": 0.9,
        },
    }


def _plan_fixture() -> dict:
    waypoints = [
        {"role": "start", "frame": 10, "world": [0.0, 0.0, 0.0]},
        {"role": "inferred_midpoint", "frame": 15, "world": [1.0, 0.0, 0.0]},
        {"role": "predicted_end", "frame": 20, "world": [2.0, 0.0, 0.0]},
    ]
    return {
        "schema_version": 2,
        "strategy": "ai_inferred_forensic_3d",
        "gap_index": 0,
        "hidden_range": {"start": 10, "end": 20},
        "fps": 10.0,
        "frame_count": 11,
        "duration_seconds": 1.1,
        "overall_confidence": 0.45,
        "camera": {"calibration_confidence": 0.9},
        "environment": {"style": "forensic_3d"},
        "render": {
            "engine": "BLENDER_EEVEE_NEXT",
            "preview_scale_percent": 50,
            "production_scale_percent": 50,
            "source_width": 100,
            "source_height": 100,
        },
        "entities": [{
            "id": "person_1",
            "kind": "person",
            "confidence": 0.45,
            "fidelity_tier": "weak",
            "lifecycle": "continuous",
            "boundary_evidence": {
                "before_frame": 9,
                "after_frame": 21,
                "heading_disagreement_degrees": 2.0,
                "post_gap_position_residual_meters": 0.2,
            },
            "animation": {
                "state": "walk",
                "speed_meters_per_second": 1.2,
                "phase_offset": 0.0,
            },
            "path_prediction": {"method": "centripetal_catmull_rom", "waypoints": waypoints},
        }],
    }


if __name__ == "__main__":
    unittest.main()
