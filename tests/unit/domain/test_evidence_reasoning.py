import copy
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from domain.evidence_reasoning import (
    DecisionTraceValidationError,
    apply_decision_trace,
    build_deterministic_decision_trace,
    build_evidence_ledger,
    validate_decision_trace,
)


class EvidenceReasoningTests(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = _plan_fixture()
        self.ledger = build_evidence_ledger(_scene_fixture(), [self.plan])

    def test_ledger_contains_only_bounded_hypotheses_and_references(self) -> None:
        gap = self.ledger["gaps"][0]

        self.assertEqual(
            {"measured_continuation", "reduced_motion", "stationary_hold"},
            {item["id"] for item in gap["hypotheses"]},
        )
        self.assertNotIn("path", self.ledger)
        self.assertEqual(
            ["gap:0:camera_calibration", "track:person_1:pre_boundary", "track:person_1:post_boundary"],
            gap["allowed_evidence_references"],
        )

    def test_trace_rejects_evidence_not_present_in_ledger(self) -> None:
        trace = build_deterministic_decision_trace(self.ledger, "test fallback")
        trace["decisions"][0]["evidence_references"].append("hidden:frame:15")

        with self.assertRaisesRegex(DecisionTraceValidationError, "not supplied"):
            validate_decision_trace(trace, self.ledger)

    def test_selected_hypothesis_updates_copy_without_mutating_input(self) -> None:
        trace = build_deterministic_decision_trace(self.ledger, "test fallback")
        trace["decisions"][0]["selected_hypothesis_id"] = "stationary_hold"
        trace["decisions"][0]["rejected_hypotheses"] = [
            {"id": "measured_continuation", "reason": "Boundary evidence is ambiguous."},
            {"id": "reduced_motion", "reason": "A hold is more conservative."},
        ]
        validated_trace = validate_decision_trace(trace, self.ledger)

        updated_plan = apply_decision_trace([self.plan], self.ledger, validated_trace)[0]

        self.assertEqual([0.0, 0.0, 0.0], updated_plan["entities"][0]["path_prediction"]["waypoints"][-1]["world"])
        self.assertEqual([2.0, 0.0, 0.0], self.plan["entities"][0]["path_prediction"]["waypoints"][-1]["world"])
        self.assertEqual("idle", updated_plan["entities"][0]["animation"]["state"])


def _scene_fixture() -> dict:
    return {
        "tracks": [{"id": "person_1"}],
        "camera_motion_report": {"classification": "static"},
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
        "overall_confidence": 0.8,
        "camera": {"calibration_confidence": 0.9},
        "render": {
            "engine": "BLENDER_EEVEE_NEXT",
            "preview_scale_percent": 75,
            "production_scale_percent": 100,
            "source_width": 100,
            "source_height": 100,
        },
        "entities": [{
            "id": "person_1",
            "kind": "person",
            "confidence": 0.8,
            "fidelity_tier": "supported",
            "lifecycle": "continuous",
            "boundary_evidence": {
                "before_frame": 9,
                "after_frame": 21,
                "heading_disagreement_degrees": 2.0,
                "post_gap_position_residual_meters": 0.2,
            },
            "animation": {"state": "walk", "speed_meters_per_second": 1.2},
            "path_prediction": {"method": "centripetal_catmull_rom", "waypoints": waypoints},
        }],
    }


if __name__ == "__main__":
    unittest.main()
