import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from application.evidence_reasoning import (
    _decision_batch_schema,
    _narrative_schema,
    reason_about_reconstruction,
)
from domain.gap_decisions import GapDecisionValidationError
from domain.reconstruction_plan_v2 import build_reconstruction_plan_v2, write_reconstruction_plan_v2


class EvidenceReasoningApplicationTests(unittest.TestCase):
    def test_azure_schemas_pin_immutable_evidence_context(self) -> None:
        payload = {
            "evidence_digest": "evidence",
            "clue_digest": "clues",
            "hypothesis_digest": "hypotheses",
        }

        decision_schema = _decision_batch_schema(payload)
        narrative_schema = _narrative_schema({"clue_digest": "clues"}, "azure")

        self.assertEqual(
            ["evidence"], decision_schema["properties"]["evidence_digest"]["enum"],
        )
        self.assertEqual(
            ["clues"], decision_schema["properties"]["clue_digest"]["enum"],
        )
        self.assertEqual(
            ["hypotheses"], decision_schema["properties"]["hypothesis_digest"]["enum"],
        )
        self.assertEqual(
            ["clues"], narrative_schema["properties"]["clue_digest"]["enum"],
        )
        self.assertEqual(["azure"], narrative_schema["properties"]["mode"]["enum"])

    def test_unconfigured_azure_writes_explicit_fallback_artifacts_and_plan(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            work_directory = Path(temporary_directory)
            plan_path = work_directory / "gaps" / "gap_00" / "blender" / "plan_v2.json"
            scene_report, identity_registry = _evidence_fixtures()
            plan = build_reconstruction_plan_v2(
                scene_report,
                identity_registry,
                hidden_range=(10, 20),
                gap_index=0,
            )
            write_reconstruction_plan_v2(plan, plan_path)

            with patch.dict(os.environ, {}, clear=True):
                result = reason_about_reconstruction(
                    scene_report,
                    [plan_path],
                    work_directory,
                    {"enabled": True},
                    reuse_work=False,
                )

            public_summary = json.loads((work_directory / "reasoning_public.json").read_text(encoding="utf-8"))
            updated_plan = json.loads(plan_path.read_text(encoding="utf-8"))
            self.assertEqual("deterministic_fallback", result.mode)
            self.assertEqual("deterministic_fallback", public_summary["mode"])
            self.assertIn("reasoning_decision", updated_plan)
            self.assertTrue((work_directory / "evidence_ledger.json").is_file())
            self.assertTrue((work_directory / "decision_trace.json").is_file())

    def test_invalid_azure_decisions_complete_with_deterministic_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            work_directory = Path(temporary_directory)
            plan_path = work_directory / "gaps" / "gap_00" / "blender" / "plan_v2.json"
            scene_report, identity_registry = _evidence_fixtures()
            plan = build_reconstruction_plan_v2(
                scene_report,
                identity_registry,
                hidden_range=(10, 20),
                gap_index=0,
            )
            write_reconstruction_plan_v2(plan, plan_path)
            azure_environment = {
                "AZURE_OPENAI_BASE_URL": "https://example.openai.azure.com",
                "AZURE_OPENAI_API_KEY": "test-api-key",
                "AZURE_OPENAI_CHAT_DEPLOYMENT": "test-deployment",
            }

            with (
                patch.dict(os.environ, azure_environment, clear=True),
                patch(
                    "application.evidence_reasoning._request_decision_batches",
                    side_effect=GapDecisionValidationError("invalid Azure decision"),
                ),
            ):
                result = reason_about_reconstruction(
                    scene_report,
                    [plan_path],
                    work_directory,
                    {"enabled": True},
                    reuse_work=False,
                )

            self.assertEqual("deterministic_fallback", result.mode)
            self.assertTrue((work_directory / "reasoning_public.json").is_file())


def _evidence_fixtures() -> tuple[dict, dict]:
    detections = [
        {"frame": 8, "bbox": [20, 20, 60, 90], "confidence": 0.95},
        {"frame": 9, "bbox": [22, 20, 62, 90], "confidence": 0.95},
        {"frame": 21, "bbox": [40, 20, 80, 90], "confidence": 0.95},
        {"frame": 22, "bbox": [42, 20, 82, 90], "confidence": 0.95},
    ]
    track = {
        "id": "person_1",
        "class_name": "person",
        "continuity_confidence": 0.95,
        "avg_confidence": 0.95,
        "detections": detections,
        "last_bbox": detections[-1]["bbox"],
    }
    scene_report = {
        "video": {"width": 100, "height": 100, "fps": 10.0, "frames": 120},
        "tracks": [track],
        "camera_motion_report": {
            "classification": "static",
            "static_feature_inlier_score": 0.9,
            "camera_motion_fit_score": 0.9,
        },
    }
    identity_registry = {
        "schema_version": 1,
        "generator_version": "test",
        "identities": {"person_1": {
            "appearance": {
                "upper_color": [0.2, 0.4, 0.6],
                "lower_color": [0.1, 0.2, 0.3],
                "vehicle_color": [0.2, 0.4, 0.6],
            },
            "body_proportions": {"height_scale": 1.0, "shoulder_scale": 1.0, "limb_scale": 1.0},
            "associated_objects": [],
            "animation_phase": 0.0,
        }},
    }
    return scene_report, identity_registry


if __name__ == "__main__":
    unittest.main()
