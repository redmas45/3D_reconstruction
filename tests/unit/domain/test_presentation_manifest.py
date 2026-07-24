import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from domain.presentation_manifest import build_presentation_manifest


class PresentationManifestTests(unittest.TestCase):
    def test_manifest_exposes_story_clues_and_precise_gap_markers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            work_directory = Path(temporary_directory)
            _write_reasoning(work_directory)
            plan_path = _write_plan(work_directory)

            manifest = build_presentation_manifest(
                {
                    "frames": 2400,
                    "fps": 20.0,
                    "width": 1280,
                    "height": 720,
                },
                {
                    "hidden_ranges": [[200, 319]],
                    "missing_fraction_actual": 0.25,
                },
                {
                    "people_count": 1,
                    "vehicle_count": 0,
                    "tracks": [{"id": "person_1"}],
                },
                [plan_path],
                work_directory,
                work_directory / "result.mp4",
                "blender",
            )

        self.assertEqual(0.75, manifest["source"]["observed_fraction"])
        self.assertEqual(10.0, manifest["gaps"][0]["start_seconds"])
        self.assertEqual(6.0, manifest["gaps"][0]["duration_seconds"])
        self.assertEqual("Strong visible clue", manifest["top_clues"][0]["statement"])
        self.assertEqual("clue_1", manifest["gaps"][0]["clues"][0]["id"])
        self.assertEqual(
            ["track:person_1:visible_observations"],
            manifest["gaps"][0]["evidence_references"],
        )
        self.assertEqual(
            "continue_measured_motion",
            manifest["gaps"][0]["entities"][0]["selected_hypothesis_id"],
        )
        self.assertEqual("azure", manifest["story"]["planning_mode"])
        self.assertEqual(3, manifest["schema_version"])
        self.assertEqual(90.0, manifest["evidence_overview"]["observed_seconds"])
        self.assertEqual(1, manifest["evidence_overview"]["tracked_entity_count"])
        self.assertEqual("Public decision trace", manifest["method"]["label"])
        self.assertEqual(
            "Stylized 3D actors composited over observed scene context",
            manifest["gaps"][0]["patch"]["method"],
        )
        self.assertNotIn("truth_path", json.dumps(manifest).lower())


def _write_reasoning(work_directory: Path) -> None:
    payload = {
        "headline": "Observed movement continues",
        "whole_video_summary": "Visible evidence supports bounded continuation.",
        "confidence": 0.8,
        "causal_link_supported": False,
        "mode": "azure",
        "deployment": "gpt-5.4-mini",
        "story_points": [{"statement": "A person moves right."}],
        "clues": [
            {
                "id": "clue_1",
                "category": "motion",
                "statement": "Strong visible clue",
                "confidence": 0.9,
            },
        ],
        "gap_summaries": [{
            "gap_index": 0,
            "before_observed": "Person moving right.",
            "inside_inferred": "Motion likely continues.",
            "after_observed": "Person remains right of the start.",
            "confidence": 0.8,
            "unknowns": ["Exact pose"],
        }],
        "decisions": [{
            "gap_index": 0,
            "gap_summary": "Continue right.",
            "confidence": 0.8,
            "clue_ids": ["clue_1"],
            "evidence_references": ["track:person_1:visible_observations"],
            "unknowns": ["Exact pose"],
            "entities": [{
                "entity_id": "person_1",
                "selected_hypothesis_id": "continue_measured_motion",
                "decision_summary": "Measured motion is consistent across the gap.",
                "confidence": 0.85,
                "rejected_hypotheses": [{
                    "id": "hold_position",
                    "reason": "Visible velocity supports continued movement.",
                }],
            }],
            "event_beats": [{
                "time_fraction": 0.0,
                "action": "continue",
                "entity_ids": ["person_1"],
            }],
        }],
    }
    (work_directory / "reasoning_public.json").write_text(
        json.dumps(payload), encoding="utf-8",
    )


def _write_plan(work_directory: Path) -> Path:
    plan_path = work_directory / "plan.json"
    plan_path.write_text(json.dumps({
        "gap_index": 0,
        "overall_confidence": 0.8,
        "entities": [{"id": "person_1"}],
        "camera": {"calibration_confidence": 0.75},
        "render": {
            "engine": "CYCLES",
            "target_fps": 6,
            "production_hud_mode": "minimal",
        },
        "environment": {"hybrid_backplate_enabled": True},
    }), encoding="utf-8")
    return plan_path


if __name__ == "__main__":
    unittest.main()
