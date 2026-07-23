import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from application.processing_job_view import build_public_job_record
from domain.processing_job import ProcessingJob


class ProcessingJobViewTests(unittest.TestCase):
    def test_exposes_bounded_reasoning_summary_from_owned_work_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory) / "job"
            summary_path = output_directory / "_work" / "video_digest" / "reasoning_public.json"
            summary_path.parent.mkdir(parents=True)
            summary_path.write_text(json.dumps(_reasoning_summary()), encoding="utf-8")
            record = _job_record(output_directory)

            public_record = build_public_job_record(record)

            self.assertEqual("azure", public_record["reasoning"]["mode"])
            self.assertEqual("measured_continuation", public_record["reasoning"]["decisions"][0]["selected_hypothesis_id"])

    def test_rejects_malformed_reasoning_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory) / "job"
            summary_path = output_directory / "_work" / "video_digest" / "reasoning_public.json"
            summary_path.parent.mkdir(parents=True)
            summary = _reasoning_summary()
            summary["decisions"][0]["rejected_hypotheses"] = ["invalid"]
            summary_path.write_text(json.dumps(summary), encoding="utf-8")

            self.assertIsNone(build_public_job_record(_job_record(output_directory))["reasoning"])

    def test_exposes_valid_story_v2_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_directory = Path(temporary_directory) / "job"
            summary_path = output_directory / "_work" / "video_digest" / "reasoning_public.json"
            summary_path.parent.mkdir(parents=True)
            summary_path.write_text(json.dumps(_story_v2_summary()), encoding="utf-8")

            reasoning = build_public_job_record(_job_record(output_directory))["reasoning"]

            self.assertEqual(2, reasoning["schema_version"])
            self.assertEqual("Visible motion continued.", reasoning["whole_video_summary"])
            self.assertEqual("person_1", reasoning["decisions"][0]["entities"][0]["entity_id"])


def _job_record(output_directory: Path) -> ProcessingJob:
    return ProcessingJob(
        job_id="a" * 32,
        source_name="fixture.mp4",
        input_path=Path("fixture.mp4"),
        output_dir=output_directory,
        created_at="2026-07-22T00:00:00+00:00",
    )


def _reasoning_summary() -> dict:
    return {
        "status": "completed",
        "mode": "azure",
        "deployment": "gpt-5.4",
        "warning": None,
        "scene_clues": ["Hidden frames were excluded from reasoning evidence."],
        "decisions": [{
            "gap_index": 0,
            "selected_hypothesis_id": "measured_continuation",
            "evidence_references": ["track:person_1:pre_boundary"],
            "decision_summary": "Visible motion supports continuation.",
            "rejected_hypotheses": [{"id": "stationary_hold", "reason": "Motion was visible."}],
            "confidence": 0.8,
            "unknowns": ["Exact limb pose is unknown."],
        }],
    }


def _story_v2_summary() -> dict:
    return {
        "status": "completed",
        "schema_version": 2,
        "mode": "azure",
        "deployment": "gpt-5.4",
        "warning": None,
        "scene_clues": ["Visible motion was measured."],
        "clues": [{
            "id": "motion",
            "scope": "gap:0",
            "category": "motion",
            "statement": "Visible motion was measured.",
            "confidence": 0.8,
        }],
        "headline": "Evidence-grounded reconstruction",
        "whole_video_summary": "Visible motion continued.",
        "story_points": [{
            "statement": "Visible motion continued.",
            "clue_ids": ["motion"],
            "gap_indexes": [0],
        }],
        "gap_summaries": [{
            "gap_index": 0,
            "before_observed": "Person visible before.",
            "inside_inferred": "Motion continued.",
            "after_observed": "Person visible after.",
            "confidence": 0.8,
            "unknowns": ["Exact pose unknown."],
        }],
        "causal_link_supported": False,
        "confidence": 0.8,
        "unknowns": ["Exact pose unknown."],
        "decisions": [{
            "gap_index": 0,
            "gap_summary": "Motion continued.",
            "evidence_references": ["track:person_1:pre_boundary"],
            "clue_ids": ["motion"],
            "confidence": 0.8,
            "unknowns": ["Exact pose unknown."],
            "entities": [{
                "entity_id": "person_1",
                "selected_hypothesis_id": "gap_00_person_1_continue_measured_motion",
                "decision_summary": "Visible motion supports continuation.",
                "rejected_hypotheses": [],
                "confidence": 0.8,
            }],
            "event_beats": [],
        }],
    }


if __name__ == "__main__":
    unittest.main()
