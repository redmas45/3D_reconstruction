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


if __name__ == "__main__":
    unittest.main()
