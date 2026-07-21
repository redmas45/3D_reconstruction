import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from domain.processing_job import JobStatus, ProcessingJob, ProcessingStage, utc_now
from infrastructure.job_metadata_store import JobMetadataStore


class JobMetadataStoreTests(unittest.TestCase):
    def test_throttles_same_stage_updates_but_persists_every_transition(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory)
            record = ProcessingJob(
                job_id="a" * 32,
                source_name="fixture.mp4",
                input_path=Path("fixture.mp4"),
                output_dir=output_dir,
                created_at=utc_now(),
            )
            store = JobMetadataStore(persist_interval_seconds=30.0)

            with patch(
                "infrastructure.job_metadata_store.time.monotonic",
                side_effect=[1.0, 2.0, 32.0, 33.0, 34.0],
            ):
                self.assertTrue(store.persist(record))
                record.progress = 0.1
                record.detail = "Same stage progress"
                self.assertFalse(store.persist(record))
                self.assertEqual(0.0, self._payload(output_dir)["progress"])

                record.progress = 0.2
                self.assertTrue(store.persist(record))
                self.assertEqual(0.2, self._payload(output_dir)["progress"])

                record.status = JobStatus.PROCESSING
                record.stage = ProcessingStage.VALIDATING
                self.assertTrue(store.persist(record))
                self.assertEqual("validating", self._payload(output_dir)["stage"])

                record.status = JobStatus.FAILED
                record.stage = ProcessingStage.FAILED
                self.assertTrue(store.persist(record))
                self.assertEqual("failed", self._payload(output_dir)["status"])

    @staticmethod
    def _payload(output_dir: Path) -> dict[str, object]:
        return json.loads((output_dir / "job.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
