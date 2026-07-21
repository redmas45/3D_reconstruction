import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from domain.processing_job import ProcessingJob


class ProcessingJobStorageTests(unittest.TestCase):
    def test_loaded_timestamps_are_normalized_to_timezone_aware_utc(self) -> None:
        payload = self._payload()
        payload["created_at"] = "2026-07-21T12:30:00+05:30"
        payload["started_at"] = "2026-07-21T07:00:01Z"
        payload["progress_updated_at"] = "2026-07-21T07:00:02+00:00"
        payload["activity_log"] = [
            {
                "timestamp": "2026-07-21T12:30:03+05:30",
                "stage": "queued",
                "detail": "Waiting",
                "progress": 0.0,
            }
        ]

        record = ProcessingJob.from_storage_payload(Path("uploads"), Path("outputs") / ("a" * 32), payload)

        self.assertEqual("2026-07-21T07:00:00+00:00", record.created_at)
        self.assertEqual("2026-07-21T07:00:01+00:00", record.started_at)
        self.assertEqual("2026-07-21T07:00:02+00:00", record.progress_updated_at)
        self.assertEqual("2026-07-21T07:00:03+00:00", record.activity_log[0]["timestamp"])

    def test_loaded_primary_timestamp_rejects_missing_timezone(self) -> None:
        payload = self._payload()
        payload["created_at"] = "2026-07-21T07:00:00"

        with self.assertRaisesRegex(ValueError, "must include a timezone"):
            ProcessingJob.from_storage_payload(Path("uploads"), Path("outputs") / ("a" * 32), payload)

    def test_activity_with_missing_timezone_is_ignored(self) -> None:
        payload = self._payload()
        payload["activity_log"] = [
            {
                "timestamp": "2026-07-21T07:00:00",
                "stage": "queued",
                "detail": "Invalid timestamp",
                "progress": 0.0,
            }
        ]

        record = ProcessingJob.from_storage_payload(Path("uploads"), Path("outputs") / ("a" * 32), payload)

        self.assertEqual([], record.activity_log)

    @staticmethod
    def _payload() -> dict[str, object]:
        return {
            "job_id": "a" * 32,
            "source_name": "fixture.mp4",
            "input_file": "fixture.mp4",
            "status": "queued",
            "stage": "queued",
            "progress": 0.0,
            "detail": "Waiting",
            "created_at": "2026-07-21T07:00:00+00:00",
        }


if __name__ == "__main__":
    unittest.main()
