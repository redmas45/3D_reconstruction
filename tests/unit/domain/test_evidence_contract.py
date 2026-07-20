import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from domain.evidence_contract import EvidenceContractError, validate_visible_evidence_only


class EvidenceContractTests(unittest.TestCase):
    def test_accepts_visible_only_detections(self) -> None:
        report = {
            "hidden_ranges": [{"start": 10, "end": 20}],
            "tracks": [{"id": "person_1", "detections": [{"frame": 9}, {"frame": 21}]}],
        }

        validate_visible_evidence_only(report)

    def test_rejects_detection_inside_hidden_range(self) -> None:
        report = {
            "hidden_ranges": [{"start": 10, "end": 20}],
            "tracks": [{"id": "person_1", "detections": [{"frame": 15}]}],
        }

        with self.assertRaisesRegex(EvidenceContractError, "forbidden hidden-frame evidence"):
            validate_visible_evidence_only(report)


if __name__ == "__main__":
    unittest.main()
