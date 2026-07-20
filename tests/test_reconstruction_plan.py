import sys
import unittest
from pathlib import Path


sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "src")))

from reconstruction_plan import build_reconstruction_plan


class ReconstructionPlanTests(unittest.TestCase):
    def test_plan_uses_boundary_evidence_and_confidence(self) -> None:
        track = {
            "id": "person_1",
            "class_name": "person",
            "direction": "right",
            "frames_seen": 6,
            "avg_area": 4_800,
            "avg_confidence": 0.9,
            "continuity_confidence": 0.8,
            "associated_objects": ["backpack_1"],
            "detections": [
                {"frame": 94, "bbox": [20, 20, 60, 140], "confidence": 0.9},
                {"frame": 97, "bbox": [24, 20, 64, 140], "confidence": 0.9},
                {"frame": 99, "bbox": [27, 20, 67, 140], "confidence": 0.9},
                {"frame": 131, "bbox": [70, 20, 110, 140], "confidence": 0.9},
                {"frame": 134, "bbox": [74, 20, 114, 140], "confidence": 0.9},
                {"frame": 136, "bbox": [77, 20, 117, 140], "confidence": 0.9},
            ],
        }

        plan = build_reconstruction_plan({"tracks": [track]}, (100, 130), 30.0)

        self.assertEqual("evidence_grounded_2_5d_compositing", plan["strategy"])
        self.assertEqual(1, len(plan["entities"]))
        entity = plan["entities"][0]
        self.assertEqual(31, len(entity["path"]))
        self.assertEqual(["backpack_1"], entity["associated_objects"])
        self.assertGreater(entity["confidence"], 0.5)
        self.assertTrue(all(point["opacity"] == 1.0 for point in entity["path"]))


if __name__ == "__main__":
    unittest.main()
