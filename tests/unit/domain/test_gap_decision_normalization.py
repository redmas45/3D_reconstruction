import copy
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from domain.gap_decision_normalization import normalize_gap_decision_references


class GapDecisionNormalizationTests(unittest.TestCase):
    def test_maps_unique_hypothesis_types_to_canonical_identifiers(self) -> None:
        source = _decision_fixture(
            selected="continue_measured_motion",
            rejected=[
                {
                    "id": "hold_position",
                    "reason": "Visible motion makes a stationary hold less likely.",
                },
            ],
        )

        normalized, report = normalize_gap_decision_references(
            source, _hypothesis_library(),
        )

        entity = normalized["decisions"][0]["entities"][0]
        self.assertEqual(_identifier("continue_measured_motion"), entity["selected_hypothesis_id"])
        self.assertEqual(_identifier("hold_position"), entity["rejected_hypotheses"][0]["id"])
        self.assertEqual(1, report["repaired_selected_hypotheses"])
        self.assertEqual(1, report["repaired_rejected_hypotheses"])

    def test_drops_unknown_duplicate_and_selected_rejections(self) -> None:
        selected = _identifier("continue_measured_motion")
        source = _decision_fixture(
            selected=selected,
            rejected=[
                {"id": "invented_motion", "reason": "Not in the supplied library."},
                {"id": selected, "reason": "Cannot reject the selected hypothesis."},
                {"id": "hold_position", "reason": "Motion remains visible."},
                {"id": _identifier("hold_position"), "reason": "Duplicate alternative."},
                {"id": "continue_reduced_motion", "reason": ""},
            ],
        )

        normalized, report = normalize_gap_decision_references(
            source, _hypothesis_library(),
        )

        rejected = normalized["decisions"][0]["entities"][0]["rejected_hypotheses"]
        self.assertEqual([_identifier("hold_position")], [item["id"] for item in rejected])
        self.assertEqual(4, report["dropped_rejected_hypotheses"])

    def test_does_not_mutate_the_model_response(self) -> None:
        source = _decision_fixture(
            selected="continue_measured_motion",
            rejected=[{"id": "hold_position", "reason": "Motion remains visible."}],
        )
        original = copy.deepcopy(source)

        normalize_gap_decision_references(source, _hypothesis_library())

        self.assertEqual(original, source)


def _identifier(hypothesis_type: str) -> str:
    return f"gap_00_person_1_{hypothesis_type}"


def _hypothesis_library() -> dict:
    return {
        "gaps": [{
            "gap_index": 0,
            "entities": [{
                "entity_id": "person_1",
                "hypotheses": [
                    {"id": _identifier("continue_measured_motion"), "type": "continue_measured_motion"},
                    {"id": _identifier("continue_reduced_motion"), "type": "continue_reduced_motion"},
                    {"id": _identifier("hold_position"), "type": "hold_position"},
                ],
            }],
        }],
    }


def _decision_fixture(selected: str, rejected: list[dict]) -> dict:
    return {
        "schema_version": 2,
        "decisions": [{
            "gap_index": 0,
            "entities": [{
                "entity_id": "person_1",
                "selected_hypothesis_id": selected,
                "decision_summary": "Visible motion supports the selected continuation.",
                "rejected_hypotheses": rejected,
                "confidence": 0.8,
            }],
        }],
    }


if __name__ == "__main__":
    unittest.main()
