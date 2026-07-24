import sys
import unittest
import tempfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from domain.render_runtime_budget import (
    approve_representative_preview,
    gap_render_costs,
    predicted_total_seconds,
    preview_is_approved,
    representative_gap_index,
)


class RenderRuntimeBudgetTests(unittest.TestCase):
    def test_heaviest_gap_is_selected_as_representative(self) -> None:
        costs = gap_render_costs([
            _plan(0, duration_seconds=1.0, detailed_entities=1),
            _plan(1, duration_seconds=3.0, detailed_entities=3),
            _plan(2, duration_seconds=2.0, detailed_entities=0),
        ])

        self.assertEqual(1, representative_gap_index(costs))

    def test_representative_elapsed_time_scales_by_cost(self) -> None:
        costs = gap_render_costs([
            _plan(0, duration_seconds=1.0, detailed_entities=0),
            _plan(1, duration_seconds=2.0, detailed_entities=0),
        ])

        predicted = predicted_total_seconds(costs, 1, 20.0)

        self.assertEqual(30.0, predicted)

    def test_preview_approval_is_bound_to_the_plan_signature(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            approval_path = Path(temporary_directory) / "approval.json"
            approve_representative_preview(approval_path, "plan-a")

            self.assertTrue(preview_is_approved(approval_path, "plan-a"))
            self.assertFalse(preview_is_approved(approval_path, "plan-b"))


def _plan(
    gap_index: int,
    duration_seconds: float,
    detailed_entities: int,
) -> dict:
    entities = [
        {"fidelity_tier": "supported"}
        for _ in range(detailed_entities)
    ]
    return {
        "gap_index": gap_index,
        "fps": 30.0,
        "duration_seconds": duration_seconds,
        "render": {"target_fps": 10},
        "entities": entities,
    }


if __name__ == "__main__":
    unittest.main()
