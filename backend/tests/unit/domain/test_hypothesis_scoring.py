import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from domain.hypothesis_scoring import choose_safe_hypothesis, score_hypothesis


class HypothesisScoringTests(unittest.TestCase):
    def test_endpoint_fit_uses_visible_post_boundary_as_soft_evidence(self) -> None:
        entity = _entity()
        baseline = _path(2.0)
        blended = _path(1.0)

        baseline_score = score_hypothesis(entity, "continue_measured_motion", baseline, 1.0)
        blended_score = score_hypothesis(entity, "boundary_consistent_motion", blended, 1.0)

        self.assertLess(
            blended_score["endpoint_residual_meters"],
            baseline_score["endpoint_residual_meters"],
        )
        self.assertGreater(
            blended_score["score_components"]["endpoint_fit"],
            baseline_score["score_components"]["endpoint_fit"],
        )

    def test_safety_gate_overrides_materially_worse_model_choice(self) -> None:
        hypotheses = [
            {
                "id": "safe", "selection_score": 0.82,
                "render_eligibility": True,
            },
            {
                "id": "model_choice", "selection_score": 0.50,
                "render_eligibility": True,
            },
        ]

        selected, overridden = choose_safe_hypothesis(hypotheses, "model_choice")

        self.assertTrue(overridden)
        self.assertEqual("safe", selected["id"])

    def test_invalid_candidate_is_not_selected_when_safe_candidate_exists(self) -> None:
        hypotheses = [
            {
                "id": "invalid", "selection_score": 0.99,
                "render_eligibility": False,
            },
            {
                "id": "safe", "selection_score": 0.45,
                "render_eligibility": True,
            },
        ]

        selected, overridden = choose_safe_hypothesis(hypotheses, "invalid")

        self.assertTrue(overridden)
        self.assertEqual("safe", selected["id"])


def _entity() -> dict:
    return {
        "confidence": 0.90,
        "lifecycle": "continuous",
        "boundary_evidence": {
            "heading_disagreement_degrees": 4.0,
            "post_gap_world": [1.0, 0.0, 0.0],
        },
        "animation": {"speed_meters_per_second": 1.0},
        "kinematics": {
            "duration_seconds": 1.0,
            "maximum_speed_meters_per_second": 3.0,
            "maximum_acceleration_meters_per_second_squared": 3.0,
            "maximum_turn_rate_degrees_per_second": 120.0,
        },
        "uncertainty": {"position_radius_meters": 0.5},
    }


def _path(endpoint: float) -> list[dict]:
    return [
        {"frame": 0, "world": [0.0, 0.0, 0.0]},
        {"frame": 15, "world": [endpoint * 0.5, 0.0, 0.0]},
        {"frame": 30, "world": [endpoint, 0.0, 0.0]},
    ]


if __name__ == "__main__":
    unittest.main()
