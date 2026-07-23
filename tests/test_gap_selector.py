import random
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "src")))

from gap_selector import choose_hidden_gaps


class GapSelectorTests(unittest.TestCase):
    def test_review_gaps_total_twenty_five_percent(self) -> None:
        total_frames = 3_600
        fps = 30.0
        selection = choose_hidden_gaps(total_frames, fps, random.Random(42))

        self.assertEqual("review", selection["profile"])
        self.assertEqual(round(total_frames * 0.25), selection["missing_frames"])
        self.assertEqual(total_frames, sum(item["frame_count"] for item in selection["timeline"]))
        self.assertGreater(selection["gap_count"], 1)
        for start, end in selection["hidden_ranges"]:
            duration_seconds = (end - start + 1) / fps
            self.assertGreaterEqual(duration_seconds, 5.0)
            self.assertLessEqual(duration_seconds, 7.0)

    def test_timeline_is_contiguous_and_alternating(self) -> None:
        selection = choose_hidden_gaps(2_400, 24.0, random.Random(9))
        timeline = selection["timeline"]

        self.assertEqual(0, timeline[0]["start"])
        self.assertEqual(2_399, timeline[-1]["end"])
        for index, segment in enumerate(timeline):
            self.assertEqual("visible" if index % 2 == 0 else "hidden", segment["kind"])
            if index:
                self.assertEqual(timeline[index - 1]["end"] + 1, segment["start"])

    def test_short_video_uses_compact_profile(self) -> None:
        selection = choose_hidden_gaps(900, 30.0, random.Random(3))

        self.assertEqual("compact", selection["profile"])
        self.assertEqual(round(900 * 0.25), selection["missing_frames"])
        self.assertTrue(all(
            1.0 <= duration <= 3.0
            for duration in selection["gap_durations_seconds"]
        ))

    def test_video_shorter_than_compact_gap_policy_fails_with_clear_message(self) -> None:
        with self.assertRaisesRegex(ValueError, "use at least 4.00 seconds"):
            choose_hidden_gaps(90, 30.0, random.Random(3))


if __name__ == "__main__":
    unittest.main()
