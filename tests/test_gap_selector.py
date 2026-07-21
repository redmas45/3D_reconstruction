import random
import sys
import unittest
from pathlib import Path


sys.path.insert(0, str((Path(__file__).resolve().parents[1] / "src")))

from gap_selector import choose_hidden_gaps


class GapSelectorTests(unittest.TestCase):
    def test_short_gaps_total_twenty_five_percent(self) -> None:
        total_frames = 3_600
        fps = 30.0
        selection = choose_hidden_gaps(total_frames, fps, random.Random(42))

        self.assertEqual(round(total_frames * 0.25), selection["missing_frames"])
        self.assertEqual(total_frames, sum(item["frame_count"] for item in selection["timeline"]))
        self.assertGreater(selection["gap_count"], 1)
        for start, end in selection["hidden_ranges"]:
            duration_seconds = (end - start + 1) / fps
            self.assertGreaterEqual(duration_seconds, 1.0)
            self.assertLessEqual(duration_seconds, 3.0)

    def test_timeline_is_contiguous_and_alternating(self) -> None:
        selection = choose_hidden_gaps(1_200, 24.0, random.Random(9))
        timeline = selection["timeline"]

        self.assertEqual(0, timeline[0]["start"])
        self.assertEqual(1_199, timeline[-1]["end"])
        for index, segment in enumerate(timeline):
            self.assertEqual("visible" if index % 2 == 0 else "hidden", segment["kind"])
            if index:
                self.assertEqual(timeline[index - 1]["end"] + 1, segment["start"])

    def test_video_shorter_than_gap_policy_fails_with_clear_message(self) -> None:
        with self.assertRaisesRegex(ValueError, "use at least 4.00 seconds"):
            choose_hidden_gaps(90, 30.0, random.Random(3))


if __name__ == "__main__":
    unittest.main()
