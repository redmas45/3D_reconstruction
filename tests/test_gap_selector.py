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


# A montage: nine takes separated by short dissolves, matching the structure the shot
# detector finds in real stock footage.
MONTAGE_SHOTS = [
    (0, 463), (493, 935), (957, 1279), (1309, 1723), (1749, 1971),
    (2017, 2259), (2285, 2699), (2721, 3035), (3057, 3492),
]
MONTAGE_FRAMES = 3_493
MONTAGE_FPS = 29.97


class ShotAwareGapPlacementTests(unittest.TestCase):
    """A gap must be reconstructable from the footage around it, which means the
    footage around it has to be the same scene."""

    def _selection(self, seed: int = 42) -> dict:
        return choose_hidden_gaps(
            MONTAGE_FRAMES, MONTAGE_FPS, random.Random(seed), shots=MONTAGE_SHOTS,
        )

    def _shot_holding(self, start: int, end: int):
        for index, (shot_start, shot_end) in enumerate(MONTAGE_SHOTS):
            if shot_start <= start and end <= shot_end:
                return index
        return None

    def test_every_gap_lies_wholly_inside_one_shot(self) -> None:
        for seed in range(12):
            selection = self._selection(seed)
            for start, end in selection["hidden_ranges"]:
                self.assertIsNotNone(
                    self._shot_holding(start, end),
                    f"seed {seed}: gap {start}-{end} straddles a cut",
                )

    def test_no_gap_covers_a_transition_frame(self) -> None:
        covered = {
            frame
            for start, end in self._selection()["hidden_ranges"]
            for frame in range(start, end + 1)
        }
        in_a_shot = {
            frame for start, end in MONTAGE_SHOTS for frame in range(start, end + 1)
        }
        self.assertEqual(set(), covered - in_a_shot)

    def test_each_gap_keeps_context_on_both_sides_within_its_own_shot(self) -> None:
        context_frames = round(2.0 * MONTAGE_FPS)
        for start, end in self._selection()["hidden_ranges"]:
            shot_start, shot_end = MONTAGE_SHOTS[self._shot_holding(start, end)]
            self.assertGreaterEqual(start - shot_start, context_frames)
            self.assertGreaterEqual(shot_end - end, context_frames)

    def test_the_timeline_still_covers_every_frame(self) -> None:
        selection = self._selection()
        self.assertEqual(
            MONTAGE_FRAMES, sum(item["frame_count"] for item in selection["timeline"]),
        )
        self.assertEqual(0, selection["timeline"][0]["start"])
        self.assertEqual(MONTAGE_FRAMES - 1, selection["timeline"][-1]["end"])

    def test_the_timeline_still_alternates_visible_and_hidden(self) -> None:
        timeline = self._selection()["timeline"]
        for index, segment in enumerate(timeline):
            self.assertEqual("visible" if index % 2 == 0 else "hidden", segment["kind"])
            if index:
                self.assertEqual(timeline[index - 1]["end"] + 1, segment["start"])

    def test_the_missing_fraction_is_still_met_when_the_shots_can_host_it(self) -> None:
        selection = self._selection()
        self.assertEqual(round(MONTAGE_FRAMES * 0.25), selection["missing_frames"])
        self.assertEqual(0, selection["shot_placement"]["unplaced_gap_count"])

    def test_gaps_are_spread_across_shots_rather_than_packed_into_one(self) -> None:
        placement = self._selection()["shot_placement"]
        self.assertEqual(placement["shots_hosting_gaps"], self._selection()["gap_count"])

    def test_a_video_with_no_shot_structure_behaves_as_before(self) -> None:
        """Passing no shots must be identical to declaring one shot over everything —
        the ordinary single-take case cannot regress."""
        without = choose_hidden_gaps(3_600, 30.0, random.Random(7))
        whole = choose_hidden_gaps(3_600, 30.0, random.Random(7), shots=[(0, 3_599)])
        self.assertEqual(without["hidden_ranges"], whole["hidden_ranges"])
        self.assertEqual(round(3_600 * 0.25), without["missing_frames"])


class ShotConstrainedGapPlacementTests(unittest.TestCase):
    def test_shots_too_short_for_any_gap_fail_with_a_clear_message(self) -> None:
        """Rapid-cut footage genuinely cannot host a five-second gap with context, and
        saying so beats placing one across a cut."""
        rapid = [(index * 100, index * 100 + 89) for index in range(36)]
        with self.assertRaisesRegex(ValueError, "No shot is long enough"):
            choose_hidden_gaps(3_600, 30.0, random.Random(5), shots=rapid)

    def test_a_shortfall_is_reported_rather_than_hidden(self) -> None:
        """A montage of short clips cannot hold the gaps a 25% target asks for. The
        selection must say so instead of quietly reporting a fraction it did not reach."""
        # Each 400-frame shot has room for exactly one gap once context is reserved,
        # so two of the five the target asks for can be placed and three cannot.
        shots = [(0, 399), (1_000, 1_399)]
        selection = choose_hidden_gaps(3_600, 30.0, random.Random(5), shots=shots)
        placement = selection["shot_placement"]
        self.assertGreater(placement["unplaced_gap_count"], 0)
        self.assertEqual(
            selection["missing_frames"],
            placement["requested_frames"] - placement["unplaced_frames"],
        )
        self.assertLess(selection["missing_fraction_actual"], 0.25)

    def test_reported_missing_frames_match_the_ranges_actually_chosen(self) -> None:
        # Each 400-frame shot has room for exactly one gap once context is reserved,
        # so two of the five the target asks for can be placed and three cannot.
        shots = [(0, 399), (1_000, 1_399)]
        selection = choose_hidden_gaps(3_600, 30.0, random.Random(5), shots=shots)
        self.assertEqual(
            selection["missing_frames"],
            sum(end - start + 1 for start, end in selection["hidden_ranges"]),
        )

    def test_a_shot_outside_the_video_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not fit a video"):
            choose_hidden_gaps(3_600, 30.0, random.Random(5), shots=[(0, 4_000)])


if __name__ == "__main__":
    unittest.main()
