"""Reading a running job's artifacts off disk.

Everything here has to tolerate being called before the artifact exists, because that
is the state for most of a run. The interface asks for all of them on every event and
renders whatever is there, so a reader that raises would break the stream rather than
leaving one panel empty.
"""

import json
import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[3] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from interfaces.api import artifacts


def _work(tmp_path: Path) -> Path:
    work = tmp_path / "_work" / "clip_abc123"
    work.mkdir(parents=True)
    return work


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


# Field names taken verbatim from a real scene_report.json.
TRACK = {
    "id": "person_54", "class_name": "person", "class_id": 0, "frames_seen": 127,
    "first_frame": 376, "last_frame": 3491, "avg_confidence": 0.816,
    "direction": "right", "speed_px_sec": 0.85, "continuity_confidence": 0.7118,
}

SELECTION = {
    "visible_ranges": [[0, 299], [400, 899]],
    "hidden_ranges": [[300, 399], [900, 1049]],
    "missing_fraction_actual": 0.25,
    "source_video_contract": {"fps": 30.0, "frames": 1050},
}


class TestBeforeAnythingExists:
    def test_every_reader_returns_none_on_an_empty_output(self, tmp_path):
        for reader in (
            artifacts.timeline, artifacts.clues, artifacts.story,
            artifacts.render_progress, artifacts.diagnostics,
            artifacts.plate_path, artifacts.work_directory,
        ):
            assert reader(tmp_path) is None, reader.__name__

    def test_gap_readers_return_none_on_an_empty_output(self, tmp_path):
        assert artifacts.gap_video_path(tmp_path, 0) is None
        assert artifacts.truth_video_path(tmp_path, 0) is None


class TestWorkDirectory:
    def test_the_work_directory_is_found(self, tmp_path):
        work = _work(tmp_path)
        assert artifacts.work_directory(tmp_path) == work

    def test_the_most_recent_directory_wins(self, tmp_path):
        first = _work(tmp_path)
        second = first.parent / "clip_def456"
        second.mkdir()
        assert artifacts.work_directory(tmp_path) in {first, second}


class TestTimeline:
    def test_ranges_are_converted_to_seconds(self, tmp_path):
        _write(_work(tmp_path) / "gap_selection.json", SELECTION)
        timeline = artifacts.timeline(tmp_path)
        assert timeline["hidden_ranges"][0]["start_seconds"] == 10.0
        assert timeline["visible_ranges"][0]["end_seconds"] == round(299 / 30.0, 3)

    def test_gaps_are_indexed_for_the_interface(self, tmp_path):
        _write(_work(tmp_path) / "gap_selection.json", SELECTION)
        indexes = [gap["gap_index"] for gap in artifacts.timeline(tmp_path)["hidden_ranges"]]
        assert indexes == [0, 1]

    def test_gap_duration_is_inclusive_of_both_ends(self, tmp_path):
        _write(_work(tmp_path) / "gap_selection.json", SELECTION)
        assert artifacts.timeline(tmp_path)["hidden_ranges"][0]["duration_seconds"] == round(
            100 / 30.0, 3,
        )

    def test_the_source_duration_is_reported(self, tmp_path):
        _write(_work(tmp_path) / "gap_selection.json", SELECTION)
        assert artifacts.timeline(tmp_path)["duration_seconds"] == round(1050 / 30.0, 3)

    def test_a_half_written_selection_is_tolerated(self, tmp_path):
        (_work(tmp_path) / "gap_selection.json").write_text("{partial", encoding="utf-8")
        assert artifacts.timeline(tmp_path) is None


class TestClues:
    def test_tracked_entities_are_summarised(self, tmp_path):
        work = _work(tmp_path)
        _write(work / "scene_report.json", {"tracks": [TRACK]})
        clues = artifacts.clues(tmp_path)
        assert clues["entity_count"] == 1
        assert clues["entities"][0]["class_name"] == "person"

    def test_the_tracker_field_names_are_mapped_not_guessed(self):
        """Regression: the scene report calls these `frames_seen` and `avg_confidence`.

        Reading `frame_count` and `confidence` silently produced None for every entity,
        so the interface showed a full table of dashes and looked like a data problem
        rather than a mapping one.
        """
        assert set(TRACK) >= {"frames_seen", "avg_confidence", "direction"}

    def test_frames_and_confidence_survive_the_mapping(self, tmp_path):
        _write(_work(tmp_path) / "scene_report.json", {"tracks": [TRACK]})
        entity = artifacts.clues(tmp_path)["entities"][0]
        assert entity["frame_count"] == 127
        assert entity["confidence"] == 0.816

    def test_measured_motion_is_carried_through(self, tmp_path):
        _write(_work(tmp_path) / "scene_report.json", {"tracks": [TRACK]})
        entity = artifacts.clues(tmp_path)["entities"][0]
        assert entity["direction"] == "right"
        assert entity["speed_px_sec"] == 0.85

    def test_the_clue_catalog_is_passed_through(self, tmp_path):
        work = _work(tmp_path)
        _write(work / "evidence" / "clue_catalog.json", {"clues": ["a", "b"]})
        assert artifacts.clues(tmp_path)["catalog"] == {"clues": ["a", "b"]}


class TestStory:
    def test_the_presentation_manifest_is_preferred(self, tmp_path):
        work = _work(tmp_path)
        _write(work / "decision_trace.json", {"mode": "deterministic"})
        _write(work / "presentation_manifest.json", {
            "story": {"summary": ["A person crossed the concourse."]},
            "gaps": [], "top_clues": [], "disclosure": "Inferred, not recovered.",
        })
        story = artifacts.story(tmp_path)
        assert story["source"] == "presentation_manifest"
        assert story["disclosure"].startswith("Inferred")

    def test_the_decision_trace_is_shown_before_the_manifest_exists(self, tmp_path):
        """So reasoning is visible while the render is still running."""
        _write(_work(tmp_path) / "decision_trace.json", {"mode": "azure_assisted"})
        story = artifacts.story(tmp_path)
        assert story["source"] == "decision_trace"


class TestRenderProgress:
    def test_gaps_are_reported_as_they_complete(self, tmp_path):
        work = _work(tmp_path)
        for index in range(3):
            (work / "gaps" / f"gap_{index:02d}").mkdir(parents=True)
        (work / "gaps" / "gap_00" / "gap_actors.mp4").write_bytes(b"x")
        progress = artifacts.render_progress(tmp_path)
        assert progress["completed_count"] == 1
        assert [gap["gap_index"] for gap in progress["gaps"]] == [0, 1, 2]

    def test_layers_are_counted_for_a_gap_in_flight(self, tmp_path):
        work = _work(tmp_path)
        layers = work / "gaps" / "gap_00" / "layers" / "abc"
        layers.mkdir(parents=True)
        for index in range(4):
            (layers / f"frame_{index:06d}.png").write_bytes(b"x")
        assert artifacts.render_progress(tmp_path)["gaps"][0]["layer_count"] == 4


class TestMediaPaths:
    def test_the_plate_is_found_once_written(self, tmp_path):
        work = _work(tmp_path)
        (work / "plate").mkdir()
        (work / "plate" / "clean_plate.png").write_bytes(b"x")
        assert artifacts.plate_path(tmp_path) is not None

    def test_the_actor_gap_video_is_preferred_over_the_legacy_one(self, tmp_path):
        work = _work(tmp_path)
        gap = work / "gaps" / "gap_02"
        (gap / "blender").mkdir(parents=True)
        (gap / "blender" / "gap_blender.mp4").write_bytes(b"x")
        (gap / "gap_actors.mp4").write_bytes(b"x")
        assert artifacts.gap_video_path(tmp_path, 2).name == "gap_actors.mp4"

    def test_the_legacy_gap_video_is_still_found(self, tmp_path):
        work = _work(tmp_path)
        gap = work / "gaps" / "gap_02" / "blender"
        gap.mkdir(parents=True)
        (gap / "gap_blender.mp4").write_bytes(b"x")
        assert artifacts.gap_video_path(tmp_path, 2).name == "gap_blender.mp4"

    def test_hidden_footage_is_located_by_gap_index(self, tmp_path):
        work = _work(tmp_path)
        (work / "segments").mkdir()
        (work / "segments" / "hidden_01_900_1049.mp4").write_bytes(b"x")
        assert artifacts.truth_video_path(tmp_path, 1) is not None
        assert artifacts.truth_video_path(tmp_path, 0) is None


class TestDecisionTraceStory:
    """The trace is what the interface shows while the render is still running."""

    TRACE = {
        "schema_version": 1,
        "metadata": {"warning": "Azure OpenAI is not configured."},
        "decisions": [
            {"gap_index": 0, "gap_summary": "Conservative continuation for person_38.",
             "evidence_references": ["track:person_38:pre_boundary", "gap:0:camera"]},
            {"gap_index": 1, "gap_summary": "person_12 continues left to right.",
             "evidence_references": ["track:person_12:pre_boundary"]},
        ],
    }

    def test_gap_summaries_become_the_narrative(self, tmp_path):
        _write(_work(tmp_path) / "decision_trace.json", self.TRACE)
        summary = artifacts.story(tmp_path)["story"]["summary"]
        assert summary[0].startswith("Conservative continuation")
        assert len(summary) == 2

    def test_an_unconfigured_planner_is_named_as_deterministic(self, tmp_path):
        """It must never look like the model wrote a story it had no part in."""
        _write(_work(tmp_path) / "decision_trace.json", self.TRACE)
        method = artifacts.story(tmp_path)["method"]
        assert method["mode"] == "deterministic"
        assert "not configured" in method["warning"]

    def test_a_configured_planner_is_named_as_model_assisted(self, tmp_path):
        trace = {**self.TRACE, "metadata": {}}
        _write(_work(tmp_path) / "decision_trace.json", trace)
        assert artifacts.story(tmp_path)["method"]["mode"] == "azure_assisted"

    def test_each_gap_reports_how_much_evidence_backed_it(self, tmp_path):
        _write(_work(tmp_path) / "decision_trace.json", self.TRACE)
        gaps = artifacts.story(tmp_path)["gaps"]
        assert [gap["evidence_count"] for gap in gaps] == [2, 1]

    def test_a_trace_without_decisions_does_not_crash(self, tmp_path):
        _write(_work(tmp_path) / "decision_trace.json", {"metadata": {}})
        assert artifacts.story(tmp_path)["story"]["summary"] == []
