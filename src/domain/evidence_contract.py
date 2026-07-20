class EvidenceContractError(ValueError):
    pass


def validate_visible_evidence_only(scene_report: dict) -> None:
    hidden_ranges = _hidden_ranges(scene_report)
    for track in scene_report.get("tracks", []):
        _validate_track_detections(track, hidden_ranges)


def _hidden_ranges(scene_report: dict) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for hidden_range in scene_report.get("hidden_ranges", []):
        start = int(hidden_range["start"])
        end = int(hidden_range["end"])
        if start > end:
            raise EvidenceContractError("Hidden evidence range has an invalid boundary")
        ranges.append((start, end))
    return ranges


def _validate_track_detections(track: dict, hidden_ranges: list[tuple[int, int]]) -> None:
    track_id = str(track.get("id", "unknown"))
    for detection in track.get("detections", []):
        frame_index = int(detection["frame"])
        if any(start <= frame_index <= end for start, end in hidden_ranges):
            raise EvidenceContractError(
                f"Track {track_id} contains forbidden hidden-frame evidence at frame {frame_index}"
            )
