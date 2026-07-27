"""Narrows a scene report to a single shot, so the camera can be fitted to one camera.

Ground-plane calibration works by pooling the apparent height of every tracked person
and assuming they are all about 1.72 m: the spread of pixel heights against image
position is what reveals the horizon and the camera's own height. That inference is only
valid while there is *one* camera. Pool a close pavement-level shot with a distant view
along a bridge and the median height describes neither, so the fitted ground plane
belongs to no camera in the video and every actor placed on it stands in the wrong place
at the wrong size.

The same holds for camera motion: a verdict measured across takes is a verdict about the
edit, not about any camera.

So each shot gets its own view of the evidence, and each gap is planned against the view
belonging to the take it sits in. Tracks already carry the shot they were seen in, which
is what makes this a filter rather than a re-analysis.
"""

from typing import Sequence


def _clip_range_dicts(
    ranges: Sequence[dict], start_frame: int, end_frame: int,
) -> list[dict]:
    clipped: list[dict] = []
    for item in ranges:
        overlap_start = max(int(item["start"]), start_frame)
        overlap_end = min(int(item["end"]), end_frame)
        if overlap_start <= overlap_end:
            clipped.append({"start": overlap_start, "end": overlap_end})
    return clipped


def scene_report_for_shot(
    scene_report: dict, shot_index: int, shot_bounds: tuple[int, int],
) -> dict:
    """Everything the scene report says about one take, and nothing about the others.

    Entities seen only during a transition carry no shot and are dropped here: during a
    dissolve the picture is a blend of two clips, so a box measured on one describes a
    position in neither. They remain in the full report, which is what the clue timeline
    and the narrative are built from.
    """
    start_frame, end_frame = int(shot_bounds[0]), int(shot_bounds[1])
    tracks = [
        track for track in scene_report.get("tracks", [])
        if track.get("shot_index") == shot_index
    ]
    return {
        **scene_report,
        "tracks": tracks,
        "people": [
            person for person in scene_report.get("people", [])
            if person["id"] in {track["id"] for track in tracks}
        ],
        "vehicles": [
            vehicle for vehicle in scene_report.get("vehicles", [])
            if vehicle["id"] in {track["id"] for track in tracks}
        ],
        "visible_ranges": _clip_range_dicts(
            scene_report.get("visible_ranges", []), start_frame, end_frame,
        ),
        "hidden_ranges": _clip_range_dicts(
            scene_report.get("hidden_ranges", []), start_frame, end_frame,
        ),
        "shot": {"index": shot_index, "start": start_frame, "end": end_frame},
    }


def combine_shot_motion(motion_by_shot: dict[int, dict]) -> dict:
    """One camera-motion verdict for the whole video, from the per-shot ones.

    The stages that read a single report — the clue catalog, the narrative, the frame
    exporter — want to know whether this footage can be treated as coming from a fixed
    camera. That is true only when it is true of every take, so the combined verdict is
    the least favourable one rather than an average that would let one handheld shot hide
    behind eight locked-off ones.
    """
    reports = [report for report in motion_by_shot.values() if report]
    if not reports:
        return {
            "classification": "unclassified",
            "render_transform_available": False,
            "shot_count": 0,
            "pair_reports": [],
        }
    classifications = {report.get("classification") for report in reports}
    if "dynamic_camera" in classifications:
        combined = "dynamic_camera"
    elif classifications == {"static_camera"}:
        combined = "static_camera"
    else:
        combined = "unclassified"

    def _worst(key: str, default: float) -> float:
        values = [
            float(report[key]) for report in reports
            if isinstance(report.get(key), (int, float))
        ]
        return max(values) if values else default

    def _lowest(key: str, default: float) -> float:
        values = [
            float(report[key]) for report in reports
            if isinstance(report.get(key), (int, float))
        ]
        return min(values) if values else default

    return {
        "classification": combined,
        "render_transform_available": False,
        "shot_count": len(reports),
        "sample_count": sum(int(report.get("sample_count", 0)) for report in reports),
        "median_translation_pixels_per_frame": _worst(
            "median_translation_pixels_per_frame", 0.0,
        ),
        "median_rotation_degrees_per_frame": _worst(
            "median_rotation_degrees_per_frame", 0.0,
        ),
        "median_scale_change_per_frame": _worst("median_scale_change_per_frame", 0.0),
        "static_feature_inlier_score": _lowest("static_feature_inlier_score", 0.0),
        "camera_motion_fit_score": _lowest("camera_motion_fit_score", 0.0),
        "per_shot_classification": {
            str(index): report.get("classification")
            for index, report in sorted(motion_by_shot.items())
        },
        "pair_reports": [
            pair for report in reports for pair in report.get("pair_reports", [])
        ],
    }
