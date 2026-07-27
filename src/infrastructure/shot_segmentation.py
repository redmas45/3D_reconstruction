"""Finds the cuts in a video, so every later stage can work inside one scene.

The measure is background feature correspondence, not appearance. Appearance measures do
not survive contact with this problem: a colour histogram cannot separate one grey
London street from another, and a dissolve never produces a single large frame-to-frame
jump for a threshold to catch. What a crowd walking through a fixed frame does *not* do
is destroy the correspondence between static background features — the shopfronts and
the paving stay exactly where they were. A cut destroys it completely.

So two frames a short baseline apart are matched with ORB and fitted with RANSAC, and
the number of surviving inliers is the signal. Within a shot it stays high whatever the
crowd is doing; across a cut it collapses.

**One thing also collapses it: someone walking straight past the lens.** When a person
fills most of the frame there is almost no background left to match, which looks
identical to a cut. That is not hypothetical — it is the only candidate the detector
finds in a video that never cuts at all. So every candidate is verified in a second
pass by registering the frame *before* it against the frame *after* it, skipping the
disputed stretch entirely. If those two agree, the camera never moved and the candidate
was an occlusion. If they do not, it was a cut.

Reports rather than raises when it cannot discriminate — footage with too little static
texture to match yields an honest "one shot, low confidence" instead of a timeline
invented from noise.
"""

from pathlib import Path

import cv2
import numpy

from domain.cancellation import CancellationCheck, raise_if_cancelled
from domain.shot_timeline import (
    MINIMUM_SHOT_FRAMES,
    Transition,
    coverage_report,
    shots_from_transitions,
)


ANALYSIS_WIDTH = 480
# Every fourth frame is enough: the shortest transition worth finding is a dissolve, and
# those run for a dozen frames or more.
SAMPLE_STRIDE = 4
# How far apart the two compared frames sit. Long enough that a cut is unambiguous,
# short enough that a slow pan does not look like one.
COMPARISON_BASELINE_FRAMES = 12
ORB_FEATURES = 800
ORB_FAST_THRESHOLD = 12
RANSAC_REPROJECTION_PIXELS = 3.0
MINIMUM_MATCHES = 8

# A sample is a candidate when its inlier count falls to a quarter of what the video
# normally sustains. Relative rather than absolute because a richly textured street and
# a bare corridor have very different baselines.
CANDIDATE_INLIER_FRACTION = 0.25
CANDIDATE_INLIER_FLOOR = 12
# Below this the footage has too little static texture for correspondence to mean
# anything, and any timeline derived from it would be noise.
DISCRIMINATION_FLOOR_INLIERS = 40

# How far outside a candidate run the verification frames are taken. Larger than the
# longest dissolve observed, so the verification pair sits in clean footage on each side.
VERIFICATION_MARGIN_FRAMES = 24
# Candidate runs closer together than this are one transition sampled twice.
CANDIDATE_MERGE_FRAMES = 16


class ShotDetectionError(RuntimeError):
    """The video could not be read for shot detection."""


def detect_shots(
    video_path: Path,
    frame_count: int,
    cancellation_check: CancellationCheck | None = None,
) -> dict:
    """Segment a video into shots. Always returns a usable timeline."""
    samples = _scan(video_path, cancellation_check)
    if len(samples) < 4:
        return _single_shot_report(frame_count, "too_short_to_analyse", samples)
    counts = numpy.array([count for _, count in samples], dtype=float)
    typical = float(numpy.median(counts))
    if typical < DISCRIMINATION_FLOOR_INLIERS:
        return _single_shot_report(frame_count, "insufficient_static_texture", samples)

    threshold = max(CANDIDATE_INLIER_FLOOR, typical * CANDIDATE_INLIER_FRACTION)
    candidates = _candidate_runs([frame for frame, count in samples if count < threshold])
    confirmed, rejected = _verify_candidates(
        video_path, candidates, frame_count, threshold, cancellation_check,
    )
    transitions = tuple(
        Transition(start_frame=start, end_frame=end) for start, end in confirmed
    )
    shots = shots_from_transitions(transitions, frame_count)
    return {
        "method": "orb_background_correspondence",
        "reason": "ok",
        "median_inliers": round(typical, 2),
        "candidate_threshold": round(threshold, 2),
        "sample_count": len(samples),
        "candidates_found": len(candidates),
        "candidates_rejected_as_occlusion": rejected,
        "transitions": [
            {"start": transition.start_frame, "end": transition.end_frame}
            for transition in transitions
        ],
        **coverage_report(shots, frame_count),
    }


def shots_from_report(report: dict, frame_count: int):
    """Rebuild the shot objects from a persisted report."""
    transitions = tuple(
        Transition(start_frame=int(item["start"]), end_frame=int(item["end"]))
        for item in report.get("transitions", [])
    )
    return shots_from_transitions(transitions, frame_count)


# --------------------------------------------------------------------------
# Pass one: stream the video and measure correspondence
# --------------------------------------------------------------------------

def _scan(
    video_path: Path, cancellation_check: CancellationCheck | None,
) -> list[tuple[int, int]]:
    """Inlier count for every sampled frame against one a baseline earlier.

    Streams sequentially and keeps only a short ring of descriptors, so cost is one
    decode of the video and memory does not grow with its length.
    """
    lookback = max(1, round(COMPARISON_BASELINE_FRAMES / SAMPLE_STRIDE))
    detector = cv2.ORB_create(nfeatures=ORB_FEATURES, fastThreshold=ORB_FAST_THRESHOLD)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ShotDetectionError(f"Cannot read {Path(video_path).name} for shot detection")
    ring: list[tuple] = []
    samples: list[tuple[int, int]] = []
    frame_index = 0
    try:
        while True:
            read_successfully, frame = capture.read()
            if not read_successfully:
                break
            if frame_index % SAMPLE_STRIDE == 0:
                raise_if_cancelled(cancellation_check)
                ring.append(detector.detectAndCompute(_analysis_frame(frame), None))
                if len(ring) > lookback:
                    earlier = ring.pop(0)
                    samples.append((frame_index, _inliers(matcher, earlier, ring[-1])))
            frame_index += 1
    finally:
        capture.release()
    return samples


def _analysis_frame(frame: numpy.ndarray) -> numpy.ndarray:
    height, width = frame.shape[:2]
    scale = ANALYSIS_WIDTH / float(width)
    resized = cv2.resize(frame, (ANALYSIS_WIDTH, max(1, round(height * scale))))
    return cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)


def _inliers(matcher, first, second) -> int:
    """How many matches survive a rigid fit between two feature sets."""
    first_keypoints, first_descriptors = first
    second_keypoints, second_descriptors = second
    if first_descriptors is None or second_descriptors is None:
        return 0
    matches = matcher.match(first_descriptors, second_descriptors)
    if len(matches) < MINIMUM_MATCHES:
        return 0
    source = numpy.float32([first_keypoints[match.queryIdx].pt for match in matches])
    target = numpy.float32([second_keypoints[match.trainIdx].pt for match in matches])
    _, inliers = cv2.estimateAffinePartial2D(
        source, target, method=cv2.RANSAC, ransacReprojThreshold=RANSAC_REPROJECTION_PIXELS,
    )
    return int(inliers.sum()) if inliers is not None else 0


# --------------------------------------------------------------------------
# Pass two: separate cuts from occlusions
# --------------------------------------------------------------------------

def _candidate_runs(low_frames: list[int]) -> list[tuple[int, int]]:
    """Group neighbouring low-correspondence samples into one candidate each."""
    runs: list[tuple[int, int]] = []
    for frame_index in sorted(low_frames):
        if runs and frame_index - runs[-1][1] <= CANDIDATE_MERGE_FRAMES:
            runs[-1] = (runs[-1][0], frame_index)
        else:
            runs.append((frame_index, frame_index))
    # A sample at frame f compared f-baseline against f, so the disputed stretch starts
    # a baseline before the first low sample.
    return [
        (max(0, start - COMPARISON_BASELINE_FRAMES), end) for start, end in runs
    ]


def _verify_candidates(
    video_path: Path,
    candidates: list[tuple[int, int]],
    frame_count: int,
    threshold: float,
    cancellation_check: CancellationCheck | None,
) -> tuple[list[tuple[int, int]], int]:
    """Keep only candidates whose two sides genuinely fail to register.

    This is what separates a cut from a pedestrian filling the lens. The frames either
    side of the disputed stretch are compared directly with each other: if the camera
    never moved they still match, however completely the middle was blocked.
    """
    if not candidates:
        return [], 0
    detector = cv2.ORB_create(nfeatures=ORB_FEATURES, fastThreshold=ORB_FAST_THRESHOLD)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ShotDetectionError(f"Cannot re-read {Path(video_path).name} to verify cuts")
    confirmed: list[tuple[int, int]] = []
    rejected = 0
    try:
        for start_frame, end_frame in candidates:
            raise_if_cancelled(cancellation_check)
            before = _read_frame(capture, max(0, start_frame - VERIFICATION_MARGIN_FRAMES))
            after = _read_frame(
                capture, min(frame_count - 1, end_frame + VERIFICATION_MARGIN_FRAMES),
            )
            if before is None or after is None:
                # Unverifiable at the very start or end of the file. Treating it as a cut
                # would split off a fragment; leaving it merges two takes at worst.
                rejected += 1
                continue
            agreement = _inliers(
                matcher,
                detector.detectAndCompute(_analysis_frame(before), None),
                detector.detectAndCompute(_analysis_frame(after), None),
            )
            if agreement >= threshold:
                rejected += 1
                continue
            confirmed.append((start_frame, end_frame))
    finally:
        capture.release()
    return _drop_unusable_fragments(confirmed, frame_count), rejected


def _drop_unusable_fragments(
    transitions: list[tuple[int, int]], frame_count: int,
) -> list[tuple[int, int]]:
    """Absorb shots too short to be usable into the transition beside them.

    A three-frame sliver between two cuts cannot support a plate or a calibration, and
    leaving it as a shot invites every later stage to try. `shots_from_transitions`
    already discards such runs; extending the transition keeps the two descriptions of
    the same footage in agreement.
    """
    if not transitions:
        return transitions
    merged = [list(transitions[0])]
    for start_frame, end_frame in transitions[1:]:
        if start_frame - merged[-1][1] - 1 < MINIMUM_SHOT_FRAMES:
            merged[-1][1] = end_frame
        else:
            merged.append([start_frame, end_frame])
    return [(start, end) for start, end in merged if end < frame_count]


def _read_frame(capture: cv2.VideoCapture, frame_index: int) -> numpy.ndarray | None:
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    read_successfully, frame = capture.read()
    return frame if read_successfully else None


def _single_shot_report(frame_count: int, reason: str, samples: list) -> dict:
    """One shot covering everything — the answer for a video with no detectable cuts."""
    shots = shots_from_transitions((), frame_count)
    return {
        "method": "orb_background_correspondence",
        "reason": reason,
        "median_inliers": (
            round(float(numpy.median([count for _, count in samples])), 2) if samples else 0.0
        ),
        "candidate_threshold": None,
        "sample_count": len(samples),
        "candidates_found": 0,
        "candidates_rejected_as_occlusion": 0,
        "transitions": [],
        **coverage_report(shots, frame_count),
    }
