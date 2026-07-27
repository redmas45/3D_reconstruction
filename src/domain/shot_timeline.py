"""The shot structure of a video, and what each stage is allowed to look at.

Everything downstream of detection assumes one scene: one background, one camera, one
ground plane. That assumption is what makes the cheap approach work — recover the
background once from the visible frames and reuse it for every gap, because the street
behind a person does not change while they walk across it.

The assumption holds *within a shot*. It does not survive a cut. Applied to a video that
is several clips joined together, a whole-video temporal median averages every camera
position in the file into one image, and the result is a ghosted composite that then
becomes the background of every reconstructed frame. Camera calibration fitted across
the same span is fitted to no camera that ever existed.

So the timeline is modelled explicitly, and three kinds of frame are distinguished:

  * **Shot frames** belong to exactly one shot and may be used for that shot's plate,
    calibration, and gaps.
  * **Transition frames** belong to no shot. During a dissolve the picture is a blend of
    the outgoing and incoming clips, so those frames are ghosted *by construction* —
    feeding them to either side's plate reintroduces the artifact this module exists to
    remove. They are excluded rather than assigned to a neighbour.
  * A video with no cuts is one shot covering every frame, which is the ordinary case
    and costs nothing.

Frame ranges here are inclusive on both ends, matching the `hidden_ranges` and
`visible_ranges` convention used across the pipeline.
"""

from dataclasses import dataclass
from typing import Iterable, Sequence


# A run of frames shorter than this is a detector artifact — a burst of motion blur, a
# camera flash, a crowd swallowing the frame — not a scene anyone would call a shot.
# Merging it into a neighbour would pollute that neighbour's plate, so it is dropped to
# transition instead, which is the honest description of a stretch that registers
# against nothing.
MINIMUM_SHOT_FRAMES = 24


class ShotTimelineError(ValueError):
    """The shot structure does not describe the video it claims to."""


@dataclass(frozen=True)
class Transition:
    """Frames between two shots that belong to neither."""

    start_frame: int
    end_frame: int

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame + 1

    def contains(self, frame_index: int) -> bool:
        return self.start_frame <= frame_index <= self.end_frame


@dataclass(frozen=True)
class Shot:
    """One continuous take: a single camera, a single background."""

    index: int
    start_frame: int
    end_frame: int

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame + 1

    def contains(self, frame_index: int) -> bool:
        return self.start_frame <= frame_index <= self.end_frame

    def seconds(self, fps: float) -> float:
        return self.frame_count / float(fps) if fps else 0.0

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "start": self.start_frame,
            "end": self.end_frame,
            "frame_count": self.frame_count,
        }


def shots_from_transitions(
    transitions: Sequence[Transition], frame_count: int,
) -> tuple[Shot, ...]:
    """The shots are whatever the transitions leave behind.

    Deriving one from the other rather than reporting both from the detector means the
    two can never disagree about which frames are covered.
    """
    if frame_count <= 0:
        raise ShotTimelineError(f"A video cannot have {frame_count} frames")
    ordered = sorted(transitions, key=lambda item: item.start_frame)
    for earlier, later in zip(ordered, ordered[1:]):
        if later.start_frame <= earlier.end_frame:
            raise ShotTimelineError(
                f"Transitions {earlier} and {later} overlap; each frame belongs to at "
                f"most one transition"
            )
    # The minimum length exists to discard slivers *created by cuts*. A video with no
    # cuts is one shot whatever its length — a short clip is the caller's to judge, and
    # filtering it here would leave a perfectly ordinary video with no scene at all.
    minimum = MINIMUM_SHOT_FRAMES if ordered else 1
    shots: list[Shot] = []
    cursor = 0
    for transition in ordered:
        _add_shot(shots, cursor, min(transition.start_frame - 1, frame_count - 1), minimum)
        cursor = transition.end_frame + 1
    _add_shot(shots, cursor, frame_count - 1, minimum)
    if not shots:
        raise ShotTimelineError(
            "Every frame was classified as a transition, which leaves no scene to "
            "reconstruct"
        )
    return tuple(shots)


def _add_shot(
    shots: list[Shot], start_frame: int, end_frame: int, minimum_frames: int,
) -> None:
    """Append a shot, discarding runs too short to be a scene."""
    if end_frame - start_frame + 1 < minimum_frames:
        return
    shots.append(Shot(index=len(shots), start_frame=start_frame, end_frame=end_frame))


def shot_containing(shots: Sequence[Shot], frame_index: int) -> Shot | None:
    """The shot a frame belongs to, or None when it falls in a transition."""
    for shot in shots:
        if shot.contains(frame_index):
            return shot
    return None


def shot_spanning(shots: Sequence[Shot], start_frame: int, end_frame: int) -> Shot | None:
    """The single shot wholly containing a range, or None if it straddles a boundary.

    A gap that straddles a cut has no single background to reconstruct against, so the
    caller must either move it or split it rather than pick one side.
    """
    for shot in shots:
        if shot.contains(start_frame) and shot.contains(end_frame):
            return shot
    return None


def clip_ranges_to_shot(
    ranges: Iterable[tuple[int, int]], shot: Shot,
) -> list[tuple[int, int]]:
    """The parts of `ranges` that lie inside one shot.

    Used to answer "which visible frames may this shot's plate be built from" without
    the plate builder needing to know anything about the timeline.
    """
    clipped: list[tuple[int, int]] = []
    for start_frame, end_frame in ranges:
        overlap_start = max(int(start_frame), shot.start_frame)
        overlap_end = min(int(end_frame), shot.end_frame)
        if overlap_start <= overlap_end:
            clipped.append((overlap_start, overlap_end))
    return clipped


def split_range_at_shots(
    start_frame: int, end_frame: int, shots: Sequence[Shot],
) -> list[tuple[int, int, Shot]]:
    """Break a frame range into per-shot pieces, dropping transition frames.

    A track that continues across a cut is two different things that happened to be
    given one identity, so callers that need per-shot evidence split on this rather than
    interpolating a trajectory through a scene change.
    """
    pieces: list[tuple[int, int, Shot]] = []
    for shot in shots:
        overlap_start = max(int(start_frame), shot.start_frame)
        overlap_end = min(int(end_frame), shot.end_frame)
        if overlap_start <= overlap_end:
            pieces.append((overlap_start, overlap_end, shot))
    return pieces


def single_shot_timeline(frame_count: int) -> tuple[Shot, ...]:
    """The timeline of a video with no cuts — the ordinary case."""
    return shots_from_transitions((), frame_count)


def coverage_report(shots: Sequence[Shot], frame_count: int) -> dict:
    """How much of the video is usable, for the run report.

    A file that is 30% transitions is telling the operator something real about why the
    reconstruction has less evidence than the frame count suggests.
    """
    covered = sum(shot.frame_count for shot in shots)
    longest = max(shots, key=lambda shot: shot.frame_count) if shots else None
    return {
        "shot_count": len(shots),
        "frame_count": int(frame_count),
        "frames_in_shots": covered,
        "frames_in_transitions": int(frame_count) - covered,
        "shot_coverage": round(covered / float(frame_count), 6) if frame_count else 0.0,
        "longest_shot_frames": longest.frame_count if longest else 0,
        "shots": [shot.as_dict() for shot in shots],
    }


def validate_shots(shots: Sequence[Shot], frame_count: int) -> None:
    """Shots must be ordered, disjoint, and inside the video."""
    if not shots:
        raise ShotTimelineError("A timeline must contain at least one shot")
    for position, shot in enumerate(shots):
        if shot.index != position:
            raise ShotTimelineError(
                f"Shot at position {position} is indexed {shot.index}"
            )
        if shot.start_frame > shot.end_frame:
            raise ShotTimelineError(f"Shot {shot.index} ends before it starts")
        if shot.start_frame < 0 or shot.end_frame >= frame_count:
            raise ShotTimelineError(
                f"Shot {shot.index} covers frames {shot.start_frame}-{shot.end_frame}, "
                f"outside a video of {frame_count} frames"
            )
    for earlier, later in zip(shots, shots[1:]):
        if later.start_frame <= earlier.end_frame:
            raise ShotTimelineError(
                f"Shots {earlier.index} and {later.index} overlap"
            )
