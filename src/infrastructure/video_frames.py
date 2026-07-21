from pathlib import Path

import cv2

from domain.cancellation import CancellationCheck, raise_if_cancelled


JPEG_QUALITY = 92


def export_video_frame(
    video_path: Path,
    frame_index: int,
    output_path: Path,
    cancellation_check: CancellationCheck | None = None,
) -> Path:
    raise_if_cancelled(cancellation_check)
    if frame_index < 0:
        raise ValueError("Frame index cannot be negative")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open source video: {video_path.name}")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        read_successfully, frame = capture.read()
    finally:
        capture.release()
    if not read_successfully:
        raise ValueError(f"Cannot read visible evidence frame {frame_index}")
    raise_if_cancelled(cancellation_check)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_successfully = cv2.imwrite(
        str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    )
    if not write_successfully:
        raise OSError(f"Cannot write visible evidence frame to {output_path}")
    return output_path
