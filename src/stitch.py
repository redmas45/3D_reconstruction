import cv2
import os

from domain.cancellation import CancellationCheck, raise_if_cancelled

def _read_all_frames(video_path: str, width: int, height: int) -> list:
    """Reads an entire video into a list of resized frames (requires enough RAM)."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height))
        frames.append(frame)
    cap.release()
    return frames

def _add_label(frame, label, color, frame_idx):
    """Draws a label overlay on a frame."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w, _ = frame.shape
    border_thickness = 10
    
    # Border
    if label != "Reconstructed Gap":
        cv2.rectangle(frame, (0, 0), (w, h), color, border_thickness)
        
        # Watermark
        font_scale = 1.0
        text = f"{label.upper()} - FRAME {frame_idx}"
        text_size = cv2.getTextSize(text, font, font_scale, 2)[0]
        text_x = w - text_size[0] - 20
        text_y = h - 25
        
        cv2.rectangle(frame, (text_x - 10, text_y - 25), (w - 10, h - 10), (0, 0, 0), -1)
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, 2, cv2.LINE_AA)
        
    return frame


def _write_stream(
    writer: cv2.VideoWriter,
    video_path: str,
    width: int,
    height: int,
    cancellation_check: CancellationCheck | None = None,
) -> int:
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    try:
        while True:
            raise_if_cancelled(cancellation_check)
            ret, frame = cap.read()
            if not ret:
                break
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
            frame_count += 1
    finally:
        cap.release()
    return frame_count

def stitch_videos(
    segment_1_path: str,
    gap_video_path: str,
    segment_2_path: str,
    output_path: str,
    fps: float = 30.0,
    crossfade_frames: int = 0,
    label_visible: bool = False
) -> None:
    """
    Concatenates segment_1, gap_video, and segment_2.
    """
    print(f"[Stitcher] Stitching videos into {output_path}...")
    
    # Check dimensions
    cap = cv2.VideoCapture(segment_1_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if crossfade_frames == 0 and not label_visible:
        print("[Stitcher] Streaming segments without overlays.")
        total_frames = 0
        total_frames += _write_stream(out, segment_1_path, width, height)
        total_frames += _write_stream(out, gap_video_path, width, height)
        total_frames += _write_stream(out, segment_2_path, width, height)
        out.release()
        print(f"[Stitcher] Successfully saved final stitched video ({total_frames} frames) to {output_path}.")
        return
    
    # Read all segments into memory (fine for 1-2 min clips on modern CPUs)
    print("[Stitcher] Loading segments into memory...")
    frames_1 = _read_all_frames(segment_1_path, width, height)
    frames_g = _read_all_frames(gap_video_path, width, height)
    frames_2 = _read_all_frames(segment_2_path, width, height)
    
    # Apply labels to visible segments
    if label_visible:
        print("[Stitcher] Applying overlays...")
        for i in range(len(frames_1)):
            frames_1[i] = _add_label(frames_1[i], "Segment 1 (Before)", (255, 0, 0), i)
        for i in range(len(frames_2)):
            frames_2[i] = _add_label(frames_2[i], "Segment 2 (After)", (255, 0, 0), i + len(frames_1) + len(frames_g))

    crossfade_frames = max(0, min(crossfade_frames, len(frames_1), len(frames_g), len(frames_2)))
    print(f"[Stitcher] Assembling with {crossfade_frames}-frame crossfades...")

    if crossfade_frames == 0:
        for frame in frames_1:
            out.write(frame)
        for frame in frames_g:
            out.write(frame)
        for frame in frames_2:
            out.write(frame)
        out.release()
        total_frames = len(frames_1) + len(frames_g) + len(frames_2)
        print(f"[Stitcher] Successfully saved final stitched video ({total_frames} frames) to {output_path}.")
        return
    
    # Write Segment 1 (up to the crossfade point)
    for i in range(len(frames_1) - crossfade_frames):
        out.write(frames_1[i])
        
    # Crossfade 1 -> Gap
    for i in range(crossfade_frames):
        alpha = i / crossfade_frames
        f1 = frames_1[-(crossfade_frames - i)]
        fg = frames_g[i]
        blended = cv2.addWeighted(fg, alpha, f1, 1 - alpha, 0)
        out.write(blended)
        
    # Write Gap (up to the crossfade point)
    for i in range(crossfade_frames, len(frames_g) - crossfade_frames):
        out.write(frames_g[i])
        
    # Crossfade Gap -> 2
    for i in range(crossfade_frames):
        alpha = i / crossfade_frames
        fg = frames_g[-(crossfade_frames - i)]
        f2 = frames_2[i]
        blended = cv2.addWeighted(f2, alpha, fg, 1 - alpha, 0)
        out.write(blended)
        
    # Write Segment 2
    for i in range(crossfade_frames, len(frames_2)):
        out.write(frames_2[i])
        
    out.release()
    total_frames = len(frames_1) + len(frames_g) + len(frames_2) - (2 * crossfade_frames)
    print(f"[Stitcher] Successfully saved final stitched video ({total_frames} frames) to {output_path}.")


def stitch_sequence(
    video_paths: list[str],
    output_path: str,
    fps: float = 30.0,
    cancellation_check: CancellationCheck | None = None,
) -> None:
    """Streams a list of same-sized videos into one output video."""
    if not video_paths:
        raise ValueError("No video paths provided to stitch_sequence")

    cap = cv2.VideoCapture(video_paths[0])
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_paths[0]}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    total = 0
    try:
        for path in video_paths:
            total += _write_stream(out, path, width, height, cancellation_check)
    finally:
        out.release()
    print(f"[Stitcher] Saved {output_path} ({total} frames).")
