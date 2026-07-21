import bisect
import cv2

from domain.cancellation import CancellationCheck, raise_if_cancelled


CLASS_COLORS = {
    "person": (0, 220, 255),
    "car": (80, 220, 80),
    "truck": (80, 180, 255),
    "bus": (255, 170, 60),
    "motorcycle": (255, 80, 180),
    "bicycle": (255, 80, 180),
    "backpack": (180, 120, 255),
    "handbag": (180, 120, 255),
    "suitcase": (180, 120, 255),
}


def _color_for(class_name):
    return CLASS_COLORS.get(class_name, (220, 220, 220))


def _lerp(a, b, t):
    return a * (1.0 - t) + b * t


def _lerp_bbox(a, b, t):
    return [int(round(_lerp(a[i], b[i], t))) for i in range(4)]


def _track_lookup(track):
    detections = sorted(track.get("detections", []), key=lambda item: item["frame"])
    return [item["frame"] for item in detections], detections


def _bbox_at(track, frame_no, max_gap):
    frames, detections = _track_lookup(track)
    if not frames:
        return None
    pos = bisect.bisect_left(frames, frame_no)
    if pos < len(frames) and frames[pos] == frame_no:
        return detections[pos]["bbox"]
    before = detections[pos - 1] if pos > 0 else None
    after = detections[pos] if pos < len(detections) else None
    if before and after and after["frame"] - before["frame"] <= max_gap:
        t = (frame_no - before["frame"]) / max(1, after["frame"] - before["frame"])
        return _lerp_bbox(before["bbox"], after["bbox"], t)
    if before and frame_no - before["frame"] <= max_gap:
        return before["bbox"]
    if after and after["frame"] - frame_no <= max_gap:
        return after["bbox"]
    return None


def _draw_label(frame, x, y, text, color):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thickness = 1
    size = cv2.getTextSize(text, font, scale, thickness)[0]
    y = max(size[1] + 6, y)
    cv2.rectangle(frame, (x, y - size[1] - 6), (x + size[0] + 8, y + 3), color, -1)
    cv2.putText(frame, text, (x + 4, y - 3), font, scale, (10, 10, 10), thickness, cv2.LINE_AA)


def _draw_hud(frame, title, subtitle, opacity=0.42):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 58), (0, 0, 0), -1)
    cv2.addWeighted(overlay, opacity, frame, 1.0 - opacity, 0, dst=frame)
    cv2.putText(frame, title, (18, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, subtitle, (18, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (230, 230, 230), 1, cv2.LINE_AA)


def render_annotated_visible_chunk(
    video_path,
    output_path,
    frame_range,
    scene_report,
    chunk_label,
    fps,
    max_gap=25,
    visual_config=None,
    cancellation_check: CancellationCheck | None = None,
):
    visual_config = visual_config or {}
    start, end = frame_range
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not out.isOpened():
        cap.release()
        raise OSError(f"Cannot create visible evidence segment: {output_path}")

    tracks = [
        track for track in scene_report.get("tracks", [])
        if track.get("frames_seen", 0) >= 3 and track.get("class_name") in CLASS_COLORS
    ]
    tracks.sort(key=lambda item: (item["class_name"] != "person", -item["frames_seen"], -item["avg_area"]))
    tracks = tracks[: visual_config.get("visible_max_tracks", 28)]
    box_thickness = visual_config.get("visible_box_thickness", 2)
    hud_opacity = visual_config.get("hud_opacity", 0.42)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frame_no = start
    try:
        while frame_no <= end:
            raise_if_cancelled(cancellation_check)
            ret, frame = cap.read()
            if not ret:
                raise ValueError(f"Video decoding stopped at visible frame {frame_no}")
            for track in tracks:
                bbox = _bbox_at(track, frame_no, max_gap=max_gap)
                if bbox is None:
                    continue
                x1, y1, x2, y2 = [int(v) for v in bbox]
                color = _color_for(track["class_name"])
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)
                label = f"{track['class_name']} {track['id'].split('_')[-1]} | {track['direction']}"
                _draw_label(frame, max(0, x1), max(0, y1 - 4), label, color)
            _draw_hud(
                frame,
                f"YOLO LIVE CLASSIFICATION - {chunk_label}",
                f"Frame {frame_no} | visible evidence",
                opacity=hud_opacity,
            )
            out.write(frame)
            frame_no += 1
    finally:
        cap.release()
        out.release()
