import bisect
import cv2
import math
import numpy as np


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
    while frame_no <= end:
        ret, frame = cap.read()
        if not ret:
            break
        for track in tracks:
            bbox = _bbox_at(track, frame_no, max_gap=max_gap)
            if bbox is None:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            color = _color_for(track["class_name"])
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, box_thickness)
            label = f"{track['class_name']} {track['id'].split('_')[-1]} | {track['direction']}"
            _draw_label(frame, max(0, x1), max(0, y1 - 4), label, color)
        _draw_hud(frame, f"YOLO LIVE CLASSIFICATION - {chunk_label}", f"Frame {frame_no} | visible evidence", opacity=hud_opacity)
        out.write(frame)
        frame_no += 1

    cap.release()
    out.release()


def _ground_point_from_bbox(bbox, width, height):
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    foot_y = y2
    nx = (cx - width / 2.0) / (width / 2.0)
    depth = 1.0 - min(1.0, max(0.0, foot_y / max(1, height)))
    screen_x = int(width / 2 + nx * width * (0.42 + depth * 0.18))
    screen_y = int(height * 0.80 - depth * height * 0.48)
    scale = max(0.45, 1.25 - depth * 0.85)
    return screen_x, screen_y, scale


def _smooth_paths(plan, smoothing):
    if smoothing <= 0:
        return plan.get("entities", [])
    entities = []
    for entity in plan.get("entities", []):
        new_entity = dict(entity)
        smoothed = []
        prev = None
        for point in entity.get("path", []):
            bbox = point["bbox"]
            if prev is None:
                smooth_bbox = bbox
            else:
                smooth_bbox = [
                    int(round(prev[i] * smoothing + bbox[i] * (1.0 - smoothing)))
                    for i in range(4)
                ]
            smoothed.append({"frame": point["frame"], "bbox": smooth_bbox})
            prev = smooth_bbox
        new_entity["path"] = smoothed
        entities.append(new_entity)
    return entities


def _draw_grid(frame):
    h, w = frame.shape[:2]
    frame[:] = (18, 22, 25)
    horizon = int(h * 0.28)
    cv2.rectangle(frame, (0, 0), (w, horizon), (28, 34, 38), -1)
    for i in range(13):
        t = i / 12
        y = int(_lerp(h - 1, horizon, t * t))
        cv2.line(frame, (0, y), (w, y), (40, 70, 72), 1)
    for i in range(-10, 11):
        x_bottom = int(w / 2 + i * w * 0.08)
        cv2.line(frame, (w // 2, horizon), (x_bottom, h), (40, 70, 72), 1)
    cv2.line(frame, (0, horizon), (w, horizon), (0, 180, 200), 1)


def _draw_person_3d(frame, point, scale, color, label):
    x, y = point
    body_h = int(88 * scale)
    body_w = int(26 * scale)
    shadow_w = int(42 * scale)
    cv2.ellipse(frame, (x, y + 5), (shadow_w, max(4, int(9 * scale))), 0, 0, 360, (0, 0, 0), -1)
    cv2.line(frame, (x, y), (x, y - body_h), color, max(2, int(5 * scale)), cv2.LINE_AA)
    cv2.circle(frame, (x, y - body_h - int(13 * scale)), max(5, int(11 * scale)), color, -1, cv2.LINE_AA)
    cv2.line(frame, (x, y - int(body_h * 0.58)), (x - body_w, y - int(body_h * 0.35)), color, 2, cv2.LINE_AA)
    cv2.line(frame, (x, y - int(body_h * 0.58)), (x + body_w, y - int(body_h * 0.35)), color, 2, cv2.LINE_AA)
    cv2.line(frame, (x, y), (x - body_w, y + int(26 * scale)), color, 2, cv2.LINE_AA)
    cv2.line(frame, (x, y), (x + body_w, y + int(26 * scale)), color, 2, cv2.LINE_AA)
    _draw_label(frame, max(0, x - 36), max(18, y - body_h - 34), label, color)


def _draw_vehicle_3d(frame, point, scale, color, label):
    x, y = point
    bw = int(96 * scale)
    bh = int(36 * scale)
    top = np.array([[x - bw // 2, y - bh], [x + bw // 2, y - bh], [x + bw // 2 + 18, y - bh - 18], [x - bw // 2 + 18, y - bh - 18]])
    front = np.array([[x - bw // 2, y - bh], [x + bw // 2, y - bh], [x + bw // 2, y], [x - bw // 2, y]])
    side = np.array([[x + bw // 2, y - bh], [x + bw // 2 + 18, y - bh - 18], [x + bw // 2 + 18, y - 18], [x + bw // 2, y]])
    cv2.fillPoly(frame, [front], color)
    cv2.fillPoly(frame, [top], tuple(min(255, c + 35) for c in color))
    cv2.fillPoly(frame, [side], tuple(max(0, c - 45) for c in color))
    cv2.circle(frame, (x - bw // 3, y + 2), max(5, int(8 * scale)), (20, 20, 20), -1)
    cv2.circle(frame, (x + bw // 3, y + 2), max(5, int(8 * scale)), (20, 20, 20), -1)
    _draw_label(frame, max(0, x - bw // 2), max(18, y - bh - 26), label, color)


def _path_lookup(entity):
    return {point["frame"]: point["bbox"] for point in entity.get("path", [])}


def render_3d_reconstruction(output_path, plan, scene_report, width, height, fps, visual_config=None):
    visual_config = visual_config or {}
    hidden = plan["hidden_range"]
    start, end = hidden["start"], hidden["end"]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    entities = _smooth_paths(plan, visual_config.get("motion_smoothing", 0.55))[:12]
    path_maps = {entity["id"]: _path_lookup(entity) for entity in entities}
    trails = {entity["id"]: [] for entity in entities}
    trail_length = visual_config.get("trail_length", 24)
    speed_boost = visual_config.get("entity_speed_boost", 1.35)
    show_progress_bar = visual_config.get("show_progress_bar", False)
    hud_opacity = visual_config.get("hud_opacity", 0.42)

    total = max(1, end - start)
    for frame_no in range(start, end + 1):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        _draw_grid(frame)
        alpha = (frame_no - start) / total

        for entity in entities:
            bbox = path_maps[entity["id"]].get(frame_no)
            if bbox is None:
                continue
            sx, sy, scale = _ground_point_from_bbox(bbox, width, height)
            center_offset = sx - width // 2
            sx = int(width // 2 + center_offset * speed_boost)
            trails[entity["id"]].append((sx, sy))
            trails[entity["id"]] = trails[entity["id"]][-trail_length:]
            color = _color_for(entity["class_name"])
            for idx in range(1, len(trails[entity["id"]])):
                fade = idx / len(trails[entity["id"]])
                p1, p2 = trails[entity["id"]][idx - 1], trails[entity["id"]][idx]
                cv2.line(frame, p1, p2, tuple(int(c * fade) for c in color), 2, cv2.LINE_AA)
            label = f"{entity['class_name']} {entity['id'].split('_')[-1]}"
            if entity["class_name"] == "person":
                _draw_person_3d(frame, (sx, sy), scale, color, label)
            else:
                _draw_vehicle_3d(frame, (sx, sy), scale, color, label)

        _draw_hud(
            frame,
            "AI 3D RECONSTRUCTION - MISSING EVIDENCE",
            f"Frame {frame_no} | inferred from {scene_report.get('people_count', 0)} people, "
            f"{scene_report.get('vehicle_count', 0)} vehicles, {scene_report.get('carried_object_count', 0)} carried objects",
            opacity=hud_opacity,
        )
        if show_progress_bar:
            bar_w = int(width * 0.34)
            x0, y0 = width - bar_w - 24, height - 32
            cv2.rectangle(frame, (x0, y0), (x0 + bar_w, y0 + 8), (45, 45, 45), -1)
            cv2.rectangle(frame, (x0, y0), (x0 + int(bar_w * alpha), y0 + 8), (0, 220, 255), -1)
        out.write(frame)

    out.release()
