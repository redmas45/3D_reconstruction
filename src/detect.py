import cv2
import os
import json
from ultralytics import YOLO


RELEVANT_COCO_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    24: "backpack",
    26: "handbag",
    28: "suitcase",
    39: "bottle",
    41: "cup",
    43: "knife",
    67: "cell phone",
}


def _scale_frame(frame, downscale_width: int):
    height, width = frame.shape[:2]
    scale = 1.0
    if downscale_width and width > downscale_width:
        scale = downscale_width / width
        frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
    return frame, scale


def detect_scene_objects(
    video_path: str,
    visible_ranges: list,
    model_name: str = "yolo26m.pt",
    class_ids: list = None,
    frame_stride: int = 10,
    downscale_width: int = 960,
    conf: float = 0.25,
) -> list:
    """
    Runs YOLO tracking over selected visible frame ranges and returns structured detections.

    visible_ranges are inclusive global frame ranges: [(start, end), ...].
    Hidden frames should not be included.
    """
    class_ids = class_ids or sorted(RELEVANT_COCO_CLASSES.keys())
    print(f"[Detector] Initializing {model_name} for scene analysis...")
    model = YOLO(model_name)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    detections = []
    processed = 0

    for range_start, range_end in visible_ranges:
        start = max(0, int(range_start))
        end = min(total_frames - 1, int(range_end))
        if end < start:
            continue

        for frame_idx in range(start, end + 1, max(1, frame_stride)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            frame_small, scale = _scale_frame(frame, downscale_width)
            results = model.track(
                frame_small,
                persist=True,
                classes=class_ids,
                conf=conf,
                verbose=False,
            )

            if results and len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    class_id = int(box.cls[0])
                    xyxy = box.xyxy[0].cpu().numpy().tolist()
                    track_id = int(box.id[0]) if box.id is not None else -1
                    detections.append(
                        {
                            "frame": frame_idx,
                            "track_id": track_id,
                            "class_id": class_id,
                            "class_name": model.names.get(class_id, str(class_id)),
                            "confidence": float(box.conf[0]) if box.conf is not None else 0.0,
                            "bbox": [
                                int(xyxy[0] / scale),
                                int(xyxy[1] / scale),
                                int(xyxy[2] / scale),
                                int(xyxy[3] / scale),
                            ],
                        }
                    )

            processed += 1
            if processed % 100 == 0:
                print(f"[Detector] Processed {processed} sampled frames...")

    cap.release()
    print(
        f"[Detector] Finished scene analysis with {model_name}: {len(detections)} detections "
        f"from {processed} sampled frames ({width}x{height})."
    )
    return detections
