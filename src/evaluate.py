import cv2
import numpy as np
from ultralytics import YOLO


def _ssim(first: np.ndarray, second: np.ndarray) -> float:
    first_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY).astype(np.float64)
    second_gray = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY).astype(np.float64)
    constant_one = (0.01 * 255) ** 2
    constant_two = (0.03 * 255) ** 2
    first_mean = cv2.GaussianBlur(first_gray, (11, 11), 1.5)
    second_mean = cv2.GaussianBlur(second_gray, (11, 11), 1.5)
    first_variance = cv2.GaussianBlur(first_gray * first_gray, (11, 11), 1.5) - first_mean**2
    second_variance = cv2.GaussianBlur(second_gray * second_gray, (11, 11), 1.5) - second_mean**2
    covariance = cv2.GaussianBlur(first_gray * second_gray, (11, 11), 1.5) - first_mean * second_mean
    numerator = (2 * first_mean * second_mean + constant_one) * (2 * covariance + constant_two)
    denominator = (first_mean**2 + second_mean**2 + constant_one) * (
        first_variance + second_variance + constant_two
    )
    return float(np.mean(numerator / np.maximum(denominator, 1e-9)))


def _frame_similarity(truth_path: str, reconstruction_path: str) -> dict:
    truth_capture = cv2.VideoCapture(truth_path)
    reconstruction_capture = cv2.VideoCapture(reconstruction_path)
    ssim_values: list[float] = []
    psnr_values: list[float] = []
    while True:
        truth_ok, truth_frame = truth_capture.read()
        reconstruction_ok, reconstruction_frame = reconstruction_capture.read()
        if not truth_ok or not reconstruction_ok:
            break
        if truth_frame.shape != reconstruction_frame.shape:
            reconstruction_frame = cv2.resize(reconstruction_frame, (truth_frame.shape[1], truth_frame.shape[0]))
        ssim_values.append(_ssim(truth_frame, reconstruction_frame))
        psnr_values.append(float(cv2.PSNR(truth_frame, reconstruction_frame)))
    truth_capture.release()
    reconstruction_capture.release()
    return {
        "frames_compared": len(ssim_values),
        "mean_ssim": round(float(np.mean(ssim_values)), 4) if ssim_values else None,
        "mean_psnr_db": round(float(np.mean(psnr_values)), 3) if psnr_values else None,
    }


def _read_frame(video_path: str, frame_number: int) -> np.ndarray:
    capture = cv2.VideoCapture(video_path)
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_number))
    success, frame = capture.read()
    capture.release()
    if not success:
        raise ValueError(f"Could not read frame {frame_number} from {video_path}")
    return frame


def _first_and_last(video_path: str) -> tuple[np.ndarray, np.ndarray]:
    capture = cv2.VideoCapture(video_path)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    success, first = capture.read()
    if not success:
        capture.release()
        raise ValueError(f"Could not read reconstruction: {video_path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_count - 1))
    success, last = capture.read()
    capture.release()
    if not success:
        raise ValueError(f"Could not read final reconstruction frame: {video_path}")
    return first, last


def _boundary_similarity(source_path: str, reconstruction_path: str, hidden_range: tuple[int, int]) -> dict:
    hidden_start, hidden_end = hidden_range
    before = _read_frame(source_path, hidden_start - 1)
    after = _read_frame(source_path, hidden_end + 1)
    first, last = _first_and_last(reconstruction_path)
    before_score = _ssim(before, first)
    after_score = _ssim(after, last)
    return {
        "entry_ssim": round(before_score, 4),
        "exit_ssim": round(after_score, 4),
        "mean_boundary_ssim": round((before_score + after_score) / 2.0, 4),
    }


def _result_objects(result: object, class_names: dict[int, str]) -> dict[str, list[tuple[float, float]]]:
    objects: dict[str, list[tuple[float, float]]] = {}
    for box in result.boxes:
        class_id = int(box.cls[0])
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
        objects.setdefault(class_names.get(class_id, str(class_id)), []).append(((x1 + x2) / 2.0, (y1 + y2) / 2.0))
    return objects


def _count_similarity(truth: dict[str, list], reconstruction: dict[str, list]) -> float:
    classes = set(truth) | set(reconstruction)
    if not classes:
        return 1.0
    scores = []
    for class_name in classes:
        truth_count = len(truth.get(class_name, []))
        reconstruction_count = len(reconstruction.get(class_name, []))
        scores.append(1.0 - abs(truth_count - reconstruction_count) / max(1, truth_count, reconstruction_count))
    return float(np.mean(scores))


def _center_error(truth: dict[str, list], reconstruction: dict[str, list], diagonal: float) -> float:
    errors: list[float] = []
    for class_name, truth_centers in truth.items():
        remaining = list(reconstruction.get(class_name, []))
        for truth_center in truth_centers:
            if not remaining:
                errors.append(1.0)
                continue
            distances = [np.hypot(truth_center[0] - item[0], truth_center[1] - item[1]) for item in remaining]
            best_index = int(np.argmin(distances))
            errors.append(min(1.0, float(distances[best_index]) / diagonal))
            remaining.pop(best_index)
        errors.extend([1.0] * len(remaining))
    return float(np.mean(errors)) if errors else (0.0 if not reconstruction else 1.0)


def _detection_consistency(
    truth_path: str,
    reconstruction_path: str,
    model: YOLO,
    class_ids: list[int],
    confidence: float,
    frame_stride: int,
) -> dict:
    truth_capture = cv2.VideoCapture(truth_path)
    reconstruction_capture = cv2.VideoCapture(reconstruction_path)
    count_scores: list[float] = []
    person_scores: list[float] = []
    center_errors: list[float] = []
    frame_index = 0
    while True:
        truth_ok, truth_frame = truth_capture.read()
        reconstruction_ok, reconstruction_frame = reconstruction_capture.read()
        if not truth_ok or not reconstruction_ok:
            break
        if frame_index % max(1, frame_stride) == 0:
            results = model.predict([truth_frame, reconstruction_frame], classes=class_ids, conf=confidence, verbose=False)
            truth_objects = _result_objects(results[0], model.names)
            reconstructed_objects = _result_objects(results[1], model.names)
            count_scores.append(_count_similarity(truth_objects, reconstructed_objects))
            person_scores.append(_count_similarity({"person": truth_objects.get("person", [])}, {"person": reconstructed_objects.get("person", [])}))
            diagonal = float(np.hypot(truth_frame.shape[1], truth_frame.shape[0]))
            center_errors.append(_center_error(truth_objects, reconstructed_objects, diagonal))
        frame_index += 1
    truth_capture.release()
    reconstruction_capture.release()
    return {
        "sampled_frames": len(count_scores),
        "mean_object_count_similarity": round(float(np.mean(count_scores)), 4) if count_scores else None,
        "mean_person_count_similarity": round(float(np.mean(person_scores)), 4) if person_scores else None,
        "mean_normalized_center_error": round(float(np.mean(center_errors)), 4) if center_errors else None,
    }


def _evaluate_gap(
    item: dict,
    source_path: str,
    model: YOLO,
    class_ids: list[int],
    confidence: float,
    frame_stride: int,
) -> dict:
    return {
        "gap_index": item["gap_index"],
        "hidden_range": list(item["hidden_range"]),
        "frame_similarity": _frame_similarity(item["truth_path"], item["reconstruction_path"]),
        "boundary_continuity": _boundary_similarity(source_path, item["reconstruction_path"], item["hidden_range"]),
        "detection_consistency": _detection_consistency(
            item["truth_path"],
            item["reconstruction_path"],
            model,
            class_ids,
            confidence,
            max(1, frame_stride),
        ),
    }


def _mean_metric(reports: list[dict], section: str, metric: str) -> float | None:
    values = [report[section].get(metric) for report in reports]
    values = [float(value) for value in values if value is not None]
    return round(float(np.mean(values)), 4) if values else None


def evaluate_reconstructions(
    items: list[dict],
    source_path: str,
    model_name: str,
    class_ids: list[int],
    confidence: float,
    frame_stride: int,
) -> dict:
    model = YOLO(model_name)
    reports = [
        _evaluate_gap(item, source_path, model, class_ids, confidence, frame_stride)
        for item in items
    ]
    return {
        "mode": "post_reconstruction_hidden_truth_evaluation",
        "ground_truth_usage": "Hidden frames were read only after every reconstruction completed.",
        "gap_count": len(reports),
        "summary": {
            "mean_ssim": _mean_metric(reports, "frame_similarity", "mean_ssim"),
            "mean_psnr_db": _mean_metric(reports, "frame_similarity", "mean_psnr_db"),
            "mean_boundary_ssim": _mean_metric(reports, "boundary_continuity", "mean_boundary_ssim"),
            "mean_object_count_similarity": _mean_metric(reports, "detection_consistency", "mean_object_count_similarity"),
            "mean_person_count_similarity": _mean_metric(reports, "detection_consistency", "mean_person_count_similarity"),
            "mean_normalized_center_error": _mean_metric(reports, "detection_consistency", "mean_normalized_center_error"),
        },
        "gaps": reports,
    }
