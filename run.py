import argparse
import json
import random
import shutil
import sys
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from detect import RELEVANT_COCO_CLASSES, detect_scene_objects
from gap_selector import choose_hidden_chunk
from reconstruction_plan import build_reconstruction_plan
from scene_intelligence import summarize_scene
from stitch import stitch_sequence
from visual_output import render_3d_reconstruction, render_annotated_visible_chunk


VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}
DEFAULT_CONFIG_PATH = ROOT / "config" / "reconstruction_config.json"


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def yolo_class_ids(config: dict) -> list[int]:
    classes = config.get("yolo", {}).get("classes", {})
    if not classes:
        return sorted(RELEVANT_COCO_CLASSES.keys())
    return sorted(int(class_id) for class_id in classes.keys())


def normalize_confidence(value: float) -> float:
    value = float(value)
    if value > 1.0:
        return max(0.0, min(1.0, value / 100.0))
    return max(0.0, min(1.0, value))


def video_info(video_path: Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    info = {
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(cap.get(cv2.CAP_PROP_FPS) or 30.0),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    if info["frames"] < 4:
        raise ValueError(f"Video is too short: {video_path}")
    return info


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_video_range(video_path: Path, start_frame: int, end_frame: int, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, start_frame))
    frame_no = start_frame
    while frame_no <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)
        frame_no += 1

    cap.release()
    out.release()
    return output_path


def split_into_chunks(video_path: Path, chunks: list, chunk_dir: Path) -> list[Path]:
    paths = []
    for idx, (start, end) in enumerate(chunks):
        output = chunk_dir / f"chunk_{idx}.mp4"
        write_video_range(video_path, start, end, output)
        paths.append(output)
    return paths


def clean_outputs(output_dir: Path) -> None:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def process_video(video_path: Path, args, rng: random.Random) -> Path:
    config = args.config_data
    yolo_config = config.get("yolo", {})
    scene_config = config.get("scene", {})
    visualization_config = config.get("visualization", {})
    info = video_info(video_path)
    output_root = Path(args.output_dir).resolve()
    work_dir = output_root / "_work" / video_path.stem
    gap_selection_path = work_dir / "gap_selection.json"
    if args.reuse_work and gap_selection_path.exists():
        selection = json.load(gap_selection_path.open("r", encoding="utf-8"))
        selection["chunks"] = [tuple(item) for item in selection["chunks"]]
        selection["hidden_range"] = tuple(selection["hidden_range"])
        selection["visible_ranges"] = [tuple(item) for item in selection["visible_ranges"]]
        print(f"[Run] Reusing existing gap selection for {video_path.name}.")
    else:
        selection = choose_hidden_chunk(info["frames"], rng)
    hidden_start, hidden_end = selection["hidden_range"]
    chunk_dir = work_dir / "chunks"

    print(
        f"\n[Run] {video_path.name}: {info['frames']} frames, {info['fps']:.2f} fps, "
        f"hidden chunk {selection['hidden_index'] + 1}/4 = frames {hidden_start}-{hidden_end}"
    )

    expected_chunks = [chunk_dir / f"chunk_{idx}.mp4" for idx in range(4)]
    if args.reuse_work and all(path.exists() for path in expected_chunks):
        chunk_paths = expected_chunks
        print("[Run] Reusing existing chunk files.")
    else:
        chunk_paths = split_into_chunks(video_path, selection["chunks"], chunk_dir)
    hidden_truth_path = chunk_paths[selection["hidden_index"]]
    write_json(work_dir / "gap_selection.json", selection)

    detections_path = work_dir / "detections.json"
    if args.reuse_work and detections_path.exists():
        detections = json.load(detections_path.open("r", encoding="utf-8"))
        print(f"[Run] Reusing {len(detections)} cached detections.")
    else:
        print("[Run] Detecting people, vehicles, bags, and carried objects in visible chunks...")
        detections = detect_scene_objects(
            video_path=str(video_path),
            visible_ranges=selection["visible_ranges"],
            model_name=yolo_config.get("model", "yolo26m.pt"),
            class_ids=yolo_class_ids(config),
            frame_stride=yolo_config.get("frame_stride", 10),
            downscale_width=yolo_config.get("downscale_width", 960),
            conf=normalize_confidence(yolo_config.get("confidence", 0.25)),
        )
        write_json(detections_path, detections)

    print("[Run] Building scene intelligence report...")
    scene_report = summarize_scene(
        detections=detections,
        fps=info["fps"],
        frame_width=info["width"],
        hidden_range=selection["hidden_range"],
    )
    scene_report["video"] = {
        "path": str(video_path),
        "width": info["width"],
        "height": info["height"],
        "fps": info["fps"],
        "frames": info["frames"],
    }
    scene_report["visible_ranges"] = [
        {"start": start, "end": end}
        for start, end in selection["visible_ranges"]
    ]
    write_json(work_dir / "scene_report.json", scene_report)
    print(
        f"[Run] Scene: people={scene_report['people_count']}, "
        f"vehicles={scene_report['vehicle_count']}, carried_objects={scene_report['carried_object_count']}"
    )

    print("[Run] Planning missing chunk reconstruction...")
    plan = build_reconstruction_plan(
        scene_report,
        selection["hidden_range"],
        info["fps"],
        max_entities=scene_config.get("max_render_entities", 8),
        min_track_frames=scene_config.get("min_track_frames", 3),
    )
    write_json(work_dir / "reconstruction_plan.json", plan)
    print(f"[Run] Planned {len(plan['entities'])} entity path(s) for the hidden chunk.")

    visual_dir = work_dir / "visual_chunks"
    visual_dir.mkdir(parents=True, exist_ok=True)
    sequence = []
    reconstructed_gap = visual_dir / "chunk_hidden_3d_reconstruction.mp4"

    for idx, chunk_path in enumerate(chunk_paths):
        chunk_range = selection["chunks"][idx]
        if idx == selection["hidden_index"]:
            print("[Run] Rendering animated 3D missing-chunk reconstruction...")
            render_3d_reconstruction(
                output_path=str(reconstructed_gap),
                plan=plan,
                scene_report=scene_report,
                width=info["width"],
                height=info["height"],
                fps=info["fps"],
                visual_config=visualization_config,
            )
            sequence.append(str(reconstructed_gap))
        else:
            annotated = visual_dir / f"chunk_{idx}_yolo_annotated.mp4"
            print(f"[Run] Rendering YOLO-classified visible chunk {idx + 1}/4...")
            render_annotated_visible_chunk(
                video_path=str(video_path),
                output_path=str(annotated),
                frame_range=chunk_range,
                scene_report=scene_report,
                chunk_label=f"CHUNK {idx + 1}/4",
                fps=info["fps"],
                max_gap=max(
                    20,
                    yolo_config.get("frame_stride", 10)
                    * scene_config.get("track_interpolation_max_gap_multiplier", 4),
                ),
                visual_config=visualization_config,
            )
            sequence.append(str(annotated))

    report = {
        "mode": "3d_reconstruction_visualization",
        "hidden_chunk": selection["hidden_index"] + 1,
        "hidden_range": selection["hidden_range"],
        "people_tracks": scene_report["people_count"],
        "vehicle_tracks": scene_report["vehicle_count"],
        "carried_object_tracks": scene_report["carried_object_count"],
        "planned_entities": len(plan["entities"]),
        "note": "Visible chunks are YOLO-annotated evidence. Hidden chunk is animated 3D reconstruction, so pixel SSIM is not a meaningful quality metric.",
    }
    write_json(work_dir / "accuracy_report.json", report)

    final_output = output_root / f"{video_path.stem}_reconstructed.mp4"
    stitch_sequence(sequence, str(final_output), fps=info["fps"])
    print(f"[Run] Final output: {final_output}")
    return final_output


def main() -> None:
    parser = argparse.ArgumentParser(description="Scene-intelligence evidence gap reconstruction.")
    parser.add_argument("--input_dir", default=str(ROOT / "data" / "input"))
    parser.add_argument("--output_dir", default=str(ROOT / "outputs"))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Inference/render config JSON.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--no_clean", action="store_true")
    parser.add_argument("--reuse_work", action="store_true", help="Reuse existing _work detections/chunks when present.")
    args = parser.parse_args()
    args.config_data = load_config(Path(args.config).resolve())

    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not args.no_clean:
        clean_outputs(output_dir)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(path for path in input_dir.iterdir() if path.suffix.lower() in VIDEO_EXTENSIONS)
    if not videos:
        raise FileNotFoundError(f"No input videos found in {input_dir}")

    rng = random.Random(args.seed)
    outputs = []
    print(f"[Run] Found {len(videos)} input video(s).")
    for video_path in videos:
        outputs.append(process_video(video_path, args, rng))

    print("\n[Run] Completed.")
    for output in outputs:
        print(f"[Run] {output}")


if __name__ == "__main__":
    main()
