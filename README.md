# AI Evidence Gap Reconstruction

Local pipeline for reconstructing a randomly missing quarter of every video in `data/input`.

## Goal

Given a full video, the system should:

1. Split the video into 4 equal time chunks.
2. Randomly hide 1 chunk on every run.
3. Keep the other 3 chunks as evidence footage, but overlay live YOLO classification boxes.
4. Analyze the visible 75% to understand the scene.
5. Reconstruct the missing 25% as an animated 3D-style inference view.
6. Write one reconstructed full-length video per input video.

This is not meant to be a random skeleton overlay or pasted ghost footage. The visible chunks should show live YOLO bounding boxes and class labels. The missing chunk should switch into an animated 3D reconstruction view driven by evidence gathered from the visible chunks: people, vehicles, bags, carried objects, movement direction, object continuity, likely paths, and scene layout.

## Expected Output

If `data/input` contains:

```text
input_vid3.mp4
input_vid4.mp4
```

then `python run.py` should produce:

```text
outputs/input_vid3_reconstructed.mp4
outputs/input_vid4_reconstructed.mp4
```

Intermediate files belong under:

```text
outputs/_work/<video_name>/
```

## Target Pipeline

The intended architecture is:

1. `run.py` finds every video in `data/input`.
2. Each video is split into 4 equal chunks.
3. One chunk is randomly selected as the hidden/missing evidence.
4. The other 3 chunks are passed through scene analysis and rendered with live YOLO boxes.
5. YOLOv8m detects people, cars, bikes, bags, backpacks, handbags, suitcases, and other relevant objects.
6. Tracking links detections across visible chunks.
7. The system writes an intelligence report:
   - how many people are visible
   - what objects they carry or interact with
   - vehicle/object presence
   - direction of movement
   - approximate paths before and after the gap
   - what likely happened during the missing chunk
8. The missing chunk is reconstructed as an animated 3D inference scene using those clues.
9. The final video shows YOLO-annotated evidence chunks plus the 3D missing-chunk reconstruction.

## Current Status

The active pipeline is scene-intelligence-first:

- model candidates are configured in `config/reconstruction_config.json`
- multiple object classes are analyzed, not only one person
- important entities are tracked across the visible 75%
- motion direction is inferred from tracks
- a structured reconstruction plan is written before rendering
- the hidden chunk is rendered as an animated 3D reconstruction view

## Run

```bash
python run.py
```

## YOLO Config

Inference settings live in:

```text
config/reconstruction_config.json
```

Current defaults:

```json
{
  "model": "yolo26m.pt",
  "frame_stride": 8,
  "downscale_width": 960,
  "confidence": 0.3
}
```

There is one active model in the config. Change it to `yolo11m.pt` or `yolov8m.pt` only if you want to test another model.

All 80 COCO classes are enabled in the config, including people, vehicles, bags, cup, bottle, knife, and cell phone. These are deliberately included for the future statement feature.

Confidence must be between `0` and `1`. If you write `3`, the runner treats it as `3%` and converts it to `0.03`.

Useful options:

```bash
python run.py --seed 123
python run.py --config config/reconstruction_config.json
```

The statement feature is not active yet. The next version should parse statements like "red shirt", "phone in hand", or "knife in other hand" and use those details to alter the 3D reconstruction plan.

## Setup

```bash
pip install -r requirements.txt
```

Lower-resolution input videos are preferred while developing. 720p is a good target because object detection, tracking, and rendering are much faster and more stable than 4K.
